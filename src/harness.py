"""Single-run gossip orchestrator, per PCTA_Experimental_Methodology.md B.4/B.6/B.8.

Entry point: run_experiment(...). No shared server base -- every node trains from
its OWN current weights each round (the fundamental topology difference from the
DBT-CFL-PTT star-topology harness this project reuses common.py/data.py from).
"""
import collections
import copy
import json
import math
import os
import time
import types

import numpy as np
import torch
import torch.nn.functional as Fnl
from scipy.stats import norm as norm_dist

from common import DEVICE, SmallCNN, get_flat_params, set_flat_params, num_params, set_seed
from data import load_dataset, dirichlet_partition, preload_client_tensors, preload_test_tensors
from topology import GossipTopology
from attacks import AttackState, apply_attack
from pcta_engine import (
    make_predictor, standardized_innovation_Z, raw_drift_Z, CUSUM, BetaReputation,
    CanaryProbe, chi2_quantile,
)
import defenses as dfx

MAX_ACTIVE_PAIRS = 2500   # LRU-bounded per-run cap on persisted PairState objects
                          # (~2.7MB/pair at Kalman+d~137K => ~6.8GB ceiling); a
                          # real deployed node wouldn't keep unbounded historical
                          # trust state forever either, so LRU eviction here is a
                          # legitimate design choice, not just a memory workaround
                          # -- see harness.py PairState docstring / Phase 5 OOM note.
# Revision v2 (2026-10-16, post desk-rejection): the submitted v1 config
# (4 batches, lr 0.02, no momentum, ETA 0.5, clip 1.0) left clean FedAvg at
# ~0.23 on CIFAR-10 alpha=0.1 -- every method compared in a barely-trained
# regime. diag_training.py sweep (results_diag_training.log) -> config E:
# clean CIFAR-10 0.61/0.51/0.39 at alpha 0.5/0.1/0.03, FMNIST 0.84 at 0.1.
# v1 code is archived in ../archive_submitted_v1/.
DELTA_CLIP_NORM = 10.0    # honest-node safety clip only (honest round-1 norms: median 2.7, max 8.5)
BALANCE_GAMMA = 1.0       # v2.1 (F19): calibrated on the dev config (grid 0.3/1/2/3/5, rule argmax mean(clean, sign-flip)); paper default 0.3 rejected ~100% of neighbors here
LOCAL_BATCHES = 20
BATCH_SIZE = 32
LOCAL_LR = 0.05
LOCAL_MOMENTUM = 0.9      # optimizer state is reset every round (no cross-round momentum)
GOSSIP_ETA = 1.0          # v1's 0.5 damping halved every SGD step
CLIP_BYZANTINE = False    # v1 clipped Byzantine deltas too -- an implicit defense a
                          # protocol-deviating attacker would never apply to itself
N_CAL = 6
PROBE_SIZE = 32
PROBE_TRIGGER_PROB = 0.05
PROBE_CAL_PERCENTILE = 99
ADAPTIVE_MARGIN = 1.15    # matches DBT-CFL-PTT's calibration-margin convention


LABEL_ATTACKS = ("label_flip", "label_flip_targeted")
FINAL_STATE_METHODS = ("clipped_gossip", "balance", "ios")   # revision v2 baselines (defenses.py)
DATA_CHECK_PER_CLASS = 30   # public reference set size per class for the data-integrity channel (v2;
                            # 30 > 10 improved targeted-flip AUROC at alpha=0.1, results_feasibility_C5_ref30)
# Revision v2 decision rule: textbook CUSUM on robust-standardized evidence, one
# accumulator per channel. Each observer standardizes a neighbor's statistic
# against the median/MAD of its OWN recent observations (all neighbors, appended
# unconditionally -- robust while the local Byzantine share stays < 1/2), then
# S <- max(0, S + z - CUSUM_K); alarm at S > CUSUM_H. Unlike v1's percentile-of-
# own-history thresholds (which flag a fixed share of honest peers BY
# CONSTRUCTION -- the ~30% FPR the editor cited), the honest alarm rate here is
# governed by CUSUM_H (in-control ARL), not forced by a quantile.
CUSUM_K = 1.0        # trajectory channel; k=0.5 let honest S creep up once trust memory
CUSUM_H = 5.0        # persisted (honest mean penalty 0.85 -> 0.96 at k=1.0, detection AUROC
                     # unchanged: offline sweep on results_smoke_v3, 2026-10-17)
# Data channel: two sub-CUSUMs, LEVEL (gain_max: does the update improve ANY
# class? catches sybil/sign-flip, AUROC .99/.90 offline) and CONTRAST (log-odds
# + dAUC: prior-shift-invariant, catches targeted label flips, AUROC .87) --
# complementary, so each gets its own accumulator rather than one blended
# score. Larger k tolerates the persistent per-node offsets honest Non-IID
# senders show on a shared reference set (offline sweep on results_smoke_v3:
# k=1.0,h=10 -> clean FPR ~7%; k=0.5,h=5 -> 31%).
CUSUM_K_DATA = 1.0
CUSUM_H_DATA = 10.0
Z_FLAG = 3.0          # per-round evidence counted as a reputation failure
Z_CLIP = 50.0         # bound a single round's influence on S
ROBUST_WINDOW = 300
ROBUST_MIN_HIST = 20
ROBUST_SEED_SIZE = 30   # F15: calibration values seeded per window (~3 rounds), only to cover warm-up; a full-window seed lagged convergence for ~30 rounds
WINDOW_GATE_PEN = 0.5   # F17: min last CUSUM penalty for a pair's evidence to enter the robust windows


def robust_z(x, window):
    """(x - median) / (1.4826 * MAD) over the observer's recent window; 0 until warm.
    Non-finite x is returned as +/-Z_CLIP (callers orient 'suspicious' as positive
    for the trajectory channel, and negate the data channel's)."""
    if not math.isfinite(x):
        return Z_CLIP if (x != x or x > 0) else -Z_CLIP
    if len(window) < ROBUST_MIN_HIST:
        return 0.0
    arr = np.asarray(window, dtype=float)
    med = float(np.median(arr))
    mad = 1.4826 * float(np.median(np.abs(arr - med))) + 1e-12
    return float(np.clip((x - med) / mad, -Z_CLIP, Z_CLIP))


def flip_labels(ys, attack_type):
    """label_flip: random wrong label per sample (v1 attack). label_flip_targeted
    (v2): fixed permutation c -> c+1, which yields a CONFIDENT wrong model --
    the harder case for any check based on prediction quality."""
    if attack_type == "label_flip_targeted":
        return (ys + 1) % 10
    return (ys + torch.randint(1, 10, ys.shape, device=ys.device)) % 10


def class_true_logp(model_template, flat_w, ref_x, ref_y, num_classes):
    """Per-class mean log-probability of the TRUE label on the reference set."""
    set_flat_params(model_template, flat_w)
    model_template.eval()
    with torch.no_grad():
        lp = Fnl.log_softmax(model_template(ref_x), dim=1)
        tl = lp.gather(1, ref_y.view(-1, 1)).squeeze(1)
    out = torch.zeros(num_classes, device=ref_x.device).index_add_(0, ref_y, tl)
    cnt = torch.bincount(ref_y, minlength=num_classes).clamp(min=1)
    return out / cnt


def ref_logp(model_template, flat_w, ref_x):
    set_flat_params(model_template, flat_w)
    model_template.eval()
    with torch.no_grad():
        return Fnl.log_softmax(model_template(ref_x), dim=1)


def data_integrity_scores(lp_base, lp_sent, ref_y, num_classes):
    """Data-integrity channel scores for one sent update (revision v2, Option C).
    delta[x, k] = change in log p(k|x) the update induces on the reference set.
      promo_k  = mean_x delta[x, k]                (how much the update promotes class k overall)
      disc_k   = mean_{x in k} delta[x, k] - mean_{x not in k} delta[x, k]
                 (does it promote k specifically on images OF k -- learning k's
                 features -- or everywhere, i.e. a prior shift?)
    An honest Non-IID update promotes its own classes AND discriminatively
    (disc > 0); a label-flipped one promotes a class mostly on images of a
    DIFFERENT class (disc < 0). Returns (gain_max, gain_sum, gain_min,
    disc_at_promo, disc_weighted)."""
    delta = lp_sent - lp_base
    onehot = Fnl.one_hot(ref_y, num_classes).float()
    n_in = onehot.sum(0).clamp(min=1)
    n_out = (1 - onehot).sum(0).clamp(min=1)
    in_k = (delta * onehot).sum(0) / n_in
    out_k = (delta * (1 - onehot)).sum(0) / n_out
    disc = in_k - out_k
    promo = delta.mean(0)
    w = torch.softmax(promo / max(float(promo.std()), 1e-6), dim=0)
    # Log-odds contrast (prior-shift invariant; log-prob saturates near p=1,
    # which biases disc negative for honest single-class nodes at severe Non-IID).
    def log_odds(lp):
        lse_other = torch.logsumexp(lp.unsqueeze(1).expand(-1, num_classes, -1)
                                    .masked_fill(torch.eye(num_classes, dtype=torch.bool,
                                                           device=lp.device).unsqueeze(0), float("-inf")), dim=2)
        return lp - lse_other
    dlo = log_odds(lp_sent) - log_odds(lp_base)
    disc_lo = (dlo * onehot).sum(0) / n_in - (dlo * (1 - onehot)).sum(0) / n_out
    # Ranking change: per-class AUC of log-odds separating class-k images from the rest.
    def class_auc(lo):
        r = lo.argsort(0).argsort(0).float()          # ranks within each class column
        pos_r = (r * onehot).sum(0)
        return (pos_r - n_in * (n_in - 1) / 2) / (n_in * n_out)
    dauc = class_auc(log_odds(lp_sent)) - class_auc(log_odds(lp_base))
    promo_lo = dlo.mean(0)
    w_lo = torch.softmax(promo_lo / max(float(promo_lo.std()), 1e-6), dim=0)
    return (float(in_k.max()), float(in_k.sum()), float(in_k.min()),
            float(disc[int(promo.argmax())]), float((w * disc).sum()),
            float((w_lo * disc_lo).sum()), float((w_lo * dauc).sum()), float(dauc.mean()))


def _new_model(in_channels, num_classes, input_size):
    return SmallCNN(in_channels, num_classes, input_size).to(DEVICE)


def weighted_percentile(values, weights, percentile):
    """Weighted quantile: the value v* such that the cumulative WEIGHT (not
    count) of observations <= v* reaches `percentile` of the total weight.
    Used by the adaptive-threshold mechanism's v8 design (see
    observer_s_history's docstring in run_experiment) to let each pair's
    trust score continuously scale its influence on h's calibration, rather
    than a binary include/exclude cutoff."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cum_w = np.cumsum(weights_sorted)
    total_w = cum_w[-1]
    if total_w <= 0:
        return float(np.median(values_sorted))
    cum_w_norm = cum_w / total_w
    idx = int(np.searchsorted(cum_w_norm, percentile / 100.0))
    idx = min(idx, len(values_sorted) - 1)
    return float(values_sorted[idx])


def local_train(model_template, base_weights, xs, ys, attack_type, is_byz, t, onset_round):
    """One node's local training step, from its OWN current weights (no shared
    server base). Label-flip corruption happens here, on the labels, before
    training -- not as a post-hoc delta corruption (attacks.py's design note)."""
    set_flat_params(model_template, base_weights)
    model_template.train()
    ys_used = ys
    if is_byz and attack_type in LABEL_ATTACKS and t >= onset_round:
        ys_used = flip_labels(ys, attack_type)
    opt = torch.optim.SGD(model_template.parameters(), lr=LOCAL_LR, momentum=LOCAL_MOMENTUM)
    n = xs.shape[0]
    if n == 0:
        return torch.zeros_like(base_weights)
    for _ in range(LOCAL_BATCHES):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), device=xs.device)
        opt.zero_grad()
        out = model_template(xs[idx])
        loss = Fnl.cross_entropy(out, ys_used[idx])
        loss.backward()
        opt.step()
    delta = get_flat_params(model_template) - base_weights
    return delta


def local_loss(model_template, weights, xs, ys):
    if xs.shape[0] == 0:
        return 0.0
    set_flat_params(model_template, weights)
    model_template.eval()
    with torch.no_grad():
        out = model_template(xs[: min(64, xs.shape[0])])
        loss = Fnl.cross_entropy(out, ys[: min(64, ys.shape[0])])
    return loss.item()


class PairState:
    """Per (observer, neighbor) trust state -- created lazily on first contact,
    persisted across the whole run (survives topology reshuffles, since a real
    system remembers past encounters with a peer identity; PCTA_Theory.md SS8's
    cold-start policy applies at first-ever contact, not at every reshuffle)."""

    # Per-pair trust state is sequential, elementwise, one pair at a time -- it
    # gets no benefit from GPU parallelism, unlike model training. With N*k up to
    # ~1000 active pairs and topology re-randomization causing the number of
    # DISTINCT pairs ever seen over a full run to run into the tens of thousands,
    # persisting this state on GPU (as the first implementation did) exhausts VRAM
    # (observed: CUDA OOM at 20+ GiB during the first full-scale timing probe,
    # Phase 5). Fix: this state lives on CPU RAM, which is far more abundant;
    # only the small per-round delta transfer crosses the PCIe bus.
    STATE_DEVICE = torch.device("cpu")

    def __init__(self, d, cfg):
        self.cfg = cfg
        self.predictor = make_predictor(cfg["predictor_kind"], d, device=self.STATE_DEVICE,
                                          **cfg["pred_kwargs"]) \
            if cfg["predictor_kind"] != "none" else None
        self.cusum = CUSUM(cfg["nu"], cfg["h"]) if cfg["use_cusum"] else None
        self.reputation = BetaReputation(
            alpha0=cfg["alpha0"], beta0=cfg["beta0"], theta=cfg["theta"], gamma=cfg["gamma"],
            probation_K=cfg["probation_K"], M_max=cfg["M_max"],
        )
        self.last_delta = None  # for the "none" predictor arm's raw-drift Z

    def compute_Z(self, observed_delta):
        """Phase 1 of the per-round adaptive-threshold mechanism (Methodology
        SS B.8's pre-registered fallback, actually implemented after a Phase 6
        post-mortem found the fixed-at-calibration nu/threshold going stale as
        training genuinely progresses -- honest FPR drifted to 9-44% across
        Non-IID severities instead of the ~1% design target). Advances the
        predictor and returns Z only; does NOT touch CUSUM/reputation yet,
        because the adaptive reference value needs EVERY neighbor's Z for this
        round computed first (see apply_decision and the pcta_soft loop in
        run_experiment)."""
        observed_delta = observed_delta.to(self.STATE_DEVICE)
        d = observed_delta.numel()
        if self.predictor is not None:
            pred, scale = self.predictor.predict()
            Z, _ = standardized_innovation_Z(pred, scale, observed_delta)
            self.predictor.observe(observed_delta)
        else:
            prev = self.last_delta if self.last_delta is not None else torch.zeros_like(observed_delta)
            Z, _ = raw_drift_Z(prev, observed_delta, scale=float(d))
            Z = Z * d  # rescale so it's on a roughly comparable footing to a chi2_d-scale threshold
            self.last_delta = observed_delta.clone()
        return Z

    def apply_decision(self, Z, adaptive_nu, adaptive_h, adaptive_thresh):
        """Phase 2: given this round's Z and the OBSERVER's freshly-computed
        per-round adaptive references, update CUSUM/reputation. Replaces the
        single fixed nu/h/chi2_thresh calibrated once at t~3-6 and held for the
        whole T=100 run. adaptive_h is required, not optional -- an earlier
        version of this method only ever adapted nu and left h frozen at its
        stale calibration value, which was the actual root cause behind FOUR
        consecutive failed adaptive-threshold designs (FPR 54-87% each time,
        Phase 6 post-mortem): once nu becomes a MEDIAN-like reference (as any
        working design requires), roughly half of all Z observations legitimately
        exceed it by design -- CUSUM's accumulator S is SUPPOSED to absorb that
        as ordinary noise, but only if h is recalibrated to match; a h frozen
        under the old regime gets crossed almost immediately regardless of any
        real attack."""
        if self.cusum is not None:
            self.cusum.nu = adaptive_nu
            self.cusum.h = adaptive_h
            S = self.cusum.update(Z)
            cusum_penalty = self.cusum.penalty()
            flagged = self.cusum.flagged
        else:
            S, cusum_penalty, flagged = 0.0, 1.0, (Z > adaptive_thresh)

        self.reputation.record(Z, adaptive_thresh)
        return S, cusum_penalty, flagged


def decide_v2(ps, z_t, z_dl, z_dc, use_cusum):
    """Revision v2 per-pair decision: one CUSUM for the trajectory channel (z_t)
    and two for the data channel (level z_dl, contrast z_dc); all oriented so
    'suspicious' is positive. Returns (S_t, S_dl, S_dc, penalty, flagged)."""
    if not hasattr(ps, "S_t"):
        ps.S_t, ps.S_dl, ps.S_dc = 0.0, 0.0, 0.0
    if use_cusum:
        ps.S_t = max(0.0, ps.S_t + z_t - CUSUM_K)
        ps.S_dl = max(0.0, ps.S_dl + z_dl - CUSUM_K_DATA)
        ps.S_dc = max(0.0, ps.S_dc + z_dc - CUSUM_K_DATA)
        load = ps.S_t / CUSUM_H + (ps.S_dl + ps.S_dc) / CUSUM_H_DATA
        flagged = ps.S_t > CUSUM_H or ps.S_dl > CUSUM_H_DATA or ps.S_dc > CUSUM_H_DATA
    else:  # no-CUSUM ablation: memoryless, per-round evidence only
        load = (max(0.0, z_t - CUSUM_K) / CUSUM_H
                + (max(0.0, z_dl - CUSUM_K_DATA) + max(0.0, z_dc - CUSUM_K_DATA)) / CUSUM_H_DATA)
        flagged = z_t > Z_FLAG or z_dl > Z_FLAG or z_dc > Z_FLAG
    return ps.S_t, ps.S_dl, ps.S_dc, math.exp(-load), flagged


class LRUPairCache:
    """Bounds the number of simultaneously-persisted PairState objects (harness.py
    Phase 5 OOM note above PairState). Evicts the least-recently-touched pair when
    over capacity -- a fresh re-encounter with an evicted peer starts cold again
    (Section 8's cold-start policy), which is the correct, honest behavior for a
    memory-bounded real deployment, not a simulation shortcut."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._d = collections.OrderedDict()

    def get_or_create(self, key, factory):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        if len(self._d) >= self.capacity:
            self._d.popitem(last=False)
        val = factory()
        self._d[key] = val
        return val

    def peek(self, key, default=None):
        return self._d.get(key, default)


def trust_weight(pair_state, cusum_penalty, use_reputation):
    if not use_reputation:
        return 0.0 if pair_state.cusum is not None and pair_state.cusum.flagged else 1.0
    return pair_state.reputation.soft_mask() * cusum_penalty


def run_experiment(dataset, method, attack, byz_fraction, N=100, k=10, T=100, R=5,
                    placement="uniform", alpha=0.1, seed=0, variant=None, out_dir="results_full",
                    tag=None, attack_params=None, quiet=True, verbose_timing=False):
    """method in {fedavg, local_krum, trimmed_mean, median, foolsgold, fltrust,
    openclaw, pcta_soft}. variant (pcta_soft only): dict overriding predictor_kind
    (kalman/ema/none), use_cusum, use_probe, use_reputation, cold_start_fix (bool).
    verbose_timing: print per-phase/per-round wall-clock, for diagnosing where time
    goes at full scale (Phase 5 timing-probe diagnostic)."""
    t0 = time.time()
    set_seed(seed)
    variant = variant or {}
    attack_params = attack_params or {}

    train_ds, test_ds, num_classes, in_ch, input_size = load_dataset(dataset)
    client_idxs = dirichlet_partition(train_ds, N, alpha, num_classes, seed=seed)
    clients = preload_client_tensors(train_ds, client_idxs, DEVICE)
    test_x, test_y = preload_test_tensors(test_ds, DEVICE, max_n=3000, seed=seed)

    model_template = _new_model(in_ch, num_classes, input_size)
    d = num_params(model_template)

    # Trust computation operates on a fixed random coordinate subsample of DeltaW,
    # not the full d~1e5-dim vector -- PCTA_Theory.md SS2.1's low-dimensional-
    # projection option, now load-bearing rather than optional: at full N=100,k=10
    # scale, persisting full-d Kalman state per (observer,neighbor) pair churns
    # through tens of thousands of distinct pairs over a run (topology
    # re-randomization), making full-d PairState creation the dominant per-round
    # cost (Phase 5 timing diagnostic: ~7-16s/round, dominated by pair creation,
    # not model training). AGGREGATION still uses the full delta -- trust_weight
    # is a scalar computed from the subsample, applied to the full ΔW. A random
    # coordinate subsample of a high-dim vector is a standard, valid way to reduce
    # dimensionality for a chi-squared-type statistic (sketching).
    TRUST_DIM = min(2000, d)
    trust_idx = torch.randperm(d, generator=torch.Generator().manual_seed(seed))[:TRUST_DIM].to(DEVICE)

    rng = np.random.default_rng(seed + 777)
    n_byz = int(round(byz_fraction * N))
    byz_ids = set(rng.choice(N, size=n_byz, replace=False).tolist()) if n_byz > 0 else set()

    # Shared initial model, broadcast to every node once at t=0 -- NOT the same
    # thing as DBT-CFL-PTT's per-round shared server base (B.4's "no shared base"
    # design is about round-to-round local training: every node trains from its
    # OWN evolving weights each round, never resyncing to a common base, which is
    # the real gossip-learning difference and is preserved below). An earlier
    # version of this harness also gave every node an INDEPENDENT random init at
    # t=0, which silently broke learning entirely (confirmed empirically: clean
    # FedAvg, zero attacks, stayed flat at ~10%/chance-level accuracy for all 100
    # rounds -- a Phase 6 post-mortem finding). Root cause: parameter-space
    # averaging of independently-initialized neural networks is fundamentally
    # broken (permutation/symmetry misalignment -- different random inits learn
    # different internal orderings of hidden units, so averaging deltas computed
    # from unrelated starting points produces noise, not a useful direction).
    # This is exactly why every real FedAvg-family method, including fully
    # decentralized ones, broadcasts a shared init once at t=0 even when there is
    # no shared base thereafter.
    node_weights = {i: get_flat_params(model_template).clone() for i in range(N)}

    topology = GossipTopology(N, k, R, seed=seed, byz_ids=byz_ids, placement=placement)

    attack_state = AttackState(attack, byz_ids, d, seed=seed + 1, params=attack_params)

    # ---- Data-integrity channel reference set (revision v2, Option C) ----
    # A small labeled public reference set (DATA_CHECK_PER_CLASS per class),
    # the same kind of shared-data assumption the probe batch above already makes.
    _tg = np.array(train_ds.targets)
    ref_idx = np.concatenate([rng.choice(np.where(_tg == c)[0], size=DATA_CHECK_PER_CLASS, replace=False)
                              for c in range(num_classes)])
    ref_x = torch.stack([train_ds[int(i)][0] for i in ref_idx]).to(DEVICE)
    ref_y = torch.tensor([int(_tg[i]) for i in ref_idx], dtype=torch.long).to(DEVICE)

    # ---- Calibration (B.8): N_CAL clean rounds, attacks force-disabled, discarded after ----
    cal_topology = GossipTopology(N, k, R, seed=seed + 2, byz_ids=set(), placement="uniform")
    cal_weights = {i: node_weights[i].clone() for i in range(N)}
    cal_Zs = []
    # PASS 1: estimate the actual per-coordinate delta variance from raw honest
    # deltas, BEFORE any predictor exists. Every predictor's scale (init_var, q, r)
    # must be calibrated to this real scale -- a hardcoded default (e.g. init_var=1.0)
    # is wildly mismatched to real SGD-step-sized deltas under DELTA_CLIP_NORM=1.0
    # spread over d~1e5 coordinates (per-coordinate variance ~1e-5, not ~1), which
    # silently starves the CUSUM/Kalman channel of any signal (Z collapses to a tiny
    # fraction of d, nu is never crossed) even though the probe channel still works --
    # exactly the failure mode PCTA_Experimental_Methodology.md B.8's pre-registered
    # diagnostic was built to catch, and did catch during Phase 5 smoke-testing.
    var_accum = torch.zeros(1, device=DEVICE)
    var_n = 0
    for t in range(1, 3):
        for i in range(N):
            xs, ys = clients[i]
            delta_i = local_train(model_template, cal_weights[i], xs, ys, "none", False, t, 999)
            n_ = torch.norm(delta_i)
            if n_ > DELTA_CLIP_NORM:
                delta_i = delta_i * (DELTA_CLIP_NORM / n_)
            var_accum += torch.sum(delta_i * delta_i)
            var_n += delta_i.numel()
    mean_var = max(float((var_accum / max(var_n, 1)).item()), 1e-10)
    if verbose_timing:
        print(f"  [t={time.time()-t0:.1f}s] variance-estimation pass done, mean_var={mean_var:.2e}")

    # PASS 2 (remaining calibration rounds): predictors now use the real scale.
    cal_predictor_kind = "kalman"
    cal_pred_kwargs = dict(q=0.05 * mean_var, r=mean_var, init_var=mean_var)
    cal_pairs = {}
    cal_dgain = []
    for t in range(3, N_CAL + 1):
        deltas = {}
        for i in range(N):
            xs, ys = clients[i]
            deltas[i] = local_train(model_template, cal_weights[i], xs, ys, "none", False, t, 999)
            n_ = torch.norm(deltas[i])
            if n_ > DELTA_CLIP_NORM:
                deltas[i] = deltas[i] * (DELTA_CLIP_NORM / n_)
        for i in range(N):
            for j in cal_topology.neighbors(i):
                key = (i, j)
                if key not in cal_pairs:
                    cal_pairs[key] = make_predictor(cal_predictor_kind, TRUST_DIM, device=PairState.STATE_DEVICE,
                                                      **cal_pred_kwargs)
                pred = cal_pairs[key]
                p, s = pred.predict()
                delta_j_cpu = deltas[j][trust_idx].to(PairState.STATE_DEVICE)
                Z, _ = standardized_innovation_Z(p, s, delta_j_cpu)
                pred.observe(delta_j_cpu)
                cal_Zs.append(Z.item())
        for i in range(N):   # v2: honest data-channel scores, to seed the robust windows
            cal_dgain.append(data_integrity_scores(ref_logp(model_template, cal_weights[i], ref_x),
                                                   ref_logp(model_template, cal_weights[i] + deltas[i], ref_x),
                                                   ref_y, num_classes))
        new_w = {}
        for i in range(N):
            nbrs = cal_topology.neighbors(i)
            agg = torch.stack([deltas[j] for j in nbrs]).mean(dim=0)
            new_w[i] = cal_weights[i] + GOSSIP_ETA * agg
        cal_weights = new_w
        cal_topology.maybe_reshuffle(t)
        if verbose_timing:
            print(f"  [t={time.time()-t0:.1f}s] calibration round {t}/{N_CAL} done")

    cal_Zs = np.array(cal_Zs) if len(cal_Zs) > 0 else np.array([float(TRUST_DIM)])
    lambda_honest_est = max(0.0, float(np.mean(cal_Zs)) - TRUST_DIM)
    nu = TRUST_DIM + lambda_honest_est + 5.0  # Delta/2 = 5.0 design slack, per PCTA_Theory.md SS3
    h = float(np.percentile(cal_Zs - nu, 90)) * ADAPTIVE_MARGIN if len(cal_Zs) > 5 else 50.0
    h = max(h, 10.0)
    # nu/h above are the INITIAL calibration-time values (used only to construct
    # each PairState's CUSUM/predictor and logged in `result` for reference) --
    # at runtime, nu and the reputation threshold are recomputed every round from
    # each observer's own current-neighbor Z distribution (the adaptive-threshold
    # fallback, see PairState.apply_decision and the pcta_soft loop below).
    # chi2_thresh (the theoretical, non-adaptive TRUST_DIM-quantile) is kept only
    # as a documented reference value, not used in any live decision.
    chi2_thresh = chi2_quantile(TRUST_DIM, 0.99)

    pcta_cfg = dict(
        predictor_kind=variant.get("predictor_kind", "kalman"),
        pred_kwargs=(dict(q=0.05 * mean_var, r=mean_var, init_var=mean_var)
                     if variant.get("predictor_kind", "kalman") == "kalman"
                     else dict(beta=0.3, beta_v=0.2, init_var=mean_var)),
        use_cusum=variant.get("use_cusum", True),
        use_probe=variant.get("use_probe", False),   # v2: Canary Probe replaced by the data-integrity channel
        decision=variant.get("decision", "v2"),        # "v1" = submitted percentile-threshold rule
        use_traj=variant.get("use_traj", True),        # trajectory channel (predictor + innovation Z)
        use_data=variant.get("use_data", True),        # data-integrity channel
        identity_age_probation=variant.get("identity_age_probation", True),
        use_reputation=variant.get("use_reputation", True),
        nu=nu, h=h,
        alpha0=1.0 if variant.get("cold_start_fix", True) else 1.0,
        beta0=variant.get("beta0", 2.0 if variant.get("cold_start_fix", True) else 1.0),
        theta=0.3, gamma=0.05,
        # v2: probation shortened 5 -> 2 interactions. v1 lifted the cap early on a
        # passed probe; without the probe, K=5 equals one full R=5 encounter, so
        # nearly every honest pair spent its whole first encounter capped (mixing
        # loss). Whitewashing damage bound M_max*K: 1.0 -> 0.4 per fresh identity.
        probation_K=variant.get("probation_K", 2 if variant.get("cold_start_fix", True) else 0),
        M_max=0.2 if variant.get("cold_start_fix", True) else 1.0,
        # v2 (2026-10-17, triggered by the held-out check, F14): absolute trust
        # mixing W_ij = tw_j/(k+1), untrusted mass stays on self. False = v1's
        # renormalized sum_j tw_j s_j / sum_j tw_j, where trust is only relative.
        residual_self=variant.get("residual_self", True),
        # F17: a pair's evidence enters the observer's robust windows only while that
        # pair's last CUSUM penalty >= WINDOW_GATE_PEN, so suspected peers cannot
        # drag the reference median/MAD toward themselves (Sybil contamination).
        gate_windows=variant.get("gate_windows", True),
        # F18: clip each neighbor's deviation s_j - own to median + Z_FLAG*MAD of this
        # round's neighbor deviation norms (bounds an unclipped attacker's leverage
        # before the CUSUM has accumulated, e.g. sign-flip under alpha=0.03).
        norm_bound=variant.get("norm_bound", True),
    )

    # Per-observer adaptive-threshold state (PCTA-Soft only).
    #
    # Design history (three failed parametric attempts, kept here so the
    # reasoning isn't lost -- do not repeat these):
    # v1 used median(this round's k=10 neighbor Z's) * 1.15 as the per-round
    # reputation threshold -- FPR 61.7% (worse than the 24.5% staleness bug it
    # was meant to fix). nu (a mean-like CUSUM reference, tolerant of
    # single-round noise) and the reputation threshold (a one-shot decision
    # needing a HIGH percentile) are different statistics and can't share one
    # formula.
    # v2 used median/MAD across an observer's k=10 neighbors within ONE round,
    # EMA-smoothed over rounds -- FPR 54-78%. Cross-neighbor variation within a
    # round conflates genuine Non-IID heterogeneity (which Z is already
    # supposed to absorb per-pair) with predictor immaturity after a reshuffle,
    # and k=10 samples is far too few for a stable MAD estimate regardless.
    # v3 used a per-observer running EMA (mean + K*mean-abs-deviation),
    # accumulated across all of an observer's interactions over time, updated
    # only from non-flagged observations -- FPR 79-87%, WORSE still. Direct
    # instrumentation of the raw Z stream revealed the actual root cause behind
    # all three failures: Z's empirical distribution is nowhere near the
    # low-relative-variance, roughly-symmetric shape a chi2_d statistic would
    # have (observed honest Z spanning ~60-440, i.e. relative spread comparable
    # to the mean itself, not the ~3% relative std a true chi2_2000 would show)
    # -- so no parametric mean+k*scale formula was ever going to work, and
    # "exclude flagged observations from the running calibration" additionally
    # creates a self-narrowing feedback loop (excluding the upper tail from
    # ever informing the scale estimate biases that estimate down over time,
    # which flags more, which narrows further).
    #
    # v4 (current): non-parametric. Track each observer's raw Z history
    # (bounded sliding window) and use the EMPIRICAL percentile directly --
    # no distributional shape assumption needed, and a 99th-percentile
    # threshold gives ~1% FPR on a population that's mostly honest BY
    # CONSTRUCTION, regardless of the underlying distribution's shape. Every
    # observation is appended unconditionally (avoiding v3's self-narrowing
    # bias); a 99th percentile stays dominated by honest observations as long
    # as the LOCAL Byzantine fraction in an observer's own recent history stays
    # meaningfully under 50%, which PCTA_Theory.md SS4's Eventual Local Balance
    # guarantees in expectation.
    #
    # v4 ALSO only ever adapted nu, leaving h (the CUSUM flagging threshold)
    # frozen at its stale calibration value -- the ACTUAL root cause behind all
    # four attempts (FPR 54-87% every time). Once nu is a median-like reference,
    # roughly half of all Z's legitimately exceed it; CUSUM's whole design
    # point is that its accumulator S absorbs that as ordinary noise, but only
    # if h is recalibrated to the new regime too. Fixed (v5, current) by ALSO
    # tracking each observer's own history of post-update S values (not Z) and
    # setting h as a high percentile of THAT history -- matching
    # PCTA_Experimental_Methodology.md B.9's description of the DBT-CFL-PTT
    # precedent exactly ("h_cusum via the CUSUM-statistic S_t, not the raw
    # signal"), which this project's own methodology wrote down correctly but
    # never actually implemented until now.
    #
    # v5's own first attempt gated s_hist's append on "not flagged" -- confirmed
    # WORSE (FPR back to 83-84%, even with ZERO attackers present). Root cause:
    # the SAME self-narrowing feedback loop as v3, just relocated -- whenever a
    # percentile-of-surviving-observations threshold is even slightly too
    # tight (by chance, by bootstrap noise, or from a genuine false positive),
    # excluding "flagged" observations from its OWN calibration data removes
    # exactly the upper tail the percentile needs to characterize itself,
    # making the estimate tighter, flagging more, excluding more -- unstable
    # regardless of whether the excluded quantity is a location (v3) or a
    # high-percentile scale-like statistic (this). The general lesson: never
    # gate a threshold's calibration data on that SAME threshold's own
    # decisions. Fixed (v6) with a FIXED, decision-independent trim instead --
    # append every S unconditionally, then compute h from a high percentile of
    # only the bottom (1-TRIM_FRAC) of the sorted history, discarding the top
    # TRIM_FRAC unconditionally regardless of flag status. This avoided the
    # feedback loop (confirmed: clean FPR down from 83-84% to 45-55%, TPR up
    # to 84-99.9%) but OVER-corrected: a fixed 30% trim discards real, valid
    # calibration data even with ZERO attackers present, since there is
    # nothing to trim away in that regime -- makes the percentile-of-trimmed
    # data systematically SMALLER than the true honest-population percentile,
    # over-flagging regardless of any actual threat. Using the ground-truth
    # byz_fraction to size the trim instead would fix this but is invalid --
    # DBT-CFL-PTT's own methodology (and this project's) requires ground truth
    # never be available to any defense's decision logic, only for evaluation.
    # Fixed (v7) by gating s_hist inclusion on each PAIR's REPUTATION score
    # (a hard cutoff, R_hat >= 0.5) instead of a fixed trim fraction or the
    # CUSUM decision itself -- reputation (Definition 3', Z vs. adaptive_thresh,
    # evolving slowly via Beta counters) is a genuinely SEPARATE signal from the
    # S-vs-h comparison h itself calibrates, so gating on it doesn't recreate
    # v3/v6's direct self-reference. Confirmed a real, if modest, improvement
    # (clean FPR 25-34%, down from v6's 83-84% and v7's blind-trim 45-55%; TPR
    # 75-98%) -- better than the original fixed-threshold problem's 9-44% FPR
    # range, but not a decisive fix, and a HARD 0.5 cutoff sits awkwardly next
    # to PCTA's own design philosophy (Definition 11', Table 3.3's "sigmoid
    # soft-masking" -- continuous graduated trust, explicitly framed as
    # superior to binary accept/reject everywhere else in the architecture).
    # v8-continuous tried replacing the hard cutoff with that same soft-mask as
    # a WEIGHT on each S observation's contribution to h's calibration
    # (weighted_percentile() below), used the reputation-only component
    # intentionally (not the full trust_weight(), which also folds in the
    # CUSUM penalty derived from S/h itself -- that would reintroduce the
    # self-reference this whole design line has been trying to avoid). It DID
    # improve FPR further (clean 17-19%, attacked 4-15%, both the best of any
    # attempt) -- but at real cost to detection: TPR fell to 42-59% (from
    # v7's 75-98%). Root cause: an attacker's reputation only drops AFTER
    # detection begins, so during a fresh attack's early rounds its high
    # initial weight still lets its inflated S values into calibration near
    # full-strength, transiently inflating h exactly when sensitivity matters
    # most -- worst for slow, ramping attacks (slow_poison: 42% TPR, the
    # longest inflation window) and less bad for sudden ones (sign_flip/
    # sybil: ~59%). A genuine specificity/sensitivity trade-off, not a strict
    # improvement over v7.
    #
    # LOCKED (2026-09-01): v7's hard cutoff (R_hat >= 0.5, binary
    # include/exclude, plain unweighted percentile) is the final design --
    # explicitly chosen over v8-continuous because keeping TPR high matters
    # more here than the marginal FPR gain, given Byzantine detection is this
    # paper's core claim and Non-IID false-positive control, while important,
    # is the secondary property being validated in this specific experiment.
    obs_win_traj, obs_win_lo, obs_win_au, obs_win_g = {}, {}, {}, {}   # v2 per-observer robust windows
    # v2: SCALAR trust memory (CUSUM sums + Beta reputation) persists for every
    # (observer, sender) pair for the whole run. v1 kept it inside the LRU-capped
    # PairState, and with N=100, k=10, R=5 the 2500-pair cap evicts most pairs
    # before they ever meet again -- each pair's evidence was discarded after <=5
    # observations, i.e. the architecture's temporal memory was effectively off.
    # Only the large per-pair Kalman predictor state (TRUST_DIM vectors) stays LRU-bounded.
    trust_mem = {}
    join_round = {j: 0 for j in range(N)}   # all identities join at t=0; attack_params['whitewash_every'] re-joins Byzantine identities

    def _new_trust_mem():
        m = types.SimpleNamespace(S_t=0.0, S_dl=0.0, S_dc=0.0, pen=1.0)   # pen: last CUSUM penalty (F17 gate)
        m.reputation = BetaReputation(alpha0=pcta_cfg["alpha0"], beta0=pcta_cfg["beta0"],
                                      theta=pcta_cfg["theta"], gamma=pcta_cfg["gamma"],
                                      probation_K=pcta_cfg["probation_K"], M_max=pcta_cfg["M_max"])
        return m
    # v2: seed every observer's windows with HONEST calibration-phase statistics,
    # so detection is live from round 1 (an unseeded window needs ~2 rounds to
    # warm up, during which an unclipped attacker can already diverge the
    # network -- observed in results_smoke_v3: sign-flip collapsed PCTA to NaN).
    _seed_rng = np.random.default_rng(seed + 31)
    _cz = np.log(np.maximum(cal_Zs, 1e-12))
    seed_traj = list(_seed_rng.choice(_cz, size=min(ROBUST_SEED_SIZE, len(_cz)), replace=False))
    _cd = np.array(cal_dgain) if cal_dgain else np.zeros((1, 8))
    _pick = _seed_rng.choice(len(_cd), size=min(ROBUST_SEED_SIZE, len(_cd)), replace=False)
    seed_lo, seed_au, seed_g = list(_cd[_pick, 5]), list(_cd[_pick, 6]), list(_cd[_pick, 0])
    def _seeded(seed_vals):
        return lambda: collections.deque(seed_vals, maxlen=ROBUST_WINDOW)
    observer_z_history = {}        # i -> deque(maxlen=HISTORY_LEN) of raw Z floats
    observer_s_history = {}        # i -> deque(maxlen=HISTORY_LEN) of post-update CUSUM S floats
    HISTORY_LEN = 300
    MIN_HISTORY_FOR_ADAPTIVE = 20   # fall back to the calibration-time nu/h/chi2_thresh until then
    NU_PERCENTILE = 50
    THRESH_PERCENTILE = 99
    H_PERCENTILE = 90               # matches DBT-CFL-PTT's own h_cusum convention
    S_HIST_MIN_REPUTATION = 0.5     # only pairs at/above this R_hat contribute to h's calibration

    # ---- Probe dataset + judge machinery (Definitions 6'/7') ----
    probe_idx = rng.choice(len(train_ds), size=PROBE_SIZE, replace=False)
    probe_x = torch.stack([train_ds[int(i)][0] for i in probe_idx]).to(DEVICE)
    probe_y = torch.tensor([train_ds[int(i)][1] for i in probe_idx], dtype=torch.long).to(DEVICE)

    def model_fn(flat_w):
        set_flat_params(model_template, flat_w)
        return model_template

    probe = CanaryProbe(probe_x, probe_y, delta_probe=1.0, model_fn=model_fn,
                         loss_fn=Fnl.cross_entropy)
    # Revision v2: calibrate the (relative) probe threshold from honest probes
    # on the clean calibration-phase models, same 99th-percentile x margin
    # convention as the other calibrated thresholds -- no ground truth used.
    cal_probe_d = []
    for i in range(N):
        nb = list(cal_topology.neighbors(i))
        if len(nb) < 4:
            continue
        cal_probe_d.append(probe.honest_distance(cal_weights[nb[0]], [cal_weights[x] for x in nb[1:4]]))
    if cal_probe_d:
        probe.delta_probe = float(np.percentile(cal_probe_d, PROBE_CAL_PERCENTILE)) * ADAPTIVE_MARGIN


    # ---- Per-method persistent state ----
    pair_states = LRUPairCache(MAX_ACTIVE_PAIRS)   # (i,j) -> PairState, pcta_soft only
    foolsgold_states = {i: dfx.FoolsGoldState() for i in range(N)}
    openclaw_states = {i: dfx.OpenCLAWReputationState() for i in range(N)}
    root_idx = rng.choice(len(train_ds), size=64, replace=False)   # draw kept so the rng stream is unchanged
    # v2.1 (2026-10-20): FLTrust's root set = the SAME public reference set PCTA's
    # data channel uses (identical public-data assumption for both methods).
    root_x, root_y = ref_x, ref_y

    f_local_assumed = max(0, int(round(byz_fraction * k)))

    detect_log = []
    acc_log = []
    probe_log = []

    whitewash_every = attack_params.get("whitewash_every", 0)
    for t in range(1, T + 1):
        topology.maybe_reshuffle(t)
        # v2 whitewashing (F16): every `whitewash_every` rounds each Byzantine node
        # rejoins under a FRESH identity -- every observer's memory of it (trust
        # CUSUMs, reputation, predictor) is lost and its identity age restarts.
        if whitewash_every and t > 1 and (t - 1) % whitewash_every == 0:
            for j in byz_ids:
                join_round[j] = t - 1
                for key in [key for key in trust_mem if key[1] == j]:
                    del trust_mem[key]
                for key in [key for key in pair_states._d if key[1] == j]:
                    del pair_states._d[key]

        raw_deltas = {}
        own_deltas = {}   # what each node applies to ITS OWN model (revision v2, see below)
        for i in range(N):
            xs, ys = clients[i]
            d_i = local_train(model_template, node_weights[i], xs, ys, attack, i in byz_ids, t, 1)
            # Revision v2: a Byzantine node keeps an honestly-trained internal
            # model and corrupts only what it SENDS. v1 applied the corrupted
            # delta to the attacker's own state, which (once Byzantine deltas
            # are no longer clipped) diverges geometrically -- and because
            # neighbors mix STATES (node_weights[j] + delta_j) while defenses
            # score DELTAS, the harm then travelled through a channel no
            # defense inspects. Keeping the attacker's base honest makes the
            # transmitted delta the entire attack surface.
            if i in byz_ids:
                own_deltas[i] = (local_train(model_template, node_weights[i], xs, ys, "none", False, t, 1)
                                 if attack in LABEL_ATTACKS else d_i.clone())
            if attack == "calibrated_adaptive" and i in byz_ids and t >= 1:
                # Root-cause fix for the Detectability Floor figure bug (Phase 6
                # post-mortem, 2026-09-03): every ratio 0.5-5.0 originally
                # produced byte-identical results, then (after a first fix
                # attempt) Z values in the MILLIONS with eventual NaN. Full
                # diagnosis and the final fix (delta-norm-relative injection,
                # matching slow_poison/sybil's proven convention -- no
                # per-coordinate scale reconstruction needed) live in
                # attacks.py's calibrated_adaptive branch; only the raw ratio
                # parameter needs passing through here now.
                attack_state.params["lambda_target"] = attack_params.get("lambda_target", 0.0)
            d_i = apply_attack(d_i, i, t, attack_state, onset_round=1)
            # Data-poisoning attackers (LABEL_ATTACKS) follow the protocol --
            # including the clip -- and poison only their labels; unclipped
            # garbage would be trivially detectable and is not that threat model.
            if CLIP_BYZANTINE or i not in byz_ids or attack in LABEL_ATTACKS:
                n_ = torch.norm(d_i)
                if n_ > DELTA_CLIP_NORM:
                    d_i = d_i * (DELTA_CLIP_NORM / n_)
            raw_deltas[i] = d_i
            if i not in byz_ids:
                own_deltas[i] = d_i
            else:
                n_ = torch.norm(own_deltas[i])
                if n_ > DELTA_CLIP_NORM:
                    own_deltas[i] = own_deltas[i] * (DELTA_CLIP_NORM / n_)

        # Omniscient crafted attacks (revision v2): every Byzantine node sends the
        # same delta built from ALL honest deltas' statistics this round.
        if attack in ("alie", "ipm") and byz_ids:
            H = torch.stack([raw_deltas[x] for x in range(N) if x not in byz_ids])
            mu = H.mean(dim=0)
            if attack == "alie":
                # Baruch et al. 2019: mu - z*sigma, z = Phi^-1((n - s)/n), s = floor(n/2 + 1) - f
                n_, f_ = N, len(byz_ids)
                s_ = math.floor(n_ / 2 + 1) - f_
                z_alie = attack_params.get("z", float(norm_dist.ppf((n_ - s_) / n_)))
                mal = mu - z_alie * H.std(dim=0)
            else:
                # Xie et al. 2019 inner-product manipulation: -eps * mu
                mal = -attack_params.get("eps", 0.5) * mu
            del H
            for b in byz_ids:
                raw_deltas[b] = mal.clone()

        # Data-integrity channel (v2): per-class change in true-label
        # log-likelihood that node j's SENT update induces on the reference
        # set, gain_j = logp(x_j + delta_j) - logp(x_j). An honest update,
        # however Non-IID, improves the true label on the classes it trained
        # on; a data-poisoned one cannot. Observer-independent here (shared
        # reference set), so computed once per sender per round.
        dgain = {}
        for j in range(N):
            dgain[j] = data_integrity_scores(ref_logp(model_template, node_weights[j], ref_x),
                                             ref_logp(model_template, node_weights[j] + raw_deltas[j], ref_x),
                                             ref_y, num_classes)

        new_weights = {}
        for i in range(N):
            nbrs = topology.neighbors(i)
            own_updated = node_weights[i] + own_deltas[i]  # commit i's own local SGD step
            if not nbrs:
                new_weights[i] = own_updated
                continue
            neighbor_deltas = {j: raw_deltas[j] for j in nbrs}
            # Aggregation mixes STATES (node_weights[j] + delta_j), not raw deltas.
            # Root-cause fix for a Phase 6 post-mortem finding: mixing deltas
            # computed from DIFFERENT, independently-diverging base points is not
            # a valid operation once nodes' trajectories have drifted apart (unlike
            # FedAvg/DBT-CFL-PTT's star topology, where every client's delta is
            # computed from the SAME shared W_base each round, making delta-
            # averaging valid). Standard decentralized SGD (Koloskova et al. 2020,
            # Lian et al. 2017 -- both cited in PCTA_Theory.md SS7) mixes NEIGHBOR
            # STATES with one's own post-gradient-step state; this restores that.
            # Weighting logic (w[j], computed from `neighbor_deltas` for similarity/
            # trust scoring) is unchanged -- only what gets combined at the end
            # changes from delta to state. Confirmed empirically: without this,
            # even clean FedAvg with zero attacks and zero self-discard stayed flat
            # at chance-level accuracy for 100 rounds.
            neighbor_states = {j: node_weights[j] + neighbor_deltas[j] for j in nbrs}
            dg = dgain
            if attack == "dissensus":
                # He, Karimireddy & Jaggi 2022: a per-RECEIVER message that cancels
                # (eps > 1: reverses) the honest neighbors' pull on node i:
                #   x_b -> x_i - eps * sum_{honest l} (x_l - x_i) / |byz in N(i)|
                hon = [l for l in nbrs if l not in byz_ids]
                byz = [l for l in nbrs if l in byz_ids]
                if hon and byz:
                    pull = torch.stack([neighbor_states[l] - own_updated for l in hon]).sum(dim=0)
                    eps_d = attack_params.get("eps", 1.0)
                    dg = dict(dgain)
                    for b in byz:
                        st = own_updated - eps_d * pull / len(byz)
                        neighbor_states[b] = st
                        neighbor_deltas[b] = st - node_weights[b]
                        dg[b] = data_integrity_scores(ref_logp(model_template, node_weights[b], ref_x),
                                                      ref_logp(model_template, st, ref_x), ref_y, num_classes)

            if method == "fedavg":
                w = dfx.fedavg(neighbor_deltas)
                agg = sum(w[j] * neighbor_states[j] for j in nbrs)

            elif method == "local_krum":
                w = dfx.local_krum(neighbor_deltas, f_local_assumed)
                agg = sum(w[j] * neighbor_states[j] for j in nbrs)

            elif method in ("trimmed_mean", "median"):
                _, agg = dfx.trimmed_mean_weights(neighbor_states, f_local_assumed,
                                                    use_median=(method == "median"))
                w = {j: 1.0 / len(nbrs) for j in nbrs}  # for logging only

            elif method == "foolsgold":
                w = foolsgold_states[i].weights(neighbor_deltas)
                agg = sum(w[j] * neighbor_states[j] for j in nbrs)

            elif method == "fltrust":
                # Root-cause fix for a Phase 6 post-mortem finding: FLTrust was
                # completely non-functional (flat at chance-level accuracy even
                # with zero attacks, confirmed via acc_log). Bug: a bare
                # loss.backward() gradient points in the ASCENT direction and is
                # unnormalized/unscaled, while neighbor_deltas are actual SGD
                # DESCENT steps (opposite sign, and scaled by lr * multiple local
                # batches). Comparing them via cosine similarity as the original
                # FLTrust paper (Cao et al. 2021) intends means the "server update"
                # must be an actual local SGD delta on the root dataset -- same
                # procedure, same sign/scale convention as every client delta --
                # not a raw one-shot gradient. Fixed by running local_train() on
                # the root batch exactly like any client's local step.
                root_delta = local_train(model_template, node_weights[i], root_x, root_y,
                                          "none", False, t, 999)
                # v2.1: same honest safety clip as every other honest update. Unclipped,
                # a root step that blew up on one node (norm 139) was copied onto ALL
                # neighbor updates by FLTrust's norm rescaling -> NaN in every run.
                _rn = float(torch.norm(root_delta))
                if _rn > DELTA_CLIP_NORM:
                    root_delta = root_delta * (DELTA_CLIP_NORM / _rn)
                w, normed = dfx.fltrust_weights(neighbor_deltas, root_delta)
                agg = sum(w[j] * (node_weights[j] + normed[j]) for j in nbrs)

            elif method == "openclaw":
                for j in nbrs:
                    ld = local_loss(model_template, node_weights[i] + neighbor_deltas[j], *clients[i]) \
                        - local_loss(model_template, node_weights[i], *clients[i])
                    openclaw_states[i].record(j, ld)
                w = openclaw_states[i].weights(nbrs)
                agg = sum(w[j] * neighbor_states[j] for j in nbrs)

            elif method in FINAL_STATE_METHODS:
                if method == "clipped_gossip":
                    final = dfx.clipped_gossip(own_updated, neighbor_states, f_local_assumed)
                elif method == "balance":
                    final = dfx.balance(own_updated, neighbor_states, t, T,
                                        gamma=variant.get("balance_gamma", BALANCE_GAMMA))
                else:
                    final = dfx.ios(own_updated, neighbor_states, f_local_assumed)
                new_weights[i] = (1 - GOSSIP_ETA) * node_weights[i] + GOSSIP_ETA * final
                for j in nbrs:   # no per-neighbor weights; log for detection metrics as uniform
                    detect_log.append({"t": t, "observer": i, "target": j, "malicious": j in byz_ids,
                                       "trust_weight": float("nan"), "dgain": dg[j]})
                continue

            elif method == "pcta_soft" and pcta_cfg["decision"] == "v2":
                if i not in obs_win_traj:
                    obs_win_traj[i], obs_win_lo[i], obs_win_au[i], obs_win_g[i] = (
                        _seeded(seed_traj)(), _seeded(seed_lo)(), _seeded(seed_au)(), _seeded(seed_g)())
                win_t, win_lo, win_au, win_g = obs_win_traj[i], obs_win_lo[i], obs_win_au[i], obs_win_g[i]
                w = {}
                agg = torch.zeros_like(node_weights[i])
                total_w = 0.0
                if pcta_cfg["norm_bound"]:
                    dev_n = {j: float(torch.norm(neighbor_states[j] - own_updated)) for j in nbrs}
                    _dn = np.array([v for v in dev_n.values() if math.isfinite(v)])
                    tau = (float(np.median(_dn) + Z_FLAG * 1.4826 * np.median(np.abs(_dn - np.median(_dn))))
                           if len(_dn) else 0.0)
                for j in nbrs:
                    ps = pair_states.get_or_create((i, j), lambda: PairState(TRUST_DIM, pcta_cfg))
                    mem = trust_mem.get((i, j))
                    if mem is None:
                        mem = trust_mem[(i, j)] = _new_trust_mem()
                    feed = (not pcta_cfg["gate_windows"]) or mem.pen >= WINDOW_GATE_PEN
                    Z, z_t, z_dl, z_dc = float("nan"), 0.0, 0.0, 0.0
                    if pcta_cfg["use_traj"]:
                        Z = float(ps.compute_Z(neighbor_deltas[j][trust_idx]))
                        x = math.log(max(Z, 1e-12)) if math.isfinite(Z) else float("inf")  # log tames Z's heavy tail
                        z_t = robust_z(x, win_t)
                        if math.isfinite(x) and feed:
                            win_t.append(x)
                    if pcta_cfg["use_data"]:
                        gm, lo, au = dg[j][0], dg[j][5], dg[j][6]   # level, log-odds contrast, dAUC (low = suspicious)
                        if not (math.isfinite(gm) and math.isfinite(lo) and math.isfinite(au)):
                            z_dl = z_dc = Z_CLIP  # non-finite data evidence = maximally suspicious
                        else:
                            z_dl = -robust_z(gm, win_g)
                            z_dc = -(robust_z(lo, win_lo) + robust_z(au, win_au)) / math.sqrt(2.0)
                            if feed:
                                win_g.append(gm)
                                win_lo.append(lo)
                                win_au.append(au)
                    S_t, S_dl, S_dc, pen, flagged = decide_v2(mem, z_t, z_dl, z_dc, pcta_cfg["use_cusum"])
                    mem.pen = pen
                    mem.reputation.record_outcome(max(z_t, z_dl, z_dc) < Z_FLAG)
                    # v2: probation keyed on IDENTITY AGE (rounds since j joined the
                    # network, certified by the membership / peer-sampling service),
                    # not on first contact with this observer. Whitewashing is a
                    # FRESH identity, so it stays capped; an established honest peer
                    # met for the first time does not. v1's per-observer probation
                    # capped nearly every new neighbor under R=5 re-shuffling, which
                    # cost ~0.03-0.04 clean accuracy (results_clean_ref.log) -- the
                    # mechanism behind the submitted ablation's "no cold-start" anomaly.
                    if pcta_cfg["identity_age_probation"] and t - join_round[j] >= pcta_cfg["probation_K"]:
                        mem.reputation.has_passed_probe = True
                    if pcta_cfg["use_reputation"]:
                        tw = mem.reputation.soft_mask() * pen
                    else:
                        tw = 0.0 if flagged else pen
                    w[j] = tw
                    total_w += tw
                    s_j = neighbor_states[j]
                    if pcta_cfg["norm_bound"] and pcta_cfg["residual_self"]:
                        dn = dev_n[j]
                        s_j = own_updated + (s_j - own_updated) * (min(1.0, tau / dn) if math.isfinite(dn) and dn > 0
                                                                   else (1.0 if dn == 0 else 0.0))
                    agg = agg + tw * s_j
                    detect_log.append({"t": t, "observer": i, "target": j, "malicious": j in byz_ids,
                                       "Z": Z, "z_t": z_t, "z_dl": z_dl, "z_dc": z_dc, "S": S_t,
                                       "S_dl": S_dl, "S_dc": S_dc,
                                       "flagged": bool(flagged), "trust_weight": float(tw),
                                       "dgain": dg[j]})
                if pcta_cfg["residual_self"]:
                    # Untrusted mass (k - sum tw) stays on i's own state: low trust
                    # means "ignore", not "redistribute to the other neighbors".
                    agg = (agg + (len(nbrs) - total_w) * own_updated) / len(nbrs)
                elif total_w > 1e-8:
                    agg = agg / total_w
                else:
                    agg = torch.stack([neighbor_states[j] for j in nbrs]).mean(dim=0)
                w = {j: (w[j] / total_w if total_w > 1e-8 else 1.0 / len(nbrs)) for j in nbrs}

            elif method == "pcta_soft":   # v1 decision rule (submitted version), kept for comparison
                # Adaptive threshold (replaces the fixed nu/chi2_thresh
                # calibrated once at t~3-6 and held for the whole run -- Phase 6
                # post-mortem finding, PCTA_Experimental_Methodology.md SS B.8's
                # pre-registered fallback; see observer_z_history's docstring
                # above for the v1-v3 parametric designs that were tried and
                # failed before this non-parametric one). Each neighbor's Z is
                # judged against this OBSERVER's own empirical Z-history
                # percentiles as they stood BEFORE this observation, then this
                # Z is appended to that history unconditionally.
                hist = observer_z_history.setdefault(i, collections.deque(maxlen=HISTORY_LEN))
                s_hist = observer_s_history.setdefault(i, collections.deque(maxlen=HISTORY_LEN))

                w = {}
                agg = torch.zeros_like(node_weights[i])
                total_w = 0.0
                for j in nbrs:
                    key = (i, j)
                    ps = pair_states.get_or_create(key, lambda: PairState(TRUST_DIM, pcta_cfg))
                    Z = ps.compute_Z(neighbor_deltas[j][trust_idx])

                    if len(hist) >= MIN_HISTORY_FOR_ADAPTIVE:
                        hist_arr = np.array(hist)
                        adaptive_nu = float(np.percentile(hist_arr, NU_PERCENTILE)) + 5.0  # Delta/2 slack
                        adaptive_thresh = float(np.percentile(hist_arr, THRESH_PERCENTILE))
                    else:
                        adaptive_nu, adaptive_thresh = nu, chi2_thresh  # calibration-time fallback

                    if len(s_hist) >= MIN_HISTORY_FOR_ADAPTIVE:
                        adaptive_h = max(float(np.percentile(np.array(s_hist), H_PERCENTILE)) * ADAPTIVE_MARGIN, 10.0)
                    else:
                        adaptive_h = h  # calibration-time fallback

                    S, cpen, flagged = ps.apply_decision(Z, adaptive_nu, adaptive_h, adaptive_thresh)
                    hist.append(float(Z))    # appended unconditionally -- see docstring on why
                    # s_hist gated on REPUTATION (a signal separate from the S-vs-h
                    # comparison it calibrates, see docstring above) -- not on
                    # `flagged` itself (v6's mistake) and not a fixed blind trim
                    # (v6's over-correction). LOCKED as the hard cutoff over
                    # v8-continuous's weighted version -- see docstring above for why.
                    if ps.reputation.R_hat >= S_HIST_MIN_REPUTATION:
                        s_hist.append(float(S))

                    if pcta_cfg["use_probe"] and (ps.reputation.needs_probe(t) or
                                                    (np.random.rand() < PROBE_TRIGGER_PROB)):
                        judge_ids = [x for x in nbrs if x != j][:3] or [i]
                        judge_weights = [node_weights[x] if x != i else node_weights[i] for x in judge_ids]
                        is_byz_j = j in byz_ids
                        if attack in LABEL_ATTACKS:
                            suspect_w = node_weights[j] + neighbor_deltas[j]
                            flipped = flip_labels(probe_y, attack)
                            corrupt_fn = lambda g, w_=suspect_w, y_=flipped: probe.gradient_on_probe(w_, labels=y_)
                        else:
                            corrupt_fn = lambda g: apply_attack(g, j, t, attack_state, onset_round=1)
                        honest, dprobe = probe.run(node_weights[j] + neighbor_deltas[j], judge_weights,
                                                     corrupt_fn, is_byz_j,
                                                     attacker_knows_probe_timing=variant.get(
                                                         "attacker_knows_probe_timing", False))
                        ps.reputation.apply_probe_verdict(honest, t)
                        probe_log.append({"t": t, "observer": i, "target": j,
                                           "malicious": is_byz_j, "verdict_honest": honest,
                                           "d_probe": dprobe})

                    tw = trust_weight(ps, cpen, pcta_cfg["use_reputation"])
                    w[j] = tw
                    total_w += tw
                    agg = agg + tw * neighbor_states[j]
                    detect_log.append({"t": t, "observer": i, "target": j, "malicious": j in byz_ids,
                                        "Z": float(Z), "S": float(S), "flagged": bool(flagged),
                                        "trust_weight": float(tw), "dgain": dg[j]})
                if total_w > 1e-8:
                    agg = agg / total_w
                else:
                    agg = torch.stack([neighbor_states[j] for j in nbrs]).mean(dim=0)
                w = {j: (w[j] / total_w if total_w > 1e-8 else 1.0 / len(nbrs)) for j in nbrs}

            else:
                raise ValueError(method)

            # Blend in node i's OWN local SGD step, always at full trust (a node
            # doesn't need to test itself). Root-cause fix for a Phase 6 post-mortem
            # finding: `nbrs = topology.neighbors(i)` never includes i (topology.py),
            # so `agg` above is computed purely from NEIGHBOR deltas -- i's own
            # local_train() work was being silently discarded every round, matching
            # the original PCTA Theorem 1's literal form (sum over j in N(i) only)
            # but inconsistent with standard decentralized SGD (Koloskova et al.
            # 2020, Lian et al. 2017 -- both already cited in PCTA_Theory.md SS7's
            # convergence theorem, and both include a self/local-gradient term).
            # Confirmed empirically: without this, even clean FedAvg with zero
            # attacks stayed flat at chance-level accuracy for all 100 rounds --
            # self-discard, not any defense mechanism, was the actual cause. Weight
            # convention: self counts as one more equally-trusted participant among
            # (len(nbrs)+1) total, i.e. uniform gossip mixing including self.
            # `agg` is now a STATE (weighted average of neighbor_states), not a
            # delta -- blend with i's own committed state the same way, then move
            # i's actual weights toward that blended target by GOSSIP_ETA (a
            # damping/relaxation parameter, not an increment scale anymore).
            self_w = 1.0 / (len(nbrs) + 1)
            blended_state = self_w * own_updated + (1 - self_w) * agg

            new_weights[i] = (1 - GOSSIP_ETA) * node_weights[i] + GOSSIP_ETA * blended_state

            if method != "pcta_soft":
                for j in nbrs:
                    detect_log.append({"t": t, "observer": i, "target": j, "malicious": j in byz_ids,
                                        "trust_weight": float(w.get(j, 0.0)), "dgain": dg[j]})

        node_weights = new_weights
        if verbose_timing and (t <= 5 or t % 10 == 0):
            print(f"  [t={time.time()-t0:.1f}s] official round {t}/{T} done "
                  f"(n_active_pairs={len(pair_states._d) if method=='pcta_soft' else 'n/a'})")

        if t % 5 == 0 or t == T:
            accs = []
            honest_ids = [x for x in range(N) if x not in byz_ids]
            for i in honest_ids[::max(1, len(honest_ids) // 10)]:  # honest nodes only (v2), subsampled for speed
                set_flat_params(model_template, node_weights[i])
                model_template.eval()
                with torch.no_grad():
                    pred = model_template(test_x).argmax(dim=1)
                    accs.append((pred == test_y).float().mean().item())
            acc_log.append({"t": t, "mean_acc": float(np.mean(accs)), "min_acc": float(np.min(accs))})

    result = {
        "dataset": dataset, "method": method, "attack": attack, "byz_fraction": byz_fraction,
        "N": N, "k": k, "T": T, "R": R, "placement": placement, "alpha": alpha, "seed": seed,
        "variant": variant, "d": d, "lambda_honest_est": lambda_honest_est, "nu": nu, "h": h,
        "delta_probe": probe.delta_probe,
        "acc_log": acc_log, "wall_clock_s": time.time() - t0,
    }
    if not quiet:
        print(f"[{tag}] final mean_acc={acc_log[-1]['mean_acc']:.3f} "
              f"({time.time()-t0:.1f}s)")

    os.makedirs(out_dir, exist_ok=True)
    tag = tag or f"{dataset}_{method}_{attack}_f{byz_fraction}_a{alpha}_s{seed}"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(result, fh)
    with open(os.path.join(out_dir, f"{tag}_detect.json"), "w") as fh:
        json.dump(detect_log, fh)
    if probe_log:
        with open(os.path.join(out_dir, f"{tag}_probe.json"), "w") as fh:
            json.dump(probe_log, fh)
    return result
