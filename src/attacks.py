"""The six attack archetypes, per PCTA_Experimental_Methodology.md B.7.

Five reused/adapted from the original PCTA report SS8.2 / DBT-CFL-PTT corruption
mechanics; one new (calibrated-adaptive) built specifically to exercise
PCTA_Theory.md SS6's Detectability Floor.

Every corrupt_fn(delta, state) -> corrupted_delta is applied post-hoc to an
otherwise-honestly-trained delta, matching DBT-CFL-PTT's exact convention -- and,
per pcta_engine.CanaryProbe, the SAME corrupt_fn is reused for a node's probe
response, since a node cannot special-case a probe it cannot identify
(PCTA_Theory.md Definitions 6'/7').
"""
import math
import torch


class AttackState:
    """Holds any persistent per-client state an attack needs (e.g. boiling-frog's
    fixed poison direction, Sybil's colluding-group assignment)."""

    def __init__(self, attack_type, byz_ids, d, seed, params=None):
        self.attack_type = attack_type
        self.byz_ids = set(byz_ids)
        self.params = params or {}
        g = torch.Generator().manual_seed(seed)
        self.poison_dir = {i: torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)
                            for i in byz_ids}
        # Sybil collusion: all Byzantine nodes share ONE fixed poison direction
        # (mutual reinforcement / echo-chamber intent), rather than independent ones.
        shared = torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)
        self.sybil_dir = shared


def _clip(delta, max_norm):
    n = torch.norm(delta)
    if n > max_norm:
        return delta * (max_norm / n)
    return delta


def apply_attack(delta, node_id, t, attack_state: AttackState, onset_round=1,
                  device=None):
    """Returns the (possibly corrupted) delta for this node/round. Honest nodes
    and pre-onset rounds pass through unchanged."""
    if node_id not in attack_state.byz_ids or t < onset_round:
        return delta

    atype = attack_state.attack_type
    p = attack_state.params

    if atype == "gaussian_noise":
        sigma = p.get("sigma", 1.0)
        return delta + sigma * torch.randn_like(delta)

    elif atype in ("label_flip", "label_flip_targeted"):
        # Label flipping happens upstream (in local training, on the labels
        # themselves) -- this branch is a no-op by design; see harness.py's
        # local_train, which checks attack_state.attack_type == "label_flip"
        # BEFORE computing delta, not here.
        return delta

    elif atype in ("alie", "ipm", "dissensus"):
        # Crafted from honest statistics / per receiver in harness.py (revision v2);
        # no per-node post-hoc corruption here.
        return delta

    elif atype == "sign_flip":
        boost = p.get("boost", 5.0)
        return -boost * delta

    elif atype == "slow_poison":
        # Boiling-frog ramp, reusing DBT-CFL-PTT's mechanics: fixed per-client
        # poison direction, linearly ramping coefficient between ramp_start/ramp_end.
        ramp_start = p.get("ramp_start", 30)
        ramp_end = p.get("ramp_end", 90)
        max_coef = p.get("max_coef", 2.5)
        if t < ramp_start:
            return delta
        coef = min(max_coef, max_coef * (t - ramp_start) / (ramp_end - ramp_start))
        u = attack_state.poison_dir[node_id].to(delta.device)
        return delta + coef * torch.norm(delta) * u

    elif atype == "sybil":
        # Colluding group: every Sybil identity pushes the SAME shared direction,
        # scaled, so their mutual "agreement" is intentional (echo-chamber intent,
        # SS1.2.1 of the original PCTA report) -- this is what the probe's
        # judge-majority mechanism (PCTA_Theory.md SS4) must see through.
        boost = p.get("boost", 3.0)
        u = attack_state.sybil_dir.to(delta.device)
        return boost * torch.norm(delta) * u

    elif atype == "calibrated_adaptive":
        # White-box attack (PCTA_Experimental_Methodology.md B.7). Root-cause
        # fix for a Phase 6 post-mortem finding: the original design tried to
        # inject an EXACT target noncentrality in the defender's own Z-units,
        # requiring a per-coordinate scale estimate (p["scale_l"]) that MATCHES
        # whatever predictor the eventual observer happens to be running. No
        # single scale can do this: (a) a single delta is broadcast to MANY
        # observers, each with a DIFFERENT, independently-evolving predictor
        # scale (Kalman filters narrow/converge at different rates per pair);
        # (b) the raw, unconverged calibration-time `mean_var` used as a
        # fallback proxy was off from real (converged) predictor scales by
        # orders of magnitude, producing Z in the MILLIONS regardless of the
        # intended ratio, and occasionally destabilizing the Kalman filter's
        # own internal state into NaN (see the corresponding fix in
        # pcta_engine.py's KalmanPredictor.observe()).
        # Fixed: scale the injected shift relative to the ATTACKER'S OWN
        # realized delta norm -- a real, properly-scaled quantity by
        # construction, exactly the same convention already proven to work for
        # slow_poison/sybil (`coef * norm(delta) * unit_vector`) rather than
        # reconstructing an independent per-coordinate noise-scale estimate
        # that has no single correct value across a broadcast to many
        # observers. `p["lambda_target"]` here is the raw ratio parameter
        # (lambda_attack/Gamma, PCTA_Experimental_Methodology.md B.7) -- CALIB
        # is an empirically-tuned constant (see smoke-scale calibration sweep,
        # Phase 6) translating that ratio into a delta-norm-relative
        # coefficient comparable in scale to slow_poison's own ramp (up to
        # 2.5x delta norm at its strongest).
        ratio = p.get("lambda_target", 0.0)
        CALIB = 0.5
        coef = CALIB * ratio
        u = attack_state.poison_dir[node_id].to(delta.device)
        return delta + coef * torch.norm(delta) * u

    else:
        raise ValueError(atype)
