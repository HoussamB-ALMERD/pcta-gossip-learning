"""PCTA-Soft decision-rule engine.

Implements, directly against PCTA_Theory.md:
  - Definition 4' (Trajectory Predictor): diagonal Kalman and EMA instantiations.
  - Definition 5'/5" (standardized innovation Z_j(t), corrected CUSUM).
  - Definition 3' (reputation success/failure driven by Z_j(t), not local loss).
  - Section 8 (cold-start / Sybil-resistance policy: conservative prior, probation
    cap, mandatory pre-graduation probe).
  - Definitions 6'/7' (silent-embedding Canary Probe, retroactive verification).

All per-neighbor state is torch tensors on DEVICE, vectorized over the d model
coordinates -- O(d) per neighbor per round (PCTA_Decisions.md #4), not O(d^3).
"""
import math
import torch
from scipy.stats import chi2

from common import DEVICE

# ---------------------------------------------------------------------------
# Section 2.1 -- Trajectory Predictor (Definition 4')
# ---------------------------------------------------------------------------


class KalmanPredictor:
    """Diagonal, constant-velocity Kalman filter: d independent scalar 2-state
    filters, closed-form (no batched matrix inversion -- PCTA_Theory.md SS2.1)."""

    def __init__(self, d, q=1e-5, r=1e-4, init_var=1.0, device=DEVICE):
        self.d = d
        self.q = q
        self.r = r
        self.x0 = torch.zeros(d, device=device)   # position estimate
        self.x1 = torch.zeros(d, device=device)   # velocity estimate
        self.p00 = torch.full((d,), init_var, device=device)
        self.p01 = torch.zeros(d, device=device)
        self.p11 = torch.full((d,), init_var, device=device)
        self.n_obs = 0

    def predict(self):
        """Returns (pred, scale) using state as of the last observation -- BEFORE
        this round's observation. scale is the predicted innovation std (sqrt of
        the Kalman innovation covariance S = P_pred[0,0] + r)."""
        x_pred0 = 2 * self.x0 - self.x1
        x_pred1 = self.x0
        p_pred00 = 4 * self.p00 - 4 * self.p01 + self.p11 + self.q
        p_pred01 = 2 * self.p00 - self.p01
        p_pred11 = self.p00 + self.q
        self._cache = (x_pred0, x_pred1, p_pred00, p_pred01, p_pred11)
        s = p_pred00 + self.r
        scale = torch.sqrt(torch.clamp(s, min=1e-12))
        return x_pred0, scale

    def observe(self, z_obs):
        # Root-cause fix for a Phase 6 post-mortem NaN finding (Detectability
        # Floor debugging, 2026-09-03): `predict()` clamps its innovation
        # covariance `s` to a positive floor before dividing by it, but this
        # method never did -- under an extreme-enough observation (which a
        # Byzantine attacker can deliberately produce, by design, unlike
        # ordinary honest noise), `p_pred00` could go non-positive from prior
        # numerical drift, making `s` near-zero or negative, `k0 = p00/s`
        # explode or invert sign, and `p00 = p_pred00*(1-k0)` go negative --
        # which then permanently poisons this pair's filter state on every
        # subsequent round (observed: Z reaching ~11 million, then outright
        # NaN a few rounds later). Since this filter must tolerate genuinely
        # adversarial, not just well-behaved, inputs by design, defensive
        # clamping here is required, not optional -- a no-op under any
        # previously-validated (well-behaved) condition, since p_pred00/p11
        # stay comfortably positive whenever the filter is actually tracking
        # something predictable.
        x_pred0, x_pred1, p_pred00, p_pred01, p_pred11 = self._cache
        s = torch.clamp(p_pred00 + self.r, min=1e-12)
        k0 = p_pred00 / s
        k1 = p_pred01 / s
        y = z_obs - x_pred0
        self.x0 = x_pred0 + k0 * y
        self.x1 = x_pred1 + k1 * y
        self.p00 = torch.clamp(p_pred00 * (1 - k0), min=1e-12)
        self.p11 = torch.clamp(p_pred11 - k1 * p_pred01, min=1e-12)
        # p01 left unclamped above could otherwise drift to a value
        # inconsistent with any valid covariance matrix; a real 2x2 covariance
        # requires p01^2 <= p00*p11 (positive semi-definiteness) -- enforce it
        # directly rather than leaving p01 as an unconstrained free variable.
        p01_raw = p_pred01 * (1 - k0)
        p01_bound = torch.sqrt(self.p00 * self.p11)
        self.p01 = torch.clamp(p01_raw, -p01_bound, p01_bound)
        self.n_obs += 1


class EMAPredictor:
    """EMA predictor + parallel EMA of squared residuals for the scale estimate --
    the ablation counterpart to KalmanPredictor (propositions.md P12, P4)."""

    def __init__(self, d, beta=0.3, beta_v=0.3, init_var=1.0, device=DEVICE):
        self.d = d
        self.beta = beta
        self.beta_v = beta_v
        self.x_hat = torch.zeros(d, device=device)
        self.var = torch.full((d,), init_var, device=device)
        self.n_obs = 0

    def predict(self):
        scale = torch.sqrt(torch.clamp(self.var, min=1e-12))
        return self.x_hat, scale

    def observe(self, z_obs):
        r = z_obs - self.x_hat
        self.x_hat = (1 - self.beta) * self.x_hat + self.beta * z_obs
        self.var = (1 - self.beta_v) * self.var + self.beta_v * r ** 2
        self.n_obs += 1


def make_predictor(kind, d, **kwargs):
    if kind == "kalman":
        return KalmanPredictor(d, **kwargs)
    elif kind == "ema":
        return EMAPredictor(d, **kwargs)
    elif kind == "none":
        return None  # No-Kalman ablation arm: raw-drift fallback, see Z_of_raw_drift below
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Section 2.2 / 3 -- standardized innovation Z_j(t), corrected CUSUM
# ---------------------------------------------------------------------------


def standardized_innovation_Z(pred, scale, observed):
    """Definition 5': Z_j(t) = sum_l z_{j,l}(t)^2. Returns (Z: scalar tensor,
    z: per-coordinate standardized innovation, for optional diagnostics)."""
    z = (observed - pred) / scale
    Z = torch.sum(z * z)
    return Z, z


def raw_drift_Z(prev_delta, observed, scale=1.0):
    """No-Kalman/No-predictor ablation arm (propositions.md P12): 'prediction' is
    just the last observed delta, no learned scale -- Z is a plain squared-norm
    ratio, not a calibrated chi-squared statistic. Kept deliberately crude; the
    point of this arm is to show what's lost without a real predictor."""
    d = observed.numel()
    diff = observed - prev_delta
    return torch.sum(diff * diff) / max(scale, 1e-8), None


class CUSUM:
    """Definition 5'' (corrected CUSUM), PCTA_Theory.md SS3.
    nu = d + Delta/2, where Delta is the design slack; see calibration.py-equivalent
    logic in harness.py for how (d, lambda_honest, Delta) become nu, h in practice.
    """

    def __init__(self, nu, h):
        self.nu = nu
        self.h = h
        self.S = 0.0

    def update(self, Z):
        self.S = max(0.0, self.S + float(Z) - self.nu)
        return self.S

    @property
    def flagged(self):
        return self.S > self.h

    def penalty(self):
        return math.exp(-self.S / self.h) if self.h > 0 else 1.0


def chi2_quantile(d, q):
    """Central chi-squared (d, q)-quantile, used both for Definition 3''s
    per-round significance threshold and for sanity-checking calibration."""
    return float(chi2.ppf(q, df=d))


# ---------------------------------------------------------------------------
# Section 2.3 / 8 -- Reputation (Definition 3') + cold-start / Sybil policy
# ---------------------------------------------------------------------------


class BetaReputation:
    """Definition 3' (success/failure driven by Z_j(t), not local loss) +
    Section 8 (cold-start policy: conservative prior, probation cap, mandatory
    pre-graduation probe -- PCTA_Decisions.md #10)."""

    def __init__(self, alpha0=1.0, beta0=2.0, theta=0.3, gamma=0.05,
                 probation_K=5, M_max=0.2, probe_cooldown=10):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.theta = theta
        self.gamma = gamma
        self.probation_K = probation_K
        self.M_max = M_max
        self.probe_cooldown = probe_cooldown
        self.S = 0.0
        self.F = 0.0
        self.n_interactions = 0
        self.has_passed_probe = False
        self.last_probe_t = None

    def record(self, Z, chi2_thresh):
        self.n_interactions += 1
        if Z <= chi2_thresh:
            self.S += 1.0
        else:
            self.F += 1.0

    def record_outcome(self, ok):
        """Revision v2: success/failure from the combined per-round evidence of
        both channels (see harness.py pcta_soft v2), rather than one raw Z vs.
        a percentile threshold."""
        self.n_interactions += 1
        if ok:
            self.S += 1.0
        else:
            self.F += 1.0

    @property
    def R_hat(self):
        return (self.alpha0 + self.S) / (self.alpha0 + self.beta0 + self.S + self.F)

    def needs_probe(self, t=None):
        """Interrogation trigger: still on probation and not yet graduated
        (Section 8's mandatory pre-graduation probe -- always fires, once),
        OR below the trust threshold AND at least probe_cooldown rounds have
        passed since the last probe of this pair. The cooldown is a genuine
        design choice, not just a speed optimization: once a node has been
        probe-confirmed malicious and its trust weight has already hit its
        floor, re-probing every single subsequent round verifies nothing new
        -- the accept/reject decision is already at its strongest possible
        verdict. Periodic re-verification (rather than continuous) is the more
        realistic operating policy this models (PCTA_Roadmap.md risk register
        / questions.md Q15's probe-frequency-vs-overhead tradeoff, now with a
        concrete mechanism)."""
        on_probation = self.n_interactions < self.probation_K
        if on_probation and not self.has_passed_probe:
            return True
        if self.R_hat >= self.theta:
            return False
        if t is None or self.last_probe_t is None:
            return True
        return (t - self.last_probe_t) >= self.probe_cooldown

    def soft_mask(self):
        m = 1.0 / (1.0 + math.exp(-(self.R_hat - self.theta) / self.gamma))
        if self.n_interactions < self.probation_K and not self.has_passed_probe:
            m = min(m, self.M_max)  # Section 8 probation cap
        return m

    def apply_probe_verdict(self, honest, t=None):
        self.last_probe_t = t
        if honest:
            self.has_passed_probe = True
            self.S += 0.3 * (self.alpha0 + self.beta0)  # modest restoration, bounded by priors
        else:
            self.F += 0.5 * (self.alpha0 + self.beta0)  # severe penalty


# ---------------------------------------------------------------------------
# Section 10 -- Silent-embedding Canary Probe (Definitions 6'/7')
# ---------------------------------------------------------------------------


class CanaryProbe:
    """Simulates Definitions 6'/7': the probe is embedded in an ordinary exchange
    (no 'is_probe' flag reaches the corruption function unless
    attacker_knows_probe_timing=True, which exists ONLY as the deliberate
    negative-control comparison to the original, broken, announced-challenge
    design -- PCTA_Experimental_Methodology.md A.1 research question 5).
    """

    def __init__(self, probe_x, probe_y, delta_probe, model_fn, loss_fn):
        """probe_x, probe_y: a small fixed self-supervised-style probe batch
        (labels used here only as a simulation convenience for a supervised
        SmallCNN backbone; PCTA_Theory.md Definition 6' calls for a genuinely
        self-supervised loss in a real deployment -- see
        PCTA_Experimental_Methodology.md C.4 for this stated simplification)."""
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.delta_probe = delta_probe
        self.model_fn = model_fn      # (flat_weights) -> torch nn.Module with those weights
        self.loss_fn = loss_fn

    def gradient_on_probe(self, flat_weights, labels=None):
        """labels: override for the probe labels -- used to simulate a
        data-poisoning (label-flipping) node, which computes ITS gradient on
        poisoned labels for every sample it trains on, probe samples included
        (revision v2 fix: v1 let label-flippers answer probes honestly, since
        their attack lives in local_train, not in apply_attack)."""
        model = self.model_fn(flat_weights)
        model.zero_grad()
        out = model(self.probe_x)
        loss = self.loss_fn(out, self.probe_y if labels is None else labels)
        loss.backward()
        g = torch.cat([p.grad.detach().view(-1) for p in model.parameters()])
        return g

    def run(self, suspect_weights, judge_weights_list, corrupt_fn, is_byzantine,
            attacker_knows_probe_timing=False):
        """Returns (verdict_honest: bool, d_probe: float).

        corrupt_fn(g) -> corrupted gradient, the SAME per-round corruption function
        currently active for this Byzantine node's regular delta. Applying it here
        too is the core of the fix: node j cannot special-case a probe it cannot
        identify, so its probe response is corrupted by construction whenever its
        regular output is (unless attacker_knows_probe_timing=True, which restores
        the original vulnerable behavior for comparison)."""
        g_suspect_true = self.gradient_on_probe(suspect_weights)
        if is_byzantine and not (attacker_knows_probe_timing):
            g_suspect_reported = corrupt_fn(g_suspect_true)
        elif is_byzantine and attacker_knows_probe_timing:
            g_suspect_reported = g_suspect_true  # detects the probe, behaves honestly
        else:
            g_suspect_reported = g_suspect_true

        g_truth = torch.stack([self.gradient_on_probe(w) for w in judge_weights_list]).mean(dim=0)
        # Revision v2: scale-free relative distance. v1 compared the ABSOLUTE
        # distance to a fixed delta_probe=1.0, which is meaningless once the
        # gradient scale changes (100% honest probe FPR under the v2 training
        # config). delta_probe is now calibrated from honest probes on clean
        # calibration rounds (harness.py).
        d_probe = (torch.norm(g_suspect_reported - g_truth) / torch.norm(g_truth).clamp(min=1e-12)).item()
        return d_probe < self.delta_probe, d_probe

    def honest_distance(self, suspect_weights, judge_weights_list):
        """Relative probe distance of an honest suspect -- for calibrating delta_probe."""
        g_s = self.gradient_on_probe(suspect_weights)
        g_truth = torch.stack([self.gradient_on_probe(w) for w in judge_weights_list]).mean(dim=0)
        return (torch.norm(g_s - g_truth) / torch.norm(g_truth).clamp(min=1e-12)).item()
