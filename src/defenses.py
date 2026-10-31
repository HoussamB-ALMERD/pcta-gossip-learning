"""Baseline decision rules, per PCTA_Experimental_Methodology.md B.5.

Every rule is run LOCALLY by each node over its own cleartext neighbor set --
per Methodology A.5's locked framing, this is the fair, apples-to-apples
reinterpretation for a gossip setting (not the server-side versions DBT-CFL-PTT
used, since PCTA-Soft's real capability-gap axis is temporal memory + active
defense, not information asymmetry).

Each function returns {neighbor_id: weight}, weights summing to 1 over the
ACCEPTED subset (weight 0 for rejected neighbors).
"""
import math
import torch


def fedavg(neighbor_deltas):
    ids = list(neighbor_deltas.keys())
    w = 1.0 / len(ids)
    return {i: w for i in ids}


def local_krum(neighbor_deltas, f_local):
    ids = list(neighbor_deltas.keys())
    k = len(ids)
    f_local = min(f_local, max(0, k - 3))  # keep k-f-2 >= 1
    m = max(1, k - f_local - 2)
    scores = {}
    for i in ids:
        dists = sorted(
            torch.sum((neighbor_deltas[i] - neighbor_deltas[j]) ** 2).item()
            for j in ids if j != i
        )
        scores[i] = sum(dists[:m])
    keep = sorted(ids, key=lambda i: scores[i])[: max(1, k - f_local)]
    w = 1.0 / len(keep)
    return {i: (w if i in keep else 0.0) for i in ids}


def trimmed_mean_weights(neighbor_deltas, f_local, use_median=False):
    """Coordinate-wise trimmed mean/median doesn't reduce to a single per-neighbor
    scalar weight (it mixes coordinates from different neighbors), so this returns
    the AGGREGATE directly rather than a weight dict -- flagged via the tuple's
    first element so harness.py can branch."""
    ids = list(neighbor_deltas.keys())
    stack = torch.stack([neighbor_deltas[i] for i in ids], dim=0)  # (k, d)
    k = stack.shape[0]
    if use_median:
        agg = torch.median(stack, dim=0).values
    else:
        f_local = min(f_local, max(0, (k - 1) // 2))
        sorted_stack, _ = torch.sort(stack, dim=0)
        if f_local > 0:
            trimmed = sorted_stack[f_local: k - f_local]
        else:
            trimmed = sorted_stack
        agg = trimmed.mean(dim=0)
    return ("aggregate", agg)


class FoolsGoldState:
    """Standard, simplified implementation of Fung et al. (2020): historical
    gradient-similarity-based Sybil penalty, the one baseline here with genuine
    (if simpler than PCTA's) cross-round memory."""

    def __init__(self, kappa=1.0):
        self.history = {}  # node_id -> running sum of past deltas
        self.kappa = kappa

    def _update_history(self, node_id, delta):
        if node_id not in self.history:
            self.history[node_id] = torch.zeros_like(delta)
        self.history[node_id] = self.history[node_id] + delta

    def weights(self, neighbor_deltas):
        ids = list(neighbor_deltas.keys())
        for i in ids:
            self._update_history(i, neighbor_deltas[i])
        hist = torch.stack([self.history[i] for i in ids], dim=0)
        norm = torch.norm(hist, dim=1, keepdim=True).clamp(min=1e-8)
        unit = hist / norm
        cs = unit @ unit.T  # (k,k) cosine similarity
        cs.fill_diagonal_(0.0)
        n = len(ids)
        v = cs.max(dim=1).values  # max similarity per node
        # Pardoning: if v_j > v_i, scale cs_ij by v_i/v_j
        cs_pardoned = cs.clone()
        for a in range(n):
            for b in range(n):
                if a != b and v[b] > v[a] and v[b] > 1e-8:
                    cs_pardoned[a, b] = cs[a, b] * (v[a] / v[b])
        wv = 1.0 - cs_pardoned.max(dim=1).values
        wv = torch.clamp(wv, 1e-4, 1.0)
        wv = wv / wv.max().clamp(min=1e-8)
        logit = self.kappa * (torch.log(wv / (1 - wv).clamp(min=1e-4)) + 0.5)
        wv = torch.clamp(logit, 0.0, 1.0)
        if wv.sum() < 1e-8:
            wv = torch.ones_like(wv) / n
        wv = wv / wv.sum()
        return {ids[i]: wv[i].item() for i in range(n)}


def fltrust_weights(neighbor_deltas, root_grad):
    """Cao et al. (2021). Requires a shared, trusted root dataset -- simulated
    here as a common, publicly-available labeled reference batch (a simplification
    of the single-elected-root-holder framing, noted explicitly in
    PCTA_Experimental_Methodology.md C.4)."""
    ids = list(neighbor_deltas.keys())
    root_norm = torch.norm(root_grad).clamp(min=1e-8)
    root_unit = root_grad / root_norm
    ts = {}
    normed = {}
    for i in ids:
        gi = neighbor_deltas[i]
        gi_norm = torch.norm(gi).clamp(min=1e-8)
        cos_sim = torch.dot(gi / gi_norm, root_unit).item()
        ts[i] = max(0.0, cos_sim)
        normed[i] = gi * (root_norm / gi_norm)
    total = sum(ts.values())
    if total < 1e-8:
        w = 1.0 / len(ids)
        return {i: w for i in ids}, normed
    return {i: ts[i] / total for i in ids}, normed


class OpenCLAWReputationState:
    """Reconstruction of OpenCLAW-Nexus (arXiv:2605.04091), refined against the
    Phase 1 literature pass (PCTA_Related_Work.md SS2): confirmed decentralized,
    GOSSIP-topology (Kademlia DHT, fanout=6, TTL=7 -- not just "distributed" in
    the loose sense), discounted Beta-reputation-weighted aggregation, no root
    dataset, no sequential/predictive layer. Exact formula confirmed:
    alpha_k^(t+1)=lambda*alpha_k^(t)+o_t, beta_k^(t+1)=lambda*beta_k^(t)+(1-o_t),
    lambda=0.95, binary per-round outcome o_t -- discount below matches this
    (was rho=0.98 pending confirmation; corrected here).

    Deliberately uses the SAME local-loss-delta success criterion PCTA_Theory.md
    Definition 3' moved away from (replaced by the innovation-residual criterion,
    closing critics.md C1) -- a system with reputation but no predictive/sequential
    layer has no other natural signal to use, and this is precisely the criterion
    whose Non-IID vulnerability this experiment is positioned to surface.
    """

    def __init__(self, alpha0=1.0, beta0=1.0, rho=0.95):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.rho = rho
        self.S = {}
        self.F = {}

    def record(self, node_id, loss_delta, fail_thresh=0.1):
        self.S[node_id] = self.S.get(node_id, 0.0) * self.rho
        self.F[node_id] = self.F.get(node_id, 0.0) * self.rho
        if loss_delta < 0:
            self.S[node_id] += 1.0
        elif loss_delta > fail_thresh:
            self.F[node_id] += 1.0

    def weights(self, neighbor_ids):
        r = {}
        for i in neighbor_ids:
            s, f = self.S.get(i, 0.0), self.F.get(i, 0.0)
            r[i] = (self.alpha0 + s) / (self.alpha0 + self.beta0 + s + f)
        total = sum(r.values())
        if total < 1e-8:
            w = 1.0 / len(neighbor_ids)
            return {i: w for i in neighbor_ids}
        return {i: r[i] / total for i in neighbor_ids}


# ---------------------------------------------------------------------------
# Revision v2 baselines (2026-10-17): decentralized-native robust aggregators.
# Each operates on neighbor STATES relative to the node's own post-local-step
# state (as in the original papers) and returns the node's NEXT STATE directly;
# harness.py bypasses its generic self-blend for these (FINAL_STATE_METHODS).
# ---------------------------------------------------------------------------

def clipped_gossip(own, neighbor_states, f_local):
    """ClippedGossip (He, Karimireddy & Jaggi, 2022):
        x_i <- x_i + sum_j W_ij CLIP(x_j - x_i, tau_i),  uniform W_ij = 1/(k+1).
    tau_i: the paper's adaptive rule needs the honest weight mass, which a node
    cannot know; we use the common practical instantiation of clipping the
    f_local largest neighbor distances to the (k - f_local)-th smallest one."""
    ids = list(neighbor_states.keys())
    k = len(ids)
    diffs = {j: neighbor_states[j] - own for j in ids}
    norms = {j: float(torch.norm(diffs[j])) for j in ids}
    order = sorted(norms.values())
    tau = order[max(0, k - f_local - 1)] if k > 0 else 0.0
    step = torch.zeros_like(own)
    for j in ids:
        scale = min(1.0, tau / norms[j]) if norms[j] > 0 else 1.0
        step = step + scale * diffs[j]
    return own + step / (k + 1)


def balance(own, neighbor_states, t, T, gamma=0.3, kappa=1.0, alpha=0.5):
    """BALANCE (Fang et al., 2024): accept neighbor j iff
        ||x_j - x_i|| <= gamma * exp(-kappa * t / T) * ||x_i||,
    then x_i <- alpha * x_i + (1 - alpha) * mean(accepted) (Eqs. 3-4, Alg. 3).
    Defaults alpha=0.5, gamma=0.3, kappa=1, lambda(t)=t/T verified against the
    paper's Sec. 6.1.5 (arXiv 2406.10416v4, p.7) on 2026-10-17. If no neighbor
    is accepted, the node keeps its own model (|S| = 0 case, undefined in Eq. 4)."""
    thr = gamma * math.exp(-kappa * t / T) * float(torch.norm(own))
    acc = [s for s in neighbor_states.values() if float(torch.norm(s - own)) <= thr]
    if not acc:
        return own.clone()
    return alpha * own + (1 - alpha) * torch.stack(acc).mean(dim=0)


def ios(own, neighbor_states, f_local):
    """IOS -- Iterative Outlier Scissor (Wu et al., 2023): starting from {self} U
    neighbors with uniform weights, repeat f_local times: compute the current
    weighted average and remove the NEIGHBOR (never self) farthest from it;
    return the average of what remains."""
    pool = {"self": own}
    pool.update({j: s for j, s in neighbor_states.items()})
    for _ in range(min(f_local, len(neighbor_states))):
        avg = torch.stack(list(pool.values())).mean(dim=0)
        far = max((j for j in pool if j != "self"), key=lambda j: float(torch.norm(pool[j] - avg)))
        del pool[far]
    return torch.stack(list(pool.values())).mean(dim=0)
