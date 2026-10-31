"""Gossip peer-sampling topology.

Implements PCTA_Theory.md SS1 Assumption 0 (Adversary-Oblivious Peer Sampling):
each node i's neighbor set N(i,t) is redrawn, INDEPENDENTLY per node (not as a
shared symmetric graph -- this matches the hypergeometric-concentration proof in
PCTA_Theory.md SS4 exactly, which needs each honest node's own draw to be an i.i.d.
uniform k-subset of the population), as a uniformly random k-subset of
{0,...,N-1}\\{i}, every R rounds (an "epoch").

Two placement conditions, per PCTA_Experimental_Methodology.md A.4/B.9:
  - "uniform": Byzantine identities scattered uniformly at random over the whole
    run (the Tier 1 default) -- no special-casing needed, honest per-node uniform
    sampling naturally produces this.
  - "adversarial_clustering": a *one-time*, t=0-only biased draw for a subset of
    pre-selected target honest nodes, then honest unbiased re-randomization for
    every subsequent epoch. This is the worst-case start the Eventual Local
    Balance theorem's recovery guarantee is checked against (Tier 2c) -- it does
    NOT mean the adversary can bias the resampling mechanism itself at any later
    epoch, which Assumption 0 forbids by construction.
"""
import numpy as np


class GossipTopology:
    def __init__(self, N, k, R, seed, byz_ids, placement="uniform",
                 target_honest_frac=0.15, cluster_bias=0.8):
        self.N = N
        self.k = k
        self.R = R  # float('inf') supported: never reshuffle after the t=0 draw
        self.byz_ids = set(int(b) for b in byz_ids)
        self.placement = placement
        self.cluster_bias = cluster_bias
        self.rng = np.random.default_rng(seed)

        if placement == "adversarial_clustering":
            honest_ids = [i for i in range(N) if i not in self.byz_ids]
            n_target = max(1, int(round(target_honest_frac * len(honest_ids))))
            self.target_honest = set(
                self.rng.choice(honest_ids, size=min(n_target, len(honest_ids)), replace=False).tolist()
            )
        else:
            self.target_honest = set()

        self._current = {}
        self._epoch = None
        self._reshuffle(epoch0=True)
        self._epoch = 0

    def _draw_neighbors(self, i, biased):
        pool = np.array([j for j in range(self.N) if j != i])
        k_eff = min(self.k, len(pool))
        if not biased or i not in self.target_honest:
            return self.rng.choice(pool, size=k_eff, replace=False).tolist()

        # Adversarial-clustering bias, t=0 only, target-honest nodes only: force as many
        # Byzantine neighbors as available (up to cluster_bias * k), fill the rest uniformly.
        byz_pool = np.array([j for j in pool if j in self.byz_ids])
        n_forced = min(len(byz_pool), int(round(self.cluster_bias * self.k)))
        forced = self.rng.choice(byz_pool, size=n_forced, replace=False).tolist() if n_forced > 0 else []
        remaining_pool = np.array([j for j in pool if j not in forced])
        n_remaining = k_eff - len(forced)
        rest = self.rng.choice(
            remaining_pool, size=min(n_remaining, len(remaining_pool)), replace=False
        ).tolist() if n_remaining > 0 else []
        return forced + rest

    def _reshuffle(self, epoch0):
        biased = epoch0 and self.placement == "adversarial_clustering"
        self._current = {i: self._draw_neighbors(i, biased=biased) for i in range(self.N)}

    def maybe_reshuffle(self, t):
        if self.R == float("inf"):
            return
        epoch = t // self.R
        if epoch != self._epoch:
            self._epoch = epoch
            self._reshuffle(epoch0=False)

    def neighbors(self, i):
        return self._current[i]

    def local_byzantine_fraction(self, i):
        nbrs = self._current[i]
        if not nbrs:
            return 0.0
        return sum(1 for j in nbrs if j in self.byz_ids) / len(nbrs)

    def all_local_byzantine_fractions(self):
        """{node_id: local Byzantine fraction}, honest nodes only -- for Tier 2c logging
        against PCTA_Theory.md SS4's Eventual Local Balance concentration bound."""
        return {
            i: self.local_byzantine_fraction(i)
            for i in range(self.N) if i not in self.byz_ids
        }


if __name__ == "__main__":
    # Quick interactive sanity check (real correctness checks live in smoke_test.py).
    N, k, f = 100, 10, 0.2
    rng = np.random.default_rng(0)
    byz = rng.choice(N, size=int(f * N), replace=False).tolist()

    topo_u = GossipTopology(N, k, R=5, seed=1, byz_ids=byz, placement="uniform")
    fracs_u = list(topo_u.all_local_byzantine_fractions().values())
    print(f"[uniform] global f={f:.2f}, local mean={np.mean(fracs_u):.3f}, "
          f"max={np.max(fracs_u):.3f}, frac(local>=0.5)={np.mean(np.array(fracs_u) >= 0.5):.3f}")

    topo_c = GossipTopology(N, k, R=5, seed=1, byz_ids=byz, placement="adversarial_clustering")
    fracs_c0 = list(topo_c.all_local_byzantine_fractions().values())
    print(f"[adversarial, t=0] mean={np.mean(fracs_c0):.3f}, max={np.max(fracs_c0):.3f}, "
          f"frac(local>=0.5)={np.mean(np.array(fracs_c0) >= 0.5):.3f}")
    for t in range(1, 21):
        topo_c.maybe_reshuffle(t)
    fracs_c20 = list(topo_c.all_local_byzantine_fractions().values())
    print(f"[adversarial, t=20, post-recovery] mean={np.mean(fracs_c20):.3f}, "
          f"max={np.max(fracs_c20):.3f}, frac(local>=0.5)={np.mean(np.array(fracs_c20) >= 0.5):.3f}")
