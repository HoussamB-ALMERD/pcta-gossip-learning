"""Revision v2 (2026-10-31): manuscript figures from results_v2/ (Tier D) and the
gossip sampler. Writes ../latex/figures_v2/*.pdf. Also prints the local
Byzantine-concentration statistics quoted in the paper (Section 4.1).

Usage: python make_paper_figures_v2.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hypergeom

from topology import GossipTopology

OUT = os.path.join("..", "latex", "figures_v2")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
DS = [("cifar10", "CIFAR-10"), ("fashion_mnist", "Fashion-MNIST")]
SEEDS = (10, 11, 12)


def floor_figure():
    rows = [r for r in csv.DictReader(open(os.path.join("results_v2", "_tables", "tierD.csv")))
            if r["attack"] == "calibrated_adaptive"]
    idx = {(r["dataset"], r["method"]): r for r in rows}
    ratios = [0.5, 1.0, 2.0, 5.0]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, (ds, dn) in zip(axes, DS):
        for m, lab, c in [("pcta_soft", "PCTA", "C0"), ("local_krum", "Krum", "C1")]:
            au = [float(idx[(ds, f"{m}-ratio{r}")]["AUROC"]) for r in ratios]
            ac = [float(idx[(ds, f"{m}-ratio{r}")]["final_acc"]) for r in ratios]
            ax.plot(ratios, au, "-o", color=c, label=f"{lab} AUROC", ms=3)
            ax.plot(ratios, ac, "--s", color=c, label=f"{lab} accuracy", ms=3)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(ratios)
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.set_xlabel("attack strength ratio $\\rho$")
        ax.set_title(dn)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_detectability_floor.pdf"))


def byz_trust_per_round(tag):
    log = json.load(open(os.path.join("results_v2", tag + "_detect.json")))
    T = max(e["t"] for e in log)
    s, n = np.zeros(T + 1), np.zeros(T + 1)
    for e in log:
        if e["malicious"]:
            s[e["t"]] += e["trust_weight"]
            n[e["t"]] += 1
    return np.arange(1, T + 1), s[1:] / np.maximum(n[1:], 1)


def whitewash_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for ax, (ds, dn) in zip(axes, DS):
        for m, lab, c in [("pcta_soft-ww5", "PCTA", "C0"), ("pcta_soft-no_probation-ww5", "PCTA $-$ probation", "C3")]:
            curves = [byz_trust_per_round(f"D_{ds}_{m}_sign_flip_f0.2_a0.1_T40_s{s}")[1] for s in SEEDS]
            t = np.arange(1, len(curves[0]) + 1)
            ax.plot(t, np.mean(curves, axis=0), color=c, label=lab, lw=1.2)
        for r in range(5, 41, 5):   # topology re-sampling (t // R changes) -> new, memoryless pairs
            ax.axvline(r, color="grey", lw=0.5, ls=":")
        for r in range(6, 41, 5):   # whitewashing identity reset ((t - 1) % W == 0)
            ax.axvline(r, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel("round")
        ax.set_title(dn)
    axes[0].set_ylabel("mean trust weight of\nByzantine neighbors")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_whitewashing.pdf"))


def local_concentration():
    N, k, R, T = 100, 10, 5, 40
    for s in SEEDS:
        rng = np.random.default_rng(s + 777)   # same Byzantine-id draw as harness.run_experiment
        byz = set(rng.choice(N, size=20, replace=False).tolist())
        top = GossipTopology(N, k, R, seed=s, byz_ids=byz)
        c3 = c5 = n = 0
        for t in range(1, T + 1):
            top.maybe_reshuffle(t)
            for i in range(N):
                if i in byz:
                    continue
                b = sum(j in byz for j in top.neighbors(i))
                n += 1
                c3 += b >= 3
                c5 += b >= 5
        print(f"seed {s}: P(b>=3)={c3 / n:.3f}  P(b>=k/2)={c5 / n:.4f}")
    print(f"hypergeometric: P(b>=3)={hypergeom.sf(2, 99, 20, 10):.3f}  P(b>=5)={hypergeom.sf(4, 99, 20, 10):.4f}; "
          f"Hoeffding bound (eps=0.3) {np.exp(-2 * 0.09 * 10):.3f}")


if __name__ == "__main__":
    floor_figure()
    whitewash_figure()
    local_concentration()
    print("figures ->", os.path.abspath(OUT))


def pipeline_figure():
    """Schematic of the per-neighbor pipeline (Section 4)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 32)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))

    box(0, 12, 11, 8, "message\n$(x_j, \Delta_j)$", "#eeeeee")
    box(16, 21, 20, 9, "trajectory channel\npredictor innovation\n$\log Z_{ij}$", "#d6e6f5")
    box(16, 2, 20, 9, "data channel (ref. set)\nlevel $g_j$, contrast $L_j, A_j$", "#f5e3cf")
    box(41, 12, 15, 8, "robust $z$ vs.\nobserver's window\n(median / MAD)", "#eeeeee")
    box(61, 12, 14, 8, "3 CUSUMs\n$S^T, S^L, S^C$\n(per pair, persistent)", "#e3f0dc")
    box(80, 21, 19, 9, "trust $w_{ij}$\n$= m_{ij}\, e^{-\Lambda_{ij}}$\n(+ identity-age probation)", "#e3f0dc")
    box(80, 2, 19, 9, "bounded-leverage\nabsolute mixing\n(clip at " + r"$\tau_i$" + ", rest on self)", "#e8dff2")
    arrow(11, 17, 16, 25)
    arrow(11, 15, 16, 7)
    arrow(36, 25, 41, 18)
    arrow(36, 7, 41, 14)
    arrow(56, 16, 61, 16)
    arrow(75, 17, 80, 24)
    arrow(89.5, 21, 89.5, 11)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"))
