"""Paired significance tests, PCTA vs each baseline (2026-10-31). Pairs share
dataset/attack/seed, i.e. identical partitions, Byzantine ids and topology.
  per cell : paired t-test over seeds (n=3, low power -- only large gaps pass)
  overall  : Wilcoxon signed-rank over all (dataset, attack, seed) triples
Usage: python sig_tests_v2.py [results_dir]  -> prints + writes <dir>/_tables/significance.md"""
import collections, glob, json, os, sys
import numpy as np
from scipy import stats
D = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
acc = {}
for f in glob.glob(os.path.join(D, "A_*.json")):
    if f.endswith(("_detect.json", "_probe.json")):
        continue
    r = json.load(open(f))
    attack = "clean" if r["byz_fraction"] == 0 else r["attack"]
    acc[(r["dataset"], attack, r["method"], r["seed"])] = r["acc_log"][-1]["mean_acc"]
methods = sorted({k[2] for k in acc} - {"pcta_soft"})
cells = sorted({(k[0], k[1]) for k in acc})
out = ["## PCTA vs baseline: overall (Wilcoxon over all dataset x attack x seed pairs)", "",
       "| baseline | n | mean Δacc | PCTA wins | p (two-sided) | cells PCTA sig. better | cells sig. worse |", "|---|---|---|---|---|---|---|"]
percell = collections.defaultdict(dict)
for m in methods:
    diffs, better, worse = [], 0, 0
    for ds, a in cells:
        seeds = sorted(s for (d_, a_, m_, s) in acc if (d_, a_, m_) == (ds, a, "pcta_soft"))
        p = [acc[(ds, a, "pcta_soft", s)] for s in seeds if (ds, a, m, s) in acc]
        b = [acc[(ds, a, m, s)] for s in seeds if (ds, a, m, s) in acc]
        if len(p) < 2:
            continue
        d = np.array(p) - np.array(b)
        diffs += list(d)
        pv = stats.ttest_rel(p, b).pvalue if np.std(d) > 0 else (0.0 if d.mean() != 0 else 1.0)
        percell[(ds, a)][m] = (d.mean(), pv)
        if pv < 0.05:
            better += d.mean() > 0
            worse += d.mean() < 0
    diffs = np.array(diffs)
    w = stats.wilcoxon(diffs).pvalue if np.any(diffs != 0) else 1.0
    out.append(f"| {m} | {len(diffs)} | {diffs.mean():+.3f} | {np.mean(diffs > 0):.0%} | {w:.1e} | {better} | {worse} |")
out += ["", "## Per cell: mean Δacc (PCTA − baseline), * = paired t p<0.05 (n=3)", "",
        "| dataset | attack | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
for (ds, a) in cells:
    row = [f"{percell[(ds, a)][m][0]:+.3f}{'*' if percell[(ds, a)][m][1] < 0.05 else ''}" if m in percell[(ds, a)] else "–"
           for m in methods]
    out.append(f"| {ds} | {a} | " + " | ".join(row) + " |")
txt = "\n".join(out)
print(txt)
os.makedirs(os.path.join(D, "_tables"), exist_ok=True)
open(os.path.join(D, "_tables", "significance.md"), "w", encoding="utf-8").write(txt + "\n")
