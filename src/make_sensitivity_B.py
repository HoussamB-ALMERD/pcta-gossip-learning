"""Tier B sensitivity (2026-11-03): paired tests PCTA vs the four strong baselines
for each (dataset, alpha, f) setting, plus an overall Wilcoxon per baseline.
Pairs share dataset/alpha/f/attack/seed (identical partitions, Byzantine ids, topology).
The no-attack runs (f=0) of alpha=0.03/0.5 are grouped with that alpha's f=0.2 setting.

Usage: python make_sensitivity_B.py
  -> results_v2/_tables_withB/significance_B.md, ../latex/tables_v2/sensitivity.tex"""
import collections
import glob
import json
import os

import numpy as np
from scipy import stats

D = "results_v2"
BASE = ["local_krum", "clipped_gossip", "balance", "ios"]
MNAME = {"local_krum": "Krum", "clipped_gossip": "ClipGossip", "balance": "BALANCE", "ios": "IOS"}
DNAME = {"cifar10": "CIFAR-10", "fashion_mnist": "Fashion-MNIST"}
SETTINGS = [(0.03, 0.2), (0.5, 0.2), (0.1, 0.1), (0.1, 0.3)]

acc = {}
for fn in glob.glob(os.path.join(D, "B_*.json")):
    if fn.endswith(("_detect.json", "_probe.json")):
        continue
    r = json.load(open(fn))
    f = r["byz_fraction"]
    attack = "clean" if f == 0 else r["attack"]
    setting = (r["alpha"], 0.2 if f == 0 else f)
    acc[(r["dataset"], setting, attack, r["method"], r["seed"])] = r["acc_log"][-1]["mean_acc"]

fpr = {}
for row in __import__("csv").DictReader(open(os.path.join(D, "_tables_withB", "tierB.csv"))):
    if row["method"] == "pcta_soft" and row["FPR_flag"]:
        f = float(row["f"])
        key = (row["dataset"], (float(row["alpha"]), 0.2 if f == 0 else f))
        fpr.setdefault(key, []).append(float(row["FPR_flag"]))


def pairs(ds, st, atk, m):
    seeds = sorted(k[4] for k in acc if k[:4] == (ds, st, atk, "pcta_soft"))
    return ([acc[(ds, st, atk, "pcta_soft", s)] for s in seeds if (ds, st, atk, m, s) in acc],
            [acc[(ds, st, atk, m, s)] for s in seeds if (ds, st, atk, m, s) in acc])


md = ["# Tier B: PCTA − baseline, mean Δacc per cell (* = paired t p<0.05, n=3)", "",
      "| dataset | alpha | f | attack | PCTA | " + " | ".join(MNAME[m] for m in BASE) + " |",
      "|---|---|---|---|---|" + "---|" * len(BASE)]
summary = {}
overall = collections.defaultdict(list)
for ds in DNAME:
    for st in SETTINGS:
        atks = sorted({k[2] for k in acc if k[0] == ds and k[1] == st})
        cell = {m: [[], 0, 0] for m in BASE}
        pacc = []
        for a in atks:
            p0, _ = pairs(ds, st, a, "pcta_soft")
            pacc.append(np.mean(p0))
            row = []
            for m in BASE:
                p, b = pairs(ds, st, a, m)
                d = np.array(p) - np.array(b)
                cell[m][0] += list(d)
                overall[m] += list(d)
                pv = stats.ttest_rel(p, b).pvalue if np.std(d) > 0 else (0.0 if d.mean() else 1.0)
                if pv < 0.05:
                    cell[m][1] += d.mean() > 0
                    cell[m][2] += d.mean() < 0
                row.append(f"{d.mean():+.3f}{'*' if pv < 0.05 else ''}")
            md.append(f"| {ds} | {st[0]} | {st[1]} | {a} | {np.mean(p0):.3f} | " + " | ".join(row) + " |")
        summary[(ds, st)] = (np.mean(pacc), len(atks), cell)

md += ["", "## Overall (Wilcoxon over all Tier B pairs)", "", "| baseline | n | mean Δacc | PCTA wins | p |",
       "|---|---|---|---|---|"]
ov = {}
for m in BASE:
    d = np.array(overall[m])
    p = stats.wilcoxon(d).pvalue
    ov[m] = (d.mean(), p, len(d))
    md.append(f"| {MNAME[m]} | {len(d)} | {d.mean():+.3f} | {np.mean(d > 0):.0%} | {p:.1e} |")
txt = "\n".join(md)
print(txt)
open(os.path.join(D, "_tables_withB", "significance_B.md"), "w", encoding="utf-8").write(txt + "\n")

# LaTeX: one row per (dataset, setting); per baseline "mean Δ (better/worse cells)".
L = [r"\begin{tabular}{@{}llcc" + "c" * len(BASE) + r"@{}}", r"\toprule",
     r" & Setting & PCTA acc & honest FPR & " + " & ".join(MNAME[m] for m in BASE) + r" \\", r"\midrule"]
for ds in DNAME:
    for i, st in enumerate(SETTINGS):
        pa, n, cell = summary[(ds, st)]
        fr = fpr.get((ds, st), [0])
        cols = [f"${np.mean(cell[m][0]):+.3f}$ ({cell[m][1]}/{cell[m][2]})" for m in BASE]
        name = DNAME[ds] if i == 0 else ""
        L.append(f"{name} & $\\alpha={st[0]}$, $f={st[1]}$ & {pa:.3f} & "
                 f"{100 * min(fr):.1f}--{100 * max(fr):.1f}\\% & " + " & ".join(cols) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\midrule"
L.append(r"\multicolumn{4}{@{}l}{Overall mean $\Delta$acc (Wilcoxon $p$, " + f"{ov[BASE[0]][2]}" + r" pairs)} & "
         + " & ".join(f"${ov[m][0]:+.3f}$ ({ov[m][1]:.0e})" for m in BASE) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
out = os.path.join("..", "latex", "tables_v2", "sensitivity.tex")
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out)
