"""Revision v2 analysis (2026-10-17): aggregates results_v2/ into per-config
tables over seeds -- final accuracy (mean +/- 95% t-CI), AUROC of -trust_weight,
Byzantine weight mass, hard-flag FPR/TPR -- as CSV + Markdown per tier.

Usage: python analyze_v2.py [results_dir] [out_dir]
"""
import collections
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

from revision_metrics import detect_metrics

METRICS = ["final_acc", "AUROC", "byz_mass", "FPR_flag", "TPR_flag"]


def label_of(r, tag):
    tier = tag.split("_", 1)[0]
    attack = "clean" if r["byz_fraction"] == 0 else r["attack"]
    name = tag.split(f"{tier}_{r['dataset']}_", 1)[1].rsplit(f"_{attack}_f", 1)[0]
    return tier, attack, name


def ci95(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None, None, 0
    if len(v) == 1:
        return float(v[0]), 0.0, 1
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))), len(v)


def collect(results_dir):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if f.endswith(("_detect.json", "_probe.json")) or "_progress" in f:
            continue
        r = json.load(open(f))
        if "acc_log" not in r:
            continue
        tag = os.path.basename(f)[:-5]
        tier, attack, name = label_of(r, tag)
        row = {"final_acc": r["acc_log"][-1]["mean_acc"]}
        det = f[:-5] + "_detect.json"
        if os.path.exists(det):
            t_min = r.get("attack_params", {}).get("ramp_start", 10) if attack == "slow_poison" else 1
            row.update(detect_metrics(det, t_min))
        key = (tier, r["dataset"], r["alpha"], r["byz_fraction"], attack, name)
        for m in METRICS:
            groups[key][m].append(row.get(m))
    return groups


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(results_dir, "_tables")
    os.makedirs(out_dir, exist_ok=True)
    groups = collect(results_dir)
    by_tier = collections.defaultdict(list)
    for key, g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(["tier", "dataset", "alpha", "f", "attack", "method"], key))
        for m in METRICS:
            mean, half, n = ci95(g[m])
            row[m], row[m + "_ci"] = mean, half
            row["n_seeds"] = max(row.get("n_seeds", 0), n)
        by_tier[key[0]].append(row)
    for tier, rows in by_tier.items():
        with open(os.path.join(out_dir, f"tier{tier}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(out_dir, f"tier{tier}.md"), "w", encoding="utf-8") as fh:
            fh.write("| dataset | α | f | attack | method | n | acc | AUROC | byz mass | FPR | TPR |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
            fmt = lambda r, m: "–" if r[m] is None else (f"{r[m]:.3f}" + (f" ± {r[m + '_ci']:.3f}" if r[m + "_ci"] else ""))
            for r in rows:
                fh.write(f"| {r['dataset']} | {r['alpha']} | {r['f']} | {r['attack']} | {r['method']} | {r['n_seeds']} | "
                         + " | ".join(fmt(r, m) for m in METRICS) + " |\n")
        print(f"tier {tier}: {len(rows)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
