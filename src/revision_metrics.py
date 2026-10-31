"""Revision v2 metrics (2026-10-16): threshold-free detection quality and the
metric that matches soft aggregation, computed from a run's *_detect.json /
*_probe.json logs.

  AUROC        -- ranking quality of -trust_weight (higher = attackers ranked less trusted)
  byz_mass     -- share of total aggregation weight given to Byzantine neighbors
  byz_share    -- Byzantine share of (observer, neighbor, round) observations (byz_mass's no-defense value)
  probe_tpr/fpr-- fraction of probes on malicious / honest targets returning "dishonest"

Usage: python revision_metrics.py <results_dir> [glob]
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def detect_metrics(detect_path, t_min=1):
    log = json.load(open(detect_path))
    log = [e for e in log if e["t"] >= t_min]
    log = [e for e in log if e["trust_weight"] == e["trust_weight"]]   # drop NaN (state-based baselines)
    if not log:
        return {}
    y = np.array([e["malicious"] for e in log], dtype=bool)
    w = np.array([e["trust_weight"] for e in log], dtype=float)
    out = {"byz_share": float(y.mean()) if len(y) else None}
    if y.any() and (~y).any():
        out["AUROC"] = float(roc_auc_score(y, -w))
        out["byz_mass"] = float(w[y].sum() / max(w.sum(), 1e-12))
    fl = [e for e in log if "flagged" in e]
    if fl:
        yf = np.array([e["malicious"] for e in fl], dtype=bool)
        ff = np.array([e["flagged"] for e in fl], dtype=bool)
        out["FPR_flag"] = float(ff[~yf].mean()) if (~yf).any() else None
        out["TPR_flag"] = float(ff[yf].mean()) if yf.any() else None
    return out


def probe_metrics(probe_path):
    if not os.path.exists(probe_path):
        return {}
    log = json.load(open(probe_path))
    mal = [not e["verdict_honest"] for e in log if e["malicious"]]
    hon = [not e["verdict_honest"] for e in log if not e["malicious"]]
    return {"probe_tpr": float(np.mean(mal)) if mal else None,
            "probe_fpr": float(np.mean(hon)) if hon else None,
            "n_probes": len(log)}


def summarize(results_dir, pattern="*.json"):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, pattern))):
        if f.endswith(("_detect.json", "_probe.json")) or "_progress" in f:
            continue
        r = json.load(open(f))
        if "acc_log" not in r:
            continue
        base = f[:-5]
        row = {"tag": os.path.basename(base), "final_acc": r["acc_log"][-1]["mean_acc"]}
        if os.path.exists(base + "_detect.json"):
            t_min = 30 if r.get("attack") == "slow_poison" else 1
            row.update(detect_metrics(base + "_detect.json", t_min))
        row.update(probe_metrics(base + "_probe.json"))
        rows.append(row)
    return rows


if __name__ == "__main__":
    for row in summarize(*sys.argv[1:]):
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
