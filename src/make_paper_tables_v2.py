"""Revision v2 (2026-10-31): generate the manuscript's LaTeX tables directly from
results_v2/_tables/tier{A,C,D}.csv and the significance tests, so every number
in latex/main_v2.tex is traceable to a result file.

Usage: python make_paper_tables_v2.py   (writes ../latex/tables_v2/*.tex)
"""
import csv
import json
import os

import numpy as np
from scipy import stats

from revision_metrics import detect_metrics

TAB = os.path.join("results_v2", "_tables")
OUT = os.path.join("..", "latex", "tables_v2")
os.makedirs(OUT, exist_ok=True)

METHODS = ["fedavg", "local_krum", "trimmed_mean", "median", "foolsgold", "fltrust",
           "openclaw", "clipped_gossip", "balance", "ios", "pcta_soft"]
MNAME = {"fedavg": "FedAvg", "local_krum": "Krum", "trimmed_mean": "TrMean", "median": "Median",
         "foolsgold": "FoolsGold", "fltrust": "FLTrust", "openclaw": "OpenCLAW",
         "clipped_gossip": "ClipGossip", "balance": "BALANCE", "ios": "IOS", "pcta_soft": "PCTA"}
ATTACKS = ["clean", "gaussian_noise", "sign_flip", "sybil", "slow_poison", "alie", "ipm",
           "dissensus", "label_flip", "label_flip_targeted", "calibrated_adaptive"]
ANAME = {"clean": "No attack", "gaussian_noise": "Gaussian", "sign_flip": "Sign-flip", "sybil": "Sybil",
         "slow_poison": "Slow poison", "alie": "ALIE", "ipm": "IPM", "dissensus": "Dissensus",
         "label_flip": "Label flip", "label_flip_targeted": "Targeted flip",
         "calibrated_adaptive": "Adaptive"}
DNAME = {"cifar10": "CIFAR-10", "fashion_mnist": "Fashion-MNIST"}


def load(tier):
    rows = list(csv.DictReader(open(os.path.join(TAB, f"tier{tier}.csv"))))
    f = lambda v: float(v) if v not in ("", None) else None
    for r in rows:
        for k in list(r):
            if k not in ("tier", "dataset", "attack", "method"):
                r[k] = f(r[k])
    return rows


def run_value(tag, metric="final_acc"):
    p = os.path.join("results_v2", tag + ".json")
    r = json.load(open(p))
    if metric == "final_acc":
        return r["acc_log"][-1]["mean_acc"]
    return detect_metrics(p[:-5] + "_detect.json", 1).get(metric)


def tag_A(ds, m, a, s):
    f = 0.0 if a == "clean" else 0.2
    atk = a
    return f"A_{ds}_{m}_{atk}_f{f}_a0.1_T40_s{s}"


def main_accuracy_table(rows):
    """Final accuracy, every method x attack, one block per dataset; best mean bold,
    and a dagger where PCTA is significantly better than the best baseline (paired t, n=3)."""
    idx = {(r["dataset"], r["attack"], r["method"]): r for r in rows}
    lines = [r"\begin{tabular}{@{}l" + "c" * len(METHODS) + "@{}}", r"\toprule",
             "Attack & " + " & ".join(MNAME[m] for m in METHODS) + r" \\", r"\midrule"]
    for ds in ("cifar10", "fashion_mnist"):
        lines.append(r"\multicolumn{%d}{@{}l}{\emph{%s}} \\" % (len(METHODS) + 1, DNAME[ds]))
        for a in ATTACKS:
            vals = {m: idx[(ds, a, m)]["final_acc"] for m in METHODS if (ds, a, m) in idx}
            best = max(vals.values())
            base_best = max((v, m) for m, v in vals.items() if m != "pcta_soft")[1]
            pa = [run_value(tag_A(ds, "pcta_soft", a, s)) for s in (10, 11, 12)]
            pb = [run_value(tag_A(ds, base_best, a, s)) for s in (10, 11, 12)]
            sig = stats.ttest_rel(pa, pb).pvalue < 0.05 and np.mean(pa) > np.mean(pb)
            cells = []
            for m in METHODS:
                v = vals[m]
                c = f"{v:.3f}"
                if abs(v - best) < 5e-4:
                    c = r"\textbf{" + c + "}"
                if m == "pcta_soft" and sig:
                    c += r"$^\dagger$"
                cells.append(c)
            lines.append(ANAME[a] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if ds == "cifar10" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(OUT, "main_accuracy.tex"), "w").write("\n".join(lines) + "\n")


def detection_table(rows):
    """PCTA detection per attack: AUROC of -trust, Byzantine weight mass, flag FPR/TPR;
    Krum's AUROC/mass (0/1 acceptance) alongside."""
    idx = {(r["dataset"], r["attack"], r["method"]): r for r in rows}
    fmt = lambda v, p=3: "--" if v is None else f"{v:.{p}f}"
    pct = lambda v: "--" if v is None else f"{100 * v:.1f}"
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{CIFAR-10} & \multicolumn{4}{c}{Fashion-MNIST} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Attack & AUROC & mass & FPR & TPR & AUROC & mass & FPR & TPR \\",
             r" & {\scriptsize (Krum)} & {\scriptsize (Krum)} & (\%) & (\%) & {\scriptsize (Krum)} & {\scriptsize (Krum)} & (\%) & (\%) \\",
             r"\midrule"]
    for a in ATTACKS:
        cells = []
        for ds in ("cifar10", "fashion_mnist"):
            p, k = idx[(ds, a, "pcta_soft")], idx.get((ds, a, "local_krum"), {})
            if a == "clean":
                cells += ["--", "--", pct(p["FPR_flag"]), "--"]
            else:
                cells += [f"{fmt(p['AUROC'])} {{\\scriptsize ({fmt(k.get('AUROC'), 2)})}}",
                          f"{fmt(p['byz_mass'])} {{\\scriptsize ({fmt(k.get('byz_mass'), 2)})}}",
                          pct(p["FPR_flag"]), pct(p["TPR_flag"])]
        lines.append(ANAME[a] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "detection.tex"), "w").write("\n".join(lines) + "\n")


def significance_table():
    """Overall paired comparison PCTA vs each baseline over 66 (dataset x attack x seed) pairs."""
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Baseline & mean $\Delta$acc & PCTA wins & Wilcoxon $p$ & cells better & cells worse \\",
             r"\midrule"]
    for m in METHODS[:-1]:
        d, cb, cw = [], 0, 0
        for ds in ("cifar10", "fashion_mnist"):
            for a in ATTACKS:
                pa = [run_value(tag_A(ds, "pcta_soft", a, s)) for s in (10, 11, 12)]
                pb = [run_value(tag_A(ds, m, a, s)) for s in (10, 11, 12)]
                d += [x - y for x, y in zip(pa, pb)]
                diff = np.array(pa) - np.array(pb)
                if np.allclose(diff, diff[0]):
                    continue
                pv = stats.ttest_rel(pa, pb).pvalue
                if pv < 0.05:
                    cb += diff.mean() > 0
                    cw += diff.mean() < 0
        p = stats.wilcoxon(d).pvalue
        lines.append(f"{MNAME[m]} & {np.mean(d):+.3f} & {100 * np.mean(np.array(d) > 0):.0f}\\% & "
                     f"{p:.1e} & {cb} & {cw} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "significance.tex"), "w").write("\n".join(lines) + "\n")


ABL = [("full", "Full PCTA"), ("no_traj", "$-$ trajectory channel"), ("no_data", "$-$ data channel"),
       ("no_cusum", "$-$ CUSUM (memoryless)"), ("renorm_mixing", "renormalized mixing"),
       ("no_norm_bound", "$-$ norm bound"), ("no_window_gate", "$-$ window gate"),
       ("no_reputation", "$-$ Beta reputation"), ("no_probation", "$-$ probation"),
       ("ema_predictor", "EMA instead of Kalman")]
ABL_ATK = ["clean", "sign_flip", "label_flip_targeted", "sybil", "alie"]


def ablation_table(rows):
    idx = {(r["attack"], r["method"]): r for r in rows}
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             "Configuration & " + " & ".join(ANAME[a] for a in ABL_ATK) + r" \\", r"\midrule"]
    for v, name in ABL:
        cells = []
        for a in ABL_ATK:
            r = idx[(a, f"pcta_soft-{v}")]
            if a == "clean":
                cells.append(f"{r['final_acc']:.3f} / {100 * r['FPR_flag']:.1f}\\%")
            else:
                cells.append(f"{r['final_acc']:.3f} / {r['AUROC']:.2f}")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "ablation.tex"), "w").write("\n".join(lines) + "\n")


def whitewash_table(rows):
    idx = {(r["dataset"], r["method"]): r for r in rows if r["attack"] == "sign_flip"}
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{CIFAR-10} & \multicolumn{3}{c}{Fashion-MNIST} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & $W{=}3$ & $W{=}5$ & $W{=}10$ & $W{=}3$ & $W{=}5$ & $W{=}10$ \\", r"\midrule"]
    for m, name in [("pcta_soft-ww{W}", "PCTA"), ("pcta_soft-no_probation-ww{W}", "PCTA $-$ probation"),
                    ("local_krum-ww{W}", "Krum")]:
        cells = []
        for ds in ("cifar10", "fashion_mnist"):
            for W in (3, 5, 10):
                r = idx[(ds, m.format(W=W))]
                cells.append(f"{r['final_acc']:.3f} / {r['byz_mass']:.3f}")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "whitewash.tex"), "w").write("\n".join(lines) + "\n")


def floor_table(rows):
    idx = {(r["dataset"], r["method"]): r for r in rows if r["attack"] == "calibrated_adaptive"}
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{CIFAR-10} & \multicolumn{4}{c}{Fashion-MNIST} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Method & 0.5 & 1 & 2 & 5 & 0.5 & 1 & 2 & 5 \\", r"\midrule"]
    for m, name in [("pcta_soft", "PCTA"), ("local_krum", "Krum")]:
        for metric, mname in [("final_acc", "acc"), ("AUROC", "AUROC")]:
            cells = [f"{idx[(ds, f'{m}-ratio{r}')][metric]:.3f}"
                     for ds in ("cifar10", "fashion_mnist") for r in (0.5, 1.0, 2.0, 5.0)]
            lines.append(f"{name} {mname} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "floor.tex"), "w").write("\n".join(lines) + "\n")


def whitewash_tests():
    out = []
    for other, label in [("pcta_soft-no_probation-ww{W}", "no_probation"), ("local_krum-ww{W}", "krum")]:
        da, dm = [], []
        for ds in ("cifar10", "fashion_mnist"):
            for W in (3, 5, 10):
                for s in (10, 11, 12):
                    t = lambda m: f"D_{ds}_{m.format(W=W)}_sign_flip_f0.2_a0.1_T40_s{s}"
                    da.append(run_value(t("pcta_soft-ww{W}")) - run_value(t(other)))
                    dm.append(run_value(t("pcta_soft-ww{W}"), "byz_mass") - run_value(t(other), "byz_mass"))
        out.append(f"PCTA vs {label}: acc {np.mean(da):+.4f} wins {sum(x > 0 for x in da)}/18 "
                   f"p={stats.wilcoxon(da).pvalue:.2g}; byz_mass {np.mean(dm):+.4f} "
                   f"lower {sum(x < 0 for x in dm)}/18 p={stats.wilcoxon(dm).pvalue:.2g}")
    open(os.path.join(OUT, "whitewash_tests.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    A, C, D = load("A"), load("C"), load("D")
    main_accuracy_table(A)
    detection_table(A)
    significance_table()
    ablation_table(C)
    whitewash_table(D)
    floor_table(D)
    whitewash_tests()
    print("tables ->", os.path.abspath(OUT))
