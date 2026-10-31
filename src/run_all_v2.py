"""Revision v2 full-matrix driver (2026-10-17). Run ONLY on the frozen design
(archive_v2_frozen/). Seeds 0 and 2 informed design decisions (dev + held-out
fixes F14/F15), so every tier here uses SEEDS = 10.. -- never 0 or 2.

Tiers:
  A  main comparison: all methods x all attacks, alpha=0.1, f=0.2, both datasets
  B  sensitivity: alpha in {0.03, 0.5}, f in {0.1, 0.3}, strongest baselines only
  C  ablations: PCTA components (channels, CUSUM, reputation, mixing, predictor)
  D  whitewashing (identity churn) + Detectability Floor (calibrated_adaptive sweep)

Resumable (skips existing result files), fault-isolated, progress in _progress.json.

Usage: python run_all_v2.py [--tiers A,B,C,D] [--seeds 3] [--dry-run]
"""
import argparse
import itertools
import json
import os
import time
import traceback

OUT_DIR = "results_v2"
N, K, R = 100, 10, 5
T = 40                      # matches the dev/held-out horizon; changing it is a budget decision
SEED_POOL = [10, 11, 12, 13, 14]

DATASETS = ["cifar10", "fashion_mnist"]
ALL_METHODS = ["fedavg", "local_krum", "trimmed_mean", "median", "foolsgold", "fltrust",
               "openclaw", "clipped_gossip", "balance", "ios", "pcta_soft"]
STRONG_METHODS = ["local_krum", "clipped_gossip", "balance", "ios", "pcta_soft"]
ATTACKS = ["clean", "gaussian_noise", "sign_flip", "sybil", "slow_poison", "alie", "ipm",
           "dissensus", "label_flip", "label_flip_targeted", "calibrated_adaptive"]
CORE_ATTACKS = ["clean", "sign_flip", "sybil", "alie", "ipm", "label_flip_targeted"]

ABLATIONS = {
    "full": {},
    "no_traj": {"use_traj": False},
    "no_data": {"use_data": False},
    "no_cusum": {"use_cusum": False},
    "no_reputation": {"use_reputation": False},
    "renorm_mixing": {"residual_self": False},          # F14 off (v1 mixing)
    "no_window_gate": {"gate_windows": False},          # F17 off
    "no_norm_bound": {"norm_bound": False},             # F18 off
    "ema_predictor": {"predictor_kind": "ema"},
    "no_probation": {"cold_start_fix": False},
}
ABLATION_ATTACKS = ["clean", "sign_flip", "sybil", "label_flip_targeted", "alie"]


def attack_params_for(attack):
    if attack == "calibrated_adaptive":
        return {"lambda_target": 2.0}
    if attack == "slow_poison":
        return {"ramp_start": 10, "ramp_end": 30}   # v1's 30-90 ramp doesn't fit T=40
    return {}


def cfg(dataset, method, attack, f, alpha, seed, tier, variant=None, vname=None, extra_params=None):
    atk = "gaussian_noise" if attack == "clean" else attack
    f = 0.0 if attack == "clean" else f
    params = dict(attack_params_for(attack), **(extra_params or {}))
    name = method if vname is None else f"{method}-{vname}"
    tag = f"{tier}_{dataset}_{name}_{attack}_f{f}_a{alpha}_T{T}_s{seed}"
    return dict(dataset=dataset, method=method, attack=atk, byz_fraction=f, alpha=alpha, seed=seed,
                tag=tag, attack_params=params, variant=variant, tier=tier)


def tier_A(seeds):
    return [cfg(ds, m, a, 0.2, 0.1, s, "A")
            for ds, m, a, s in itertools.product(DATASETS, ALL_METHODS, ATTACKS, seeds)]


def tier_B(seeds):
    out = []
    for ds, m, a, s in itertools.product(DATASETS, STRONG_METHODS, CORE_ATTACKS, seeds):
        for alpha in (0.03, 0.5):
            out.append(cfg(ds, m, a, 0.2, alpha, s, "B"))
        if a != "clean":
            for f in (0.1, 0.3):
                out.append(cfg(ds, m, a, f, 0.1, s, "B"))
    return out


def tier_C(seeds):
    return [cfg("cifar10", "pcta_soft", a, 0.2, 0.1, s, "C", variant=v, vname=vn)
            for (vn, v), a, s in itertools.product(ABLATIONS.items(), ABLATION_ATTACKS, seeds)]


def tier_D(seeds):
    out = []
    for ds, s in itertools.product(DATASETS, seeds):
        # Whitewashing: sign-flippers rejoin under a fresh identity every W rounds.
        for W in (3, 5, 10):
            for m, v, vn in [("pcta_soft", None, None), ("pcta_soft", {"cold_start_fix": False}, "no_probation"),
                             ("local_krum", None, None)]:
                out.append(cfg(ds, m, "sign_flip", 0.2, 0.1, s, "D", variant=v,
                               vname=(f"{vn}-ww{W}" if vn else f"ww{W}"),
                               extra_params={"whitewash_every": W}))
        # Detectability Floor: calibrated_adaptive strength sweep.
        for ratio in (0.5, 1.0, 2.0, 5.0):
            for m in ("pcta_soft", "local_krum"):
                out.append(cfg(ds, m, "calibrated_adaptive", 0.2, 0.1, s, "D", vname=f"ratio{ratio}",
                               extra_params={"lambda_target": ratio}))
    return out


TIERS = {"A": tier_A, "B": tier_B, "C": tier_C, "D": tier_D}


def estimate(configs, timing_dirs=("results_fix_check_f15", "results_heldout_check")):
    """Rough wall-clock estimate from wall_clock_s of finished dev/held-out runs."""
    per = {}
    for d in timing_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".json") and "_detect" not in fn and "_probe" not in fn and "_progress" not in fn:
                r = json.load(open(os.path.join(d, fn)))
                if "wall_clock_s" in r and "T" in r:
                    per.setdefault((r.get("dataset"), r.get("method")), []).append(r["wall_clock_s"] / r["T"])
    fallback = sum(sum(v) / len(v) for v in per.values()) / max(len(per), 1) if per else 8.0
    tot = 0.0
    for c in configs:
        v = per.get((c["dataset"], c["method"]))
        tot += (sum(v) / len(v) if v else fallback) * T
    return tot / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="A,B,C,D")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seeds = SEED_POOL[:a.seeds]
    assert not {0, 2} & set(seeds), "seeds 0/2 informed the design"
    configs = []
    for t in a.tiers.split(","):
        c = TIERS[t](seeds)
        print(f"Tier {t}: {len(c)} runs, ~{estimate(c):.1f} h")
        configs += c
    print(f"Total: {len(configs)} runs, ~{estimate(configs):.1f} h (~{estimate(configs) / 24:.1f} days)")
    if a.dry_run:
        return

    from harness import run_experiment
    os.makedirs(OUT_DIR, exist_ok=True)
    progress = os.path.join(OUT_DIR, "_progress.json")
    done, failed = [], []
    t0 = time.time()
    for idx, c in enumerate(configs):
        if os.path.exists(os.path.join(OUT_DIR, f"{c['tag']}.json")):
            done.append(c["tag"])
            continue
        run = {k: v for k, v in c.items() if k != "tier"}
        try:
            run_experiment(N=N, k=K, T=T, R=R, out_dir=OUT_DIR, quiet=True, **run)
            done.append(c["tag"])
            print(f"[{idx + 1}/{len(configs)}] {c['tag']} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
        except Exception as e:
            failed.append({"tag": c["tag"], "error": f"{type(e).__name__}: {e}"})
            print(f"[FAIL {idx + 1}/{len(configs)}] {c['tag']}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        json.dump({"done": len(done), "failed": failed, "total": len(configs),
                   "elapsed_h": (time.time() - t0) / 3600}, open(progress, "w"), indent=2)


if __name__ == "__main__":
    main()
