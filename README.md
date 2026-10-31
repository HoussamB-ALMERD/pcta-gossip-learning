# PCTA: Two-Channel Sequential Trust for Byzantine-Resilient Gossip Learning

Code and results for the article *"Byzantine-Resilient Decentralized Gossip Learning via Two-Channel Sequential Trust and Bounded-Leverage Mixing"* by H. Bouardi, S. Hraoui and H. El Fadili (LIASSE Laboratory, ENSA Fez, Sidi Mohamed Ben Abdellah University). The article is under review.

PCTA (Predictive-Corrective Trust Architecture) is a defense that every node of a decentralized gossip-learning network runs on its own. For each neighbor it combines two kinds of evidence:

- **Trajectory channel:** how much the neighbor's update deviates from a per-neighbor predictor of that neighbor's own updates (diagonal Kalman filter, innovation statistic).
- **Data-integrity channel:** how the update changes class-conditional predictions on a small public reference set (level, log-odds contrast and per-class AUC contrast scores).

Each score is standardized against the observer's own robust (median/MAD) history and fed into its own CUSUM statistic. The CUSUM state is kept per neighbor pair across rounds. The resulting trust weights are used in an **absolute, norm-bounded mixing rule**, so distrusted mass stays on the node's own model.

## Layout

```
src/
  harness.py              simulation: gossip network, local training, attacks, all defenses
                          (PCTA = method "pcta_soft", decision rule "v2")
  pcta_engine.py          Kalman / EMA predictors, Beta reputation
  defenses.py             baselines: FedAvg, Krum, trimmed mean, median, FoolsGold, FLTrust,
                          OpenCLAW-style reputation, ClippedGossip, BALANCE, IOS
  attacks.py              Gaussian, sign-flip, Sybil, slow poison, adaptive (ALIE, IPM, dissensus
                          and label flips are in harness.py)
  data.py, common.py, topology.py
  run_all_v2.py           experiment matrix (tiers A main, B sensitivity, C ablation,
                          D whitewashing + detectability floor)
  analyze_v2.py           per-tier tables (mean, 95% CI, AUROC, Byzantine mass, FPR/TPR)
  sig_tests_v2.py         paired significance tests
  make_paper_tables_v2.py / make_paper_figures_v2.py   every table and figure of the article
results/
  runs/                   one JSON per run (final metrics, accuracy log, configuration)
  tables/                 aggregated tables (tierA/C/D .csv/.md, significance.md)
manifests/                SHA-256 manifests of the frozen design (v2) and of v2.1 (baseline fixes)
```

## Reproducing

Python 3.11, PyTorch 2.1 (CUDA recommended). Install the dependencies with `pip install -r requirements.txt`, then:

```bash
cd src
export PCTA_DATA_ROOT=/path/to/data_cache     # CIFAR-10 / Fashion-MNIST are downloaded here
python run_all_v2.py --tiers A,C,D --dry-run  # run counts and time estimate
python -u run_all_v2.py --tiers A             # resumable; writes results_v2/
python analyze_v2.py results_v2
python sig_tests_v2.py
python make_paper_tables_v2.py
python make_paper_figures_v2.py
```

A run with N = 100 nodes and T = 40 rounds takes roughly 5-6 minutes on a single consumer GPU. Tier A has 726 runs.

## Notes on protocol

- The design was tuned on a development configuration (CIFAR-10, alpha = 0.1, seed 0) and checked on seed 2, then frozen (`manifests/MANIFEST_v2_frozen.sha256`). All reported results use seeds 10-12, which were never used during development. `run_all_v2.py` asserts this.
- v2.1 fixed two baseline implementations after the freeze: FLTrust's root-update clipping and shared reference set, and BALANCE's gamma calibrated on the development configuration. The PCTA code path is unchanged (`manifests/MANIFEST_v2_1.sha256`).
- The only change in this public copy is `DATA_ROOT` in `data.py`, which now reads the `PCTA_DATA_ROOT` environment variable.
- Per-round, per-pair trust logs (`*_detect.json`, about 12 GB) are not included. They are available from the authors on request.
- Comments in the code refer to internal design notes (e.g. `PCTA_Theory.md`, finding numbers F1-F19) that are not part of this repository.
- `harness.py` also contains the earlier v1 decision rule and Canary Probe (`variant={"decision": "v1"}`, `use_probe`). They are not used in the article.

## License

MIT (see `LICENSE`).
