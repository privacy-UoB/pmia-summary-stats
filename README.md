# Revisiting Assumptions for Membership Inference on Summary Statistics

Code and datasets for the paper *"Revisiting Assumptions for Membership
Inference on Summary Statistics"* (P. Berrang, M. Ryan, K. Wooldridge, 2026).

> **Artifact reviewers:** the full PETs Artifact Appendix — requirements, setup
> (Docker and native), environment tests, and per-experiment reproduction steps
> — is in **[ARTIFACT-APPENDIX.md](ARTIFACT-APPENDIX.md)**.

## Background

MicroRNA expressions are real-valued numbers representing how active an RNA
molecule is in a cell at a given moment. Individuals contribute their miRNA
expression profiles to public datasets to advance medical research. Prior work
(Backes et al.) showed that the L1-Distances-Difference and Likelihood-Ratio
(LLR) tests can infer membership from *complete* miRNA expression summary
statistics. This project revisits those assumptions, studying inference from a
*subset* of features, under *noise*, and across *temporal drift*, and
decomposes how much measured accuracy comes from pool-size effects versus
shared disease signal.

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency and Python
management. From a clone of the repository:

```bash
uv sync          # installs the pinned Python (3.11–3.13) and all dependencies
```

All scripts are then run via `uv run python <script>`. Figures and their
underlying data are written to `results/` (created on demand).

## Reproducing the figures

The `run_*.sh` scripts are the authoritative way to regenerate each figure;
they call `uv run python` and run experiments in parallel where possible:

| Script                          | Figures                                  |
|---------------------------------|------------------------------------------|
| `run_theory_experiments.sh`     | Fig 1 (theoretical min-error bound)      |
| `run_noise_experiments.sh`      | Fig 2a–2d, Fig 9a–9b (noise)             |
| `run_ordered_experiments.sh`    | Fig 3a/3b, Fig 10–11 (missing features)  |
| `run_misc_experiments.sh`       | Fig 4 (Fitbit), Fig 12b (timestamp)      |
| `run_fig7_fig13_experiments.sh` | Fig 7a–7c, Fig 13 (stratified)           |
| `run_experiments.sh`            | appendix per-disease ordered runs        |

Pool-size and 19-disease validation, plus the noise-threshold table, are run
directly:

```bash
uv run python experiment_nsweep.py        # pool-size sweep
uv run python experiment_19disease.py     # 19-disease / case-vs-random validation
uv run python compute_thresholds.py       # noise-threshold table (Table 2)
```

Every wired script accepts `--seed N` (default 42) for reproducibility and
saves a `.npz` + `.meta.json` (git commit + argv) next to each PDF; pass
`--replot <pdf|npz>` (or `--plot-only` for the `fig_*`/`experiment_*` scripts)
to regenerate a figure from saved data without recomputing.

See **[ARTIFACT-APPENDIX.md](ARTIFACT-APPENDIX.md)** for the full
artifact-evaluation walkthrough (requirements, setup, tests, and the
per-experiment reproduction steps).

## Datasets

Required datasets ship under `Datasets/`:

- `GSE61741_series_matrix.csv` — cross-sectional miRNA dataset (GEO)
- `GSE68951_series_matrix.txt` — longitudinal miRNA dataset (GEO)
- `dailyActivity_merged.csv` — Fitbit dataset (Kaggle, CC0)
- `Diseased_miRNAs.txt` — up/down-regulated diseased miRNAs (ours)
- `independent_90.csv` — cross-sectional miRNAs with 90% independence (ours)

## License

Code is released under the MIT License (see `LICENSE`). Bundled datasets retain
their original licenses/terms, noted in `LICENSE`.
