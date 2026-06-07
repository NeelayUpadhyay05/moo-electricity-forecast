
# MOO-Electricity-Forecast

Research benchmark for hourly electricity-load forecasting across three datasets and six selected regions. The project compares eight forecasting methods under matched training budgets, with repeated runs over seeds and a statistical analysis notebook for method ranking and pairwise significance testing.

## Current benchmark

Selected regions:
- PJM: `PJME`, `AEP`
- NYISO: `NEW_YORK_CITY`, `LONG_ISLAND`
- India: `NATIONAL`, `NORTHERN`

Evaluation setup:
- 8 methods
- 6 regions
- 10 seeds

The processed data format is the same univariate train/val/test split used by the existing training code.

## Methods

Multi-objective methods:
- `musk_ox_multi_lstm` - Musk Ox MOEA for LSTM hyperparameter search
- `nsga2_direct` - NSGA-II baseline

Single-objective and direct methods:
- `baseline_lstm` - fixed LSTM baseline
- `optuna_lstm` - Optuna TPE search
- `random_search_lstm` - random hyperparameter search
- `arima` - ARIMA statistical model
- `lightgbm` - LightGBM on lag features
- `cnn_lstm` - CNN-LSTM hybrid

## Preprocessing

The preprocessing entry point is:

```bash
python src/data/run_preprocess_all_selected.py
```

This script regenerates the processed data under `data/processed/` for the selected six regions and also writes the NYISO selection snapshot under `data/processed/nyiso_selected/`.

Expected outputs per region:
- `{ZONE}_train.csv`
- `{ZONE}_val.csv`
- `{ZONE}_test.csv`
- `{ZONE}_scaling.json`

The preprocessing pipeline is chronological and leakage-safe:
- clean and deduplicate timestamps
- split 70% train / 15% validation / 15% test
- normalize with training statistics only
- save per-zone processed files for downstream experiments

## Statistical analysis

Open `notebooks/analysis.ipynb` for the main paper analysis.

The notebook is structured to run a rigorous statistical validation pipeline:
- **Global Non-Parametric Ranking:** Friedman test across 60 complete blocks (6 geographic zones × 10 independent seeds) to evaluate overall rank variance among the forecasting methods.
- **Post-Hoc Convergence Verification:** Nemenyi post-hoc test to compute the Critical Difference (CD) and map pairwise rank significance when the Friedman test successfully rejects the null hypothesis.
- **Multi-Dimensional Pairwise Superiority:** One-sided Mann-Whitney U (Wilcoxon rank-sum) tests evaluated across three distinct performance axes: error minimization (MAPE), structural economy (Trainable Parameters), and Pareto front optimization (Hypervolume).
- **Multi-Objective Trade-Off Selection:** Mathematical extraction of the optimal representative configuration from the Pareto front using Utopia Point Selection (minimizing Euclidean distance to the normalized ideal origin $(0,0)$).

## Running experiments

Run all methods for one region and one seed:

```bash
python experiments/run_all.py --seed 42 --mode full --zone PJME
```

`run_all.py` executes methods in this order: baseline, arima, lightgbm, cnn_lstm, random_search, optuna, nsga2, then musk_ox.

Run a single method directly:

```bash
python experiments/run_moo.py --seed 42 --mode full --zone PJME
python experiments/run_nsga2.py --seed 42 --mode full --zone PJME
python experiments/run_optuna.py --seed 42 --mode full --zone PJME
python experiments/run_random_search.py --seed 42 --mode full --zone PJME
python experiments/run_baseline.py --seed 42 --mode full --zone PJME
python experiments/run_arima.py --seed 42 --mode full --zone PJME
python experiments/run_lightgbm.py --seed 42 --mode full --zone PJME
python experiments/run_cnn_lstm.py --seed 42 --mode full --zone PJME
```

Use `--mode dev` for fast debugging and `--mode full` for the main benchmark budget.

## Output layout

Key folders:

```text
data/raw/                 # original source data
data/processed/           # regenerated train/val/test splits and scaling files
results/seed_{n}/{zone}/   # metrics and method outputs per seed and region
checkpoints/              # saved model checkpoints
tables/                   # CSV summaries from analysis
plots/                    # analysis figures
notebooks/                # analysis and runner notebooks
src/                      # data, models, training, optimizers, utilities
experiments/              # per-method runners and batch runner
```

Each method writes its metrics to `results/seed_{n}/{zone}/{method}/metrics.json`. Multi-objective runs also write `pareto_front.csv`.

## Reproducibility

- Random seeds are centralized in `src/utils/seed.py`.
- Training and evaluation use the same deterministic split convention across all methods.
- The repository keeps the processed format stable so the same model code can be reused after regenerating data.

## License

MIT
