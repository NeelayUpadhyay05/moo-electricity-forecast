
# MOO-Electricity-Forecast

Research benchmark for hourly univariate electricity-load forecasting. The repository compares eight forecasting methods across multiple zones and seeds, with consistent evaluation budgets and dedicated multi-objective analyses.

Highlights
- Eight methods: two multi-objective (Musk Ox MO-LSTM, NSGA-II) and six single-objective / direct predictors.
- Standardized results saved per seed and zone under `results/seed_{n}/{zone}/`.
- Analysis notebook `notebooks/analysis_v2.ipynb` performs Friedman/Nemenyi, Wilcoxon tests, dominance checks, and knee-point selection.

---

## Methods

Multi-objective methods (optimize `val_mse` and `complexity`):

- `musk_ox_multi_lstm` — Musk Ox MOEA (our MO-LSTM implementation)
- `nsga2_direct` — NSGA-II baseline

Single-objective or direct predictive methods (optimize `val_mse` or are fixed models):

- `baseline_lstm` — fixed LSTM (no search)
- `optuna_lstm` — Optuna (single-objective TPE)
- `random_search_lstm` — random search over the LSTM space
- `arima` — ARIMA statistical model
- `lightgbm` — LightGBM on lag features
- `cnn_lstm` — CNN-LSTM hybrid

Only `musk_ox_multi_lstm` and `nsga2_direct` produce Pareto fronts and multi-objective outputs; the remaining methods are evaluated as single-objective models.

---

## Datasets & Zones

Processed data lives in `data/processed/`. Typical zone examples used in experiments:

- PJM: `PJME`, `AEP`, `DAYTON`
- NYISO: `NEW_YORK_CITY`, `LONG_ISLAND`, `CENTRAL`

Splits are chronological (no shuffle): train 70% / val 15% / test 15%.

---

## Metrics

Reported metrics (saved to `metrics.json` for each run):

- `rmse` — Root Mean Squared Error (MW)
- `mae` — Mean Absolute Error (MW)
- `mape` — Mean Absolute Percentage Error (%)
- `r2` — Coefficient of determination
- `hypervolume` — for multi-objective Pareto fronts

The analysis notebook primarily uses `mape` and average ranks for statistical comparisons and includes hypervolume and dominance checks for multi-objective methods.

---

## Layout (summary)

```
experiments/        # run_all.py + per-method runners
src/                # models, training, data, optimizers
data/               # raw/ and processed/ datasets
results/            # per-seed per-zone outputs and pareto_front.csv for MOO
checkpoints/        # saved checkpoints per seed/zone
notebooks/          # analysis notebooks
plots/              # exported figures
tables/             # CSV summaries
README.md
requirements.txt
```

Per-run outputs: `results/seed_{n}/{zone}/{method}/metrics.json`. Multi-objective runs include `pareto_front.csv`.

---

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Preprocess data (examples):

```bash
python -m src.data.run_preprocessing --zone PJME
python -m src.data.run_preprocessing --zone AEP
python -m src.data.run_preprocessing --zone DAYTON

# NYISO preprocessing
python -m src.data.run_preprocessing --dataset nyiso --data_dir data/raw/NYISO
```

3. Run the benchmark (single seed & zone):

```bash
python experiments/run_all.py --seed 42 --mode full --zone PJME
```

Or run a specific runner (per-method scripts):

```bash
python experiments/run_moo.py --seed 42 --mode full --zone PJME       # Musk Ox (MOO)
python experiments/run_nsga2.py --seed 42 --mode full --zone PJME     # NSGA-II (MOO)
python experiments/run_optuna.py --seed 42 --mode full --zone PJME    # Optuna (SO)
python experiments/run_random_search.py --seed 42 --mode full --zone PJME
python experiments/run_baseline.py --seed 42 --mode full --zone PJME
python experiments/run_arima.py --seed 42 --mode full --zone PJME
python experiments/run_lightgbm.py --seed 42 --mode full --zone PJME
python experiments/run_cnn_lstm.py --seed 42 --mode full --zone PJME
```

`--mode dev` uses reduced budgets for faster debugging; `--mode full` uses reporting budgets.

---

## Analysis

Open `notebooks/analysis_v2.ipynb` to reproduce the paper-quality analyses:

- Friedman test + Nemenyi post-hoc (global ranking)
- Wilcoxon (Mann–Whitney U) pairwise tests (MO-LSTM vs others) with FDR correction
- Pareto dominance checks (MO-LSTM vs NSGA-II)
- Knee-point selection for representative multi-objective solutions

The notebook exports tables to `tables/` and plots to `plots/`.

---

## Reproducibility

- Seeds are set for Python / NumPy / PyTorch via `src/utils/seed.py`.
- All runs save results to `results/seed_{n}/{zone}/` and checkpoints to `checkpoints/seed_{n}/{zone}/`.

---

## Contributing

To add a method: add a runner under `experiments/`, a training wrapper under `src/`, and update `src/config.py` to register the method. Please open an issue first for design discussion.

---

## License

MIT
