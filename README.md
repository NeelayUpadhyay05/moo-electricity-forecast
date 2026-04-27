# Multi-Objective Hyperparameter Optimization for LSTM-Based Electricity Load Forecasting

A systematic comparison of hyperparameter optimization (HPO) methods applied to an LSTM model for univariate electricity load forecasting on the PJM dataset.

All search-based methods are evaluated under a strictly fair budget across multiple zones and seeds. In `full` mode, each search optimizer runs a 200-evaluation budget setting.

---

## Methods Compared

| Method | Description |
|--------|-------------|
| Baseline | Fixed default hyperparameters — no search |
| Random Search (MO) | Uniform random sampling with Pareto filtering on `(val_mse, complexity)` |
| Optuna (MO-TPE) | Multi-objective TPE that minimizes `(val_mse, complexity)` |
| PSO (MOPSO) | Multi-objective particle swarm with Pareto archive and leader sampling |
| **MOO (NSGA-II)** | **Multi-objective evolutionary search minimizing `(val_mse, complexity)`** |

All search-based methods share an **equal budget** in the main comparison. In `full` mode this is set as: `MOO = pop 10, gen 20`, `MOPSO = swarm 10, iter 20`, `Random = 200 trials`, and `Optuna = 200 trials`. Baseline remains a fixed non-search reference run.

---

## Dataset

**PJM Hourly Energy Consumption** — real-world hourly electricity load (MW) from PJM Interconnection, a US regional transmission organization.

- Source: [Kaggle — Rob Mulla](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Raw files: `data/raw/PJM/` (not tracked by git)
- Format: two columns — `Datetime` and `{ZONE}_MW`
- Frequency: hourly
- Target: univariate load (MW), no exogenous features

**Zones used in experiments:**

| Zone | Rows | Span | Mean Load (MW) |
|------|------|------|----------------|
| PJME | 145,366 | 2002–2018 | ~32,400 |
| AEP | 121,273 | 2004–2018 | ~15,800 |
| DAYTON | 121,275 | 2004–2018 | ~2,050 |

Three zones were selected to cover a wide range of load magnitudes (small / medium / large), demonstrating generalizability across different regional profiles.

**NYISO Hourly Actual Load** — real-world hourly electricity load (MW) from NYISO, a US regional transmission organization.

- Source: [EIA collected from NYISO](https://www.eia.gov/electricity/gridmonitor/)
- Raw files: `data/raw/NYISO/` (not tracked by git)
- Format: yearly wide CSV files with UTC/local timestamps and regional load columns
- Frequency: hourly
- Target: univariate load (MW), no exogenous features

**Regions used in experiments:**

| Region | Rows | Span | Mean Load (MW) |
|--------|------|------|----------------|
| NEW_YORK_CITY | 33,736 | 2021–2025 | ~5,608 |
| LONG_ISLAND | 33,736 | 2021–2025 | ~2,253 |
| CENTRAL | 33,736 | 2021–2025 | ~1,709 |

Three regions were selected to cover a wide range of load magnitudes (small / medium / large), demonstrating generalizability across different regional profiles.

The yearly NYISO files are concatenated first, then the top 3 regions are selected automatically by signal strength and completeness before being written to `data/processed/`.

**Train / Val / Test split** (chronological, no shuffling):

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | LSTM training and HPO fitness evaluation |
| Val   | 15% | HPO objective (validation MSE) |
| Test  | 15% | Final held-out evaluation — never seen during HPO |

---

## Model

**Univariate LSTM** trained on a sliding window of past load values to forecast the next hour.

- Input: sequence of `seq_len=24` hourly observations `(batch, 24, 1)`
- Output: single next-hour forecast `(batch, 1)`
- Normalization: z-score (mean/std from training split only)
- Training: Adam optimizer + early stopping on validation MSE

**Hyperparameter search space (4D):**

| Hyperparameter | Range | Scale |
|---|---|---|
| `hidden_dim` | [32, 256] | Integer |
| `num_layers` | [1, 3] | Integer |
| `lr` | [1e-4, 5e-3] | Log-continuous |
| `dropout` | [0.0, 0.3] | Continuous |

---

## Evaluation Metrics

All methods are evaluated on the held-out test set using the model retrained with the best hyperparameters found during search:

| Metric | Description |
|---|---|
| NRMSE | Normalized RMSE (RMSE / mean load) — primary comparison metric |
| RMSE | Root Mean Squared Error (MW) |
| MAE | Mean Absolute Error (MW) |
| MAPE | Mean Absolute Percentage Error (%) |

---

## Project Structure

```
MOO-Electricity-Forecast/
├── data/
│   ├── raw/
│   │   └── PJM/                          # raw CSVs (not tracked by git)
│   │       ├── PJME_hourly.csv
│   │       ├── AEP_hourly.csv
│   │       ├── DAYTON_hourly.csv
│   │       └── ...
│   └── processed/                        # preprocessed splits per zone
│       ├── {zone}_train.csv
│       ├── {zone}_val.csv
│       ├── {zone}_test.csv
│       └── {zone}_scaling.json
├── checkpoints/
│   ├── seed_{n}/{zone}/                  # checkpoints from main comparison
├── results/
│   ├── seed_{n}/{zone}/                  # main equal-budget comparison results
│   │   ├── baseline/metrics.json
│   │   ├── random_search/metrics.json
│   │   ├── optuna/metrics.json
│   │   ├── pso/metrics.json
│   │   ├── moo/
│   │   │   ├── metrics.json
│   │   │   └── pareto_front.csv
│   │   └── comparison.json
├── experiments/
│   ├── run_baseline.py                   # fixed-config baseline
│   ├── run_random_search.py              # random search
│   ├── run_optuna.py                     # Optuna (TPE)
│   ├── run_pso.py                        # PSO
│   └── run_moo.py                        # MOO (NSGA-II) under equal budget
├── src/
│   ├── config.py                         # all hyperparameters and mode settings
│   ├── models/
│   │   └── lstm.py                       # LSTM model definition
│   ├── data/
│   │   ├── dataset.py                    # PyTorch Dataset (sliding window)
│   │   ├── preprocess.py                 # PJM preprocessing pipeline
│   │   └── run_preprocessing.py          # preprocessing entry point
│   ├── optimizers/
│   │   ├── pso.py                        # PSO with boundary velocity reset
│   │   └── moo.py                        # NSGA-II (SBX crossover + polynomial mutation)
│   ├── training/
│   │   ├── trainer.py                    # train_one_epoch / validate
│   │   ├── training_pipeline.py          # train_single_configuration / retrain_and_evaluate
│   │   ├── fitness.py                    # fitness functions for PSO and MOO
│   │   └── early_stopping.py             # early stopping with checkpoint saving
│   └── utils/
│       └── seed.py                       # reproducibility (Python, NumPy, PyTorch)
├── main.py                               # runs all 5 methods for one seed/zone
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download PJM CSV files from Kaggle and place them in `data/raw/PJM/`.

---

## Preprocessing

Run once per zone before any experiments:

```bash
python -m src.data.run_preprocessing --zone PJME
python -m src.data.run_preprocessing --zone AEP
python -m src.data.run_preprocessing --zone DAYTON
```

For NYISO, use:

```bash
python -m src.data.run_preprocessing --dataset nyiso --data_dir data/raw/NYISO
```

This will save the selected region CSVs under `data/processed/nyiso_selected/` and the final train/val/test splits under `data/processed/`.

---

## Running Experiments

### Main comparison (all 5 methods, one seed + zone)

```bash
python main.py --seed 42 --mode full --zone PJME
```

### Individual methods

```bash
python experiments/run_baseline.py      --seed 42 --mode full --zone PJME
python experiments/run_random_search.py --seed 42 --mode full --zone PJME
python experiments/run_optuna.py        --seed 42 --mode full --zone PJME
python experiments/run_pso.py           --seed 42 --mode full --zone PJME
python experiments/run_moo.py           --seed 42 --mode full --zone PJME
```

### Multi-seed / multi-zone runs

Results are saved under `results/seed_{n}/{zone}/` — different seeds and zones never overwrite each other.

```bash
# Example: 5 seeds × 3 zones
for seed in 0 24 42 247 296; do
  for zone in PJME AEP DAYTON; do
    python main.py --seed $seed --zone $zone
  done
done
```

---

### Dev vs Full mode

| Setting | Dev | Full |
|---------|-----|------|
| Batch size | 512 | 2048 |
| Search epochs (per eval) | 10 + early stop (patience 3) | 20 + early stop (patience 5) |
| Retrain epochs | 15 | 60 |
| Eval budget (search methods) | 12 | 200 |
| Random trials | 12 | 200 |
| PSO swarm / iterations | 4 / 2 | 10 / 20 |
| MOO population / generations | 4 / 2 | 10 / 20 |
| Timesteps (dev truncation) | 2,000 | full dataset |

`dev` mode is for rapid debugging. All reported results use `full` mode.

---

## Reproducibility

All experiments seed Python, NumPy, and PyTorch via `set_seed(seed)`. CuDNN deterministic mode is enabled when CUDA is available. Results are saved per seed and zone under `results/seed_{n}/{zone}/` and `checkpoints/seed_{n}/{zone}/`.

---

## License

MIT
