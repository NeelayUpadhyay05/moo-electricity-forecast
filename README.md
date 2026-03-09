# Multi-Objective Hyperparameter Optimization for LSTM-Based Electricity Load Forecasting

A systematic comparison of hyperparameter optimization (HPO) methods applied to an LSTM model for univariate electricity load forecasting. The core contribution is a **multi-objective optimization (MOO) approach using NSGA-II** that jointly minimizes validation error and model complexity, evaluated against four competing methods under a strictly equal function-evaluation budget.

---

## Contribution

Standard HPO treats model selection as a single-objective problem (minimize validation loss). This work frames it as a **Pareto optimization problem** — simultaneously minimizing:

1. **Validation MSE** — forecasting accuracy
2. **Model complexity** — number of trainable parameters (proxy for overfitting risk and inference cost)

The Pareto front produced by MOO surfaces configurations that no single-objective method can explore by design. The selected solution from the front is compared against the best configurations found by four baselines under identical evaluation budgets.

---

## Methods Compared

| Method | Description |
|--------|-------------|
| Baseline | Fixed default hyperparameters — no search |
| Random Search | Uniform random sampling over the search space |
| PSO | Particle Swarm Optimization — single-objective, minimizes validation MSE |
| Optuna | Bayesian optimization via TPE sampler (single-objective) |
| **MOO** | **NSGA-II — multi-objective, minimizes validation MSE and complexity jointly** |

All search methods operate under an **equal function-evaluation budget** (30 evaluations in full mode) to ensure fair comparison.

---

## Dataset

**PJM Hourly Energy Consumption** — real-world hourly electricity load (MW) from PJM Interconnection, a US regional transmission organization.

- Source: [Kaggle — Rob Mulla](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Raw files: `data/raw/PJM/` (not tracked by git)
- Format: two columns — `Datetime` and `{ZONE}_MW`
- Frequency: hourly (1H)
- Target: univariate load (MW), no exogenous features

| Zone | Rows | Span | MW Range |
|------|------|------|----------|
| PJME | 145,366 | 2002–2018 | 14,544–62,009 |
| AEP  | 121,273 | 2004–2018 | 9,581–25,695  |
| DAYTON | 121,275 | 2004–2018 | 982–3,746   |
| DUQ  | 119,068 | 2005–2018 | 1,014–3,054  |

Experiments are run independently on each zone; results are reported per-zone to demonstrate consistency across different load scales and regional profiles.

**Train / Val / Test split** (chronological, no shuffling):

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | LSTM training and HPO fitness evaluation |
| Val   | 15% | HPO objective (validation MSE) |
| Test  | 15% | Final held-out evaluation — never seen during HPO |

---

## Model

**Univariate LSTM** — a single-layer or multi-layer LSTM trained on a sliding window of past load values to forecast the next step.

- Input: sequence of `seq_len=24` hourly observations `(batch, 24, 1)`
- Output: single next-hour forecast `(batch, 1)`
- Normalization: z-score (mean/std computed on training split only)
- Training: cosine annealing LR schedule + early stopping on validation MSE

**Hyperparameter search space (4D):**

| Hyperparameter | Range | Type |
|---|---|---|
| `hidden_dim` | [32, 256] | Integer |
| `num_layers` | [1, 3] | Integer |
| `lr` | [1e-4, 5e-3] | Continuous (log scale) |
| `dropout` | [0.0, 0.3] | Continuous |

---

## Evaluation Metrics

All methods are evaluated on the held-out test set using the model retrained with the best hyperparameters found during search:

| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |

---

## Project Structure

```
MOO-Electricity-Forecast/
├── data/
│   ├── raw/
│   │   └── PJM/                        # raw CSVs (not tracked by git)
│   │       ├── PJME_hourly.csv
│   │       ├── AEP_hourly.csv
│   │       ├── DAYTON_hourly.csv
│   │       └── ...
│   └── processed/                      # preprocessed splits per zone
│       ├── {zone}_train.csv
│       ├── {zone}_val.csv
│       ├── {zone}_test.csv
│       └── {zone}_scaling.json
├── checkpoints/                        # saved model weights per seed/method
│   └── seed_{n}/
├── results/                            # JSON metrics and search histories
│   └── seed_{n}/
│       ├── baseline/metrics.json
│       ├── random_search/metrics.json
│       ├── pso/metrics.json
│       ├── optuna/metrics.json
│       └── moo/
│           ├── metrics.json
│           └── pareto_front.csv
├── experiments/
│   ├── run_baseline.py                 # fixed-config baseline
│   ├── run_random_search.py            # random search
│   ├── run_pso.py                      # PSO
│   ├── run_optuna.py                   # Optuna (TPE)
│   └── run_moo.py                      # MOO (NSGA-II)
├── src/
│   ├── config.py                       # all hyperparameters and mode settings
│   ├── models/
│   │   └── lstm.py                     # LSTM model definition
│   ├── data/
│   │   ├── dataset.py                  # PyTorch Dataset (sliding window)
│   │   ├── preprocess.py               # PJM preprocessing pipeline
│   │   └── run_preprocessing.py        # preprocessing entry point
│   ├── optimizers/
│   │   ├── pso.py                      # Particle Swarm Optimization
│   │   └── moo.py                      # NSGA-II multi-objective optimizer
│   ├── training/
│   │   ├── trainer.py                  # train_one_epoch / validate
│   │   ├── training_pipeline.py        # train_single_configuration / retrain_and_evaluate
│   │   ├── fitness.py                  # fitness functions for PSO and MOO
│   │   ├── early_stopping.py           # early stopping with checkpoint saving
│   │   └── experiment_runner.py        # orchestrates all methods
│   └── utils/
│       └── seed.py                     # reproducibility (Python, NumPy, PyTorch)
├── main.py                             # entry point
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

Run once per zone to generate the processed splits:

```bash
python -m src.data.run_preprocessing --zone PJME
```

This produces:
- `data/processed/PJME_train.csv`
- `data/processed/PJME_val.csv`
- `data/processed/PJME_test.csv`
- `data/processed/PJME_scaling.json`

---

## Running Experiments

Run all methods sequentially:

```bash
python main.py
```

Or run individual methods:

```bash
python experiments/run_baseline.py
python experiments/run_random_search.py
python experiments/run_pso.py
python experiments/run_optuna.py
python experiments/run_moo.py
```

### Dev vs Full mode

| Setting | Dev | Full |
|---------|-----|------|
| Batch size | 512 | 2048 |
| Search epochs (per eval) | 10 + early stop (patience 3) | 20 + early stop (patience 5) |
| Retrain epochs | 15 | 60 |
| Eval budget (all methods) | 12 | 30 |
| Random trials | 12 | 30 |
| PSO swarm / iterations | 4 / 2 | 6 / 4 |
| MOO population / generations | 4 / 2 | 6 / 4 |

Dev mode is for rapid iteration and debugging. Full mode is used for all reported results.

---

## Reproducibility

All experiments seed Python, NumPy, and PyTorch via `set_seed(seed)`. Default seed is 42. CuDNN deterministic mode is enabled. Results are saved per-seed under `results/seed_{n}/` and `checkpoints/seed_{n}/`.

---

## License

MIT
