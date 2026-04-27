import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
import pandas as pd
from src.config import Config


RUNNER_MAP = {
    "baseline_lstm": "run_baseline.py",
    "musk_ox_multi_lstm": "run_moo.py",
    "random_search_lstm": "run_random_search.py",
    "optuna_lstm": "run_optuna.py",
    "nsga2_direct": "run_nsga2.py",
    "arima": "run_arima.py",
    "lightgbm": "run_lightgbm.py",
    "cnn_lstm": "run_cnn_lstm.py",
}


def run_experiment(model_name, seed, mode, zone):
    """Run an experiment for a single model and return metrics."""
    if model_name not in RUNNER_MAP:
        print(f"⚠ Unknown model: {model_name}")
        return None

    runner_script = RUNNER_MAP[model_name]
    runner_path = os.path.join(os.path.dirname(__file__), runner_script)
    
    cmd = [
        sys.executable, runner_path,
        "--seed", str(seed),
        "--mode", mode,
        "--zone", zone,
    ]

    print(f"\n{'='*70}")
    print(f"Running: {model_name} (seed={seed}, mode={mode}, zone={zone})")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True, timeout=3600)
        # Infer output dir from runner convention
        if model_name == "baseline_lstm":
            out_dir = f"results/seed_{seed}/{zone}/baseline"
        elif model_name == "musk_ox_multi_lstm":
            out_dir = f"results/seed_{seed}/{zone}/musk_ox"
        elif model_name == "random_search_lstm":
            out_dir = f"results/seed_{seed}/{zone}/random_search"
        elif model_name == "optuna_lstm":
            out_dir = f"results/seed_{seed}/{zone}/optuna"
        elif model_name == "nsga2_direct":
            out_dir = f"results/seed_{seed}/{zone}/nsga2"
        else:
            out_dir = f"results/seed_{seed}/{zone}/{model_name}"

        metrics_file = os.path.join(out_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                return json.load(f)
        else:
            print(f"⚠ Metrics file not found: {metrics_file}")
            return None
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout running {model_name}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {model_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run all models in the benchmark suite and aggregate results."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="full", choices=["dev", "full"], help="Execution mode")
    parser.add_argument("--zone", type=str, default="PJME", help="Energy zone/region")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to run (default: all)")
    args = parser.parse_args()

    config = Config(mode=args.mode)
    models_to_run = args.models.split(",") if args.models else config.model_list

    print(f"\n{'#'*70}")
    print(f"# Benchmark Suite: seed={args.seed}, mode={args.mode}, zone={args.zone}")
    print(f"# Models: {models_to_run}")
    print(f"{'#'*70}")

    results = {}
    for model_name in models_to_run:
        metrics = run_experiment(model_name, args.seed, args.mode, args.zone)
        if metrics:
            results[model_name] = metrics

    # Aggregate and display results
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    summary_rows = []
    for model_name in models_to_run:
        if model_name in results:
            m = results[model_name]
            row = {
                "Model": model_name,
                "Test NRMSE": m.get("best_test_nrmse", m.get("nrmse", "N/A")),
                "Test RMSE (MW)": m.get("best_test_rmse", m.get("rmse", "N/A")),
                "Test MAE (MW)": m.get("best_test_mae", m.get("mae", "N/A")),
                "Test MAPE (%)": m.get("best_test_mape", m.get("mape", "N/A")),
                "Objectives": m.get("objectives", ["single"]),
            }
            summary_rows.append(row)
        else:
            summary_rows.append({"Model": model_name, "Status": "FAILED"})

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Save aggregated results
    summary_path = f"results/seed_{args.seed}/{args.zone}/summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
