from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.visualization import (  # noqa: E402
    generate_training_figures,
    generate_figures_from_csvs,
    create_performance_comparison_figure,
    generate_performance_metric_figures,
)
from metrics.calculator import compare_controllers  # noqa: E402


def _infer_hourly_prices(df: pd.DataFrame) -> np.ndarray:
    if "hour" not in df.columns or "electricity_price" not in df.columns:
        raise ValueError("CSV must contain 'hour' and 'electricity_price' columns")
    hourly = df.groupby("hour")["electricity_price"].mean().reindex(range(24)).ffill().bfill()
    vals = hourly.to_numpy(dtype=float)
    if np.isnan(vals).all():
        vals = np.zeros(24, dtype=float)
    return vals


def _csv_to_episode_records(csv_path: Path) -> List[Dict]:
    df = pd.read_csv(csv_path)
    required = {"hour", "power_command", "heat_demand", "heat_delivered", "temperature", "soc", "cost"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {', '.join(sorted(missing))}")
    return df.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Energy Reports-style figures from CSVs")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing baseline_steps.csv, rl_steps.csv, training_history.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to place generated figures")
    parser.add_argument("--smooth", type=int, default=20, help="Smoothing window for training curves")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = args.data_dir / "baseline_steps.csv"
    rl_csv = args.data_dir / "rl_steps.csv"
    train_csv = args.data_dir / "training_history.csv"

    if not baseline_csv.is_file() or not rl_csv.is_file():
        raise FileNotFoundError("baseline_steps.csv and rl_steps.csv must exist in data-dir")

    pub_figs = generate_figures_from_csvs(baseline_csv, rl_csv, figure_dir=args.out_dir)

    if train_csv.is_file():
        training_df = pd.read_csv(train_csv)
        training_figs = generate_training_figures(training_df, figure_dir=args.out_dir, smoothing_window=args.smooth)
    else:
        training_figs = {}

    base_df = pd.read_csv(baseline_csv)
    rl_df = pd.read_csv(rl_csv)

    price_vec = _infer_hourly_prices(pd.concat([base_df, rl_df], ignore_index=True))

    baseline_records = _csv_to_episode_records(baseline_csv)
    rl_records = _csv_to_episode_records(rl_csv)

    comparison = compare_controllers(baseline_records, rl_records, electricity_prices=price_vec)
    perf_path = args.out_dir / "tes_performance_comparison.png"
    create_performance_comparison_figure(comparison, perf_path)
    per_metric_paths = generate_performance_metric_figures(comparison, args.out_dir)

    print("Figures written:")
    for name, p in {**pub_figs, **training_figs, **per_metric_paths, "performance_comparison": perf_path}.items():
        print(f"  {name:28s}: {p}")


if __name__ == "__main__":
    main()
