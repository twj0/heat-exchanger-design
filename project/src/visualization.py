"""Visualization and data export utilities for TES heat exchanger studies.

Provides helpers to convert simulation outputs to structured CSV datasets and
generate publication-quality figures that align with the Energy Reports styling
guidelines referenced in the DRL-to-TES co-simulation paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Adopt a publication-ready aesthetic inspired by Energy Reports figures.
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ER_COLORS = {
    "Baseline": "#1f77b4",
    "DRL": "#d62728",
    "Demand": "#222222",
}

LINESTYLES = {
    "Baseline": "--",
    "DRL": "-",
}

def _cm_to_in(cm: float) -> float:
    return cm / 2.54

def _figsize_cm(width_cm: float = 12.0, nrows: int = 1, ncols: int = 1, height_scale: float = 0.9) -> tuple[float, float]:
    width_in = _cm_to_in(width_cm)
    height_in = width_in * (nrows / max(ncols, 1)) * height_scale
    return (width_in, height_in)

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")


def _ensure_directory(path: Path) -> None:
    """Create *path* (and parents) if it does not exist."""

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - defensive branch
        raise IOError(f"Unable to create directory: {path}") from exc


def _episode_to_frame(step_data: Sequence[Dict], controller: str) -> pd.DataFrame:
    """Convert raw step dictionaries into a tidy *DataFrame* with metadata."""

    if not step_data:
        raise ValueError(f"No step data provided for controller '{controller}'.")

    frame = pd.DataFrame(step_data).copy()
    frame["controller"] = controller

    # Ensure expected temporal ordering to preserve cumulative metrics.
    ordering_cols = [col for col in ("step", "hour") if col in frame.columns]
    if ordering_cols:
        frame = frame.sort_values(ordering_cols).reset_index(drop=True)

    return frame


def export_simulation_steps(
    baseline_steps: Sequence[Dict],
    rl_steps: Sequence[Dict],
    output_dir: Path | str = Path("data"),
    filename: str = "tes_simulation_steps.csv",
) -> Path:
    """Write per-timestep results for both controllers to a CSV file.

    The exported dataset captures hour-level thermodynamic states, control
    actions, and economic signals. It complies with the reproducibility
    requirements typically enforced in academic benchmarking studies.
    """

    output_path = Path(output_dir)
    _ensure_directory(output_path)

    baseline_df = _episode_to_frame(baseline_steps, controller="Baseline")
    rl_df = _episode_to_frame(rl_steps, controller="DRL")
    combined = pd.concat([baseline_df, rl_df], ignore_index=True)

    required_columns = {
        "step",
        "hour",
        "temperature",
        "soc",
        "power_command",
        "heat_demand",
        "heat_delivered",
        "cost",
        "electricity_price",
    }
    missing_columns = required_columns.difference(combined.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Missing required columns for export: "
            f"{missing_str}. Ensure episode data includes these keys."
        )

    csv_path = output_path / filename
    try:
        combined.to_csv(csv_path, index=False)
    except OSError as exc:
        raise IOError(f"Failed to write simulation data to {csv_path}") from exc

    return csv_path


def export_training_history(
    training_history: Sequence[Dict],
    output_dir: Path | str = Path("data"),
    csv_name: str = "tes_training_history.csv",
    json_name: str = "tes_training_history.json",
) -> Dict[str, Path]:
    """Persist episode-level training statistics in CSV and JSON formats."""

    if not training_history:
        raise ValueError("training_history is empty; nothing to export.")

    output_path = Path(output_dir)
    _ensure_directory(output_path)

    history_df = pd.DataFrame(training_history).copy()
    if "timesteps" not in history_df.columns:
        raise ValueError("training_history must include a 'timesteps' field per record.")

    history_df = history_df.sort_values("timesteps").reset_index(drop=True)
    if "episode" not in history_df.columns:
        history_df["episode"] = history_df.index + 1

    csv_path = output_path / csv_name
    json_path = output_path / json_name

    try:
        history_df.to_csv(csv_path, index=False)
        history_df.to_json(json_path, orient="records", indent=2)
    except OSError as exc:
        raise IOError(f"Failed to export training history to {output_path}") from exc

    return {"csv": csv_path, "json": json_path}


def export_hyperparameters(
    hyperparameters: Dict,
    output_dir: Path | str = Path("data"),
    filename: str = "tes_hyperparameters.json",
) -> Path:
    """Save training hyperparameters and configuration to JSON."""

    if not hyperparameters:
        raise ValueError("hyperparameters dictionary is empty.")

    output_path = Path(output_dir)
    _ensure_directory(output_path)

    json_path = output_path / filename
    try:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(hyperparameters, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise IOError(f"Failed to write hyperparameters to {json_path}") from exc

    return json_path


def export_performance_metrics(
    performance_metrics: Dict,
    output_dir: Path | str = Path("data"),
    filename: str = "tes_performance_metrics.json",
) -> Path:
    """Write aggregated evaluation metrics to JSON for archival."""

    if not performance_metrics:
        raise ValueError("performance_metrics is empty; nothing to export.")

    output_path = Path(output_dir)
    _ensure_directory(output_path)

    json_path = output_path / filename
    try:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(performance_metrics, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise IOError(f"Failed to write performance metrics to {json_path}") from exc

    return json_path


def export_metric_summary(
    comparison_metrics: Dict,
    output_dir: Path | str = Path("data"),
    filename: str = "tes_metric_summary.csv",
) -> Path:
    """Flatten controller KPIs into a CSV table suitable for appendix material."""

    output_path = Path(output_dir)
    _ensure_directory(output_path)

    if not comparison_metrics:
        raise ValueError("comparison_metrics is empty; cannot export summary.")

    if not {"baseline", "rl"}.issubset(comparison_metrics.keys()):
        raise KeyError("comparison_metrics must contain 'baseline' and 'rl' entries.")

    baseline = comparison_metrics["baseline"]
    rl = comparison_metrics["rl"]
    improvements = comparison_metrics.get("improvements", {})

    records: List[Dict[str, str]] = []

    def _append_metric(
        title: str,
        baseline_value: object,
        rl_value: object,
        improvement: Optional[object] = None,
    ) -> None:
        records.append(
            {
                "Metric": title,
                "Baseline": baseline_value,
                "DRL": rl_value,
                "Improvement": improvement if improvement is not None else "-",
            }
        )

    # Cost metrics
    cost_baseline = baseline.get("cost", {})
    cost_rl = rl.get("cost", {})
    _append_metric(
        "Total Cost (CNY)",
        f"{cost_baseline.get('total_cost', float('nan')):.2f}",
        f"{cost_rl.get('total_cost', float('nan')):.2f}",
        f"{improvements.get('cost_savings_percent', float('nan')):.2f}%",
    )
    _append_metric(
        "Average Cost per Hour (CNY)",
        f"{cost_baseline.get('average_cost_per_hour', float('nan')):.4f}",
        f"{cost_rl.get('average_cost_per_hour', float('nan')):.4f}",
    )

    # Energy delivery
    energy_baseline = baseline.get("energy", {})
    energy_rl = rl.get("energy", {})
    _append_metric(
        "TES Coverage",
        f"{energy_baseline.get('demand_satisfaction_rate', float('nan')):.2%}",
        f"{energy_rl.get('demand_satisfaction_rate', float('nan')):.2%}",
    )
    _append_metric(
        "Storage Efficiency",
        f"{energy_baseline.get('storage_efficiency', float('nan')):.2%}",
        f"{energy_rl.get('storage_efficiency', float('nan')):.2%}",
    )

    # Temperature compliance
    temp_baseline = baseline.get("temperature", {})
    temp_rl = rl.get("temperature", {})
    _append_metric(
        "Temperature Violation Rate",
        f"{temp_baseline.get('violation_rate', float('nan')):.2%}",
        f"{temp_rl.get('violation_rate', float('nan')):.2%}",
        f"{improvements.get('violation_improvement', float('nan')):.2%}",
    )

    # Storage utilisation
    storage_baseline = baseline.get("storage", {})
    storage_rl = rl.get("storage", {})
    _append_metric(
        "State of Charge Utilisation",
        f"{storage_baseline.get('soc_utilization', float('nan')):.2f}",
        f"{storage_rl.get('soc_utilization', float('nan')):.2f}",
    )

    summary_df = pd.DataFrame.from_records(records)

    csv_path = output_path / filename
    try:
        summary_df.to_csv(csv_path, index=False)
    except OSError as exc:
        raise IOError(f"Failed to write metric summary to {csv_path}") from exc

    return csv_path


def _prepare_plotting_frame(step_df: pd.DataFrame) -> pd.DataFrame:
    """Augment the step-level frame with helper columns for plotting."""

    if step_df["controller"].isna().any():
        raise ValueError("Controller labels are missing in the step data.")

    frame = step_df.copy()
    frame["cumulative_cost"] = frame.groupby("controller")["cost"].cumsum()
    frame["time_index"] = frame.groupby("controller").cumcount()
    return frame


def create_reward_timestep_curve(
    training_df: pd.DataFrame,
    figure_path: Path,
    reward_column: str = "episode_reward",
) -> None:
    """Plot episode reward against training timesteps."""

    required = {"timesteps", reward_column}
    missing = required.difference(training_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create reward curve because the following columns are "
            f"missing: {missing_str}."
        )

    fig, ax = plt.subplots(
        figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.75),
        constrained_layout=True,
    )
    ax.plot(training_df["timesteps"], training_df[reward_column])
    ax.set_title("Training Reward vs. Timesteps")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.grid(alpha=0.3)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(f"Unable to save reward curve to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_single_metric_comparison_figure(
    comparison_data: Dict[str, Dict[str, float]],
    metric: str,
    figure_path: Path,
) -> None:
    """Create a two-bar comparison figure for a single metric.

    metric one of: 'total_cost', 'storage_efficiency', 'tes_coverage', 'temp_violation_rate'
    """

    if not comparison_data:
        raise ValueError("comparison_data is empty; cannot create metric figure.")

    if "baseline" not in comparison_data or "rl" not in comparison_data:
        raise KeyError("comparison_data must contain 'baseline' and 'rl' entries.")

    b = comparison_data["baseline"]
    r = comparison_data["rl"]

    if metric == "total_cost":
        label = "Total Cost (CNY)"
        y_label = "Cost (CNY)"
        b_val = float(b["cost"]["total_cost"])  # type: ignore[index]
        r_val = float(r["cost"]["total_cost"])  # type: ignore[index]
        low_better = True
    elif metric == "storage_efficiency":
        label = "Storage Efficiency (%)"
        y_label = "Percent (%)"
        b_val = float(b["energy"]["storage_efficiency"]) * 100.0  # type: ignore[index]
        r_val = float(r["energy"]["storage_efficiency"]) * 100.0  # type: ignore[index]
        low_better = False
    elif metric == "tes_coverage":
        label = "TES Coverage (%)"
        y_label = "Percent (%)"
        b_val = float(b["energy"]["demand_satisfaction_rate"]) * 100.0  # type: ignore[index]
        r_val = float(r["energy"]["demand_satisfaction_rate"]) * 100.0  # type: ignore[index]
        low_better = False
    elif metric == "temp_violation_rate":
        label = "Temp Violation Rate (%)"
        y_label = "Percent (%)"
        b_val = float(b["temperature"]["violation_rate"]) * 100.0  # type: ignore[index]
        r_val = float(r["temperature"]["violation_rate"]) * 100.0  # type: ignore[index]
        low_better = True
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    x = np.arange(2)
    width = 0.5

    fig, ax = plt.subplots(
        figsize=_figsize_cm(10.0, nrows=1, ncols=1, height_scale=0.7),
        constrained_layout=True,
    )

    bars = ax.bar(x, [b_val, r_val], width=width, color=[ER_COLORS["Baseline"], ER_COLORS["DRL"]])
    ax.set_title(label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "DRL"])
    ax.grid(axis="y", alpha=0.3)

    for rect, h in zip(bars, [b_val, r_val]):
        ax.text(rect.get_x() + rect.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    if b_val != 0:
        if low_better:
            delta = (b_val - r_val) / b_val * 100.0
        else:
            delta = (r_val - b_val) / b_val * 100.0
    else:
        delta = 0.0

    y = max(b_val, r_val)
    ax.text(0.5, y * 1.04 if y != 0 else 0.04, f"Δ {delta:+.1f}%", ha="center", va="bottom", fontsize=7, color="#444444")

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(f"Unable to save metric figure to {figure_path}") from exc
    finally:
        plt.close(fig)


def generate_performance_metric_figures(
    comparison_data: Dict[str, Dict[str, float]],
    figure_dir: Path | str,
) -> Dict[str, Path]:
    """Generate four separate metric figures and return their paths."""

    out_dir = Path(figure_dir)
    _ensure_directory(out_dir)

    paths: Dict[str, Path] = {}

    p1 = out_dir / "tes_performance_cost.png"
    create_single_metric_comparison_figure(comparison_data, "total_cost", p1)
    paths["performance_cost"] = p1

    p2 = out_dir / "tes_performance_storage_efficiency.png"
    create_single_metric_comparison_figure(comparison_data, "storage_efficiency", p2)
    paths["performance_storage_efficiency"] = p2

    p3 = out_dir / "tes_performance_tes_coverage.png"
    create_single_metric_comparison_figure(comparison_data, "tes_coverage", p3)
    paths["performance_tes_coverage"] = p3

    p4 = out_dir / "tes_performance_temp_violation_rate.png"
    create_single_metric_comparison_figure(comparison_data, "temp_violation_rate", p4)
    paths["performance_temp_violation_rate"] = p4

    return paths

def create_learning_curve(
    training_df: pd.DataFrame,
    figure_path: Path,
    reward_column: str = "episode_reward",
    smoothing_window: int = 20,
) -> None:
    """Plot smoothed reward trajectory to highlight convergence trends."""

    required = {"timesteps", reward_column}
    missing = required.difference(training_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create learning curve because the following columns are "
            f"missing: {missing_str}."
        )

    smoothed = training_df[reward_column].rolling(
        window=smoothing_window,
        min_periods=1,
        center=False,
    ).mean()

    fig, ax = plt.subplots(
        figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.75),
        constrained_layout=True,
    )
    ax.plot(training_df["timesteps"], smoothed)
    ax.set_title("Learning Curve (Smoothed Reward)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rolling Mean Reward")
    ax.grid(alpha=0.3)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(f"Unable to save learning curve to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_episode_length_curve(
    training_df: pd.DataFrame,
    figure_path: Path,
    length_column: str = "episode_length",
) -> None:
    """Visualise episode duration trends throughout training."""

    required = {"timesteps", length_column}
    missing = required.difference(training_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create episode length curve because the following columns "
            f"are missing: {missing_str}."
        )

    fig, ax = plt.subplots(
        figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.75),
        constrained_layout=True,
    )
    ax.plot(training_df["timesteps"], training_df[length_column])
    ax.set_title("Episode Length Across Training")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length (steps)")
    ax.grid(alpha=0.3)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(
            f"Unable to save episode length curve to {figure_path}"
        ) from exc
    finally:
        plt.close(fig)


def create_success_rate_curve(
    training_df: pd.DataFrame,
    figure_path: Path,
    success_column: str = "success_rate",
) -> None:
    """Plot success rate progression if the metric is available."""

    required = {"timesteps", success_column}
    missing = required.difference(training_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create success rate curve because the following columns "
            f"are missing: {missing_str}."
        )

    fig, ax = plt.subplots(
        figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.75),
        constrained_layout=True,
    )
    ax.plot(training_df["timesteps"], training_df[success_column])
    ax.set_title("Training Success Rate")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(f"Unable to save success rate curve to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_training_iteration_curve(
    training_df: pd.DataFrame,
    figure_path: Path,
    iteration_column: str = "episode",
) -> None:
    """Show iteration progression relative to timesteps."""

    required = {"timesteps", iteration_column}
    missing = required.difference(training_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create training iteration curve because the following "
            f"columns are missing: {missing_str}."
        )

    fig, ax = plt.subplots(
        figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.75),
        constrained_layout=True,
    )
    ax.plot(training_df["timesteps"], training_df[iteration_column])
    ax.set_title("Training Episodes vs. Timesteps")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Index")
    ax.grid(alpha=0.3)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(
            f"Unable to save training iteration curve to {figure_path}"
        ) from exc
    finally:
        plt.close(fig)


def create_training_overview_figure(
    training_df: pd.DataFrame,
    figure_path: Path,
    reward_column: str = "episode_reward",
    success_column: str = "success_rate",
    smoothing_window: int = 20,
    convergence_window: int = 30,
    convergence_rel_slope: float = 0.005,
) -> None:
    """Create a compact training overview with smoothed reward and success.

    Adds a convergence region where the normalised slope of the smoothed reward
    stays below a relative threshold up to the end of training.
    """

    required = {"timesteps", reward_column}
    missing = required.difference(training_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create training overview because the following columns are "
            f"missing: {missing_str}."
        )

    x = training_df["timesteps"].to_numpy()
    y = training_df[reward_column].to_numpy()
    y_smooth = pd.Series(y).rolling(window=smoothing_window, min_periods=1).mean().to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=_figsize_cm(12.0, nrows=2, ncols=1, height_scale=0.95),
        constrained_layout=True,
        sharex=True,
    )

    ax1.plot(x, y, color=ER_COLORS.get("Baseline", "#1f77b4"), alpha=0.35, label="Reward (raw)")
    ax1.plot(x, y_smooth, color=ER_COLORS.get("DRL", "#d62728"), label=f"Reward (smooth {smoothing_window})")
    ax1.set_title("Training Reward (Raw & Smoothed)")
    ax1.set_ylabel("Episode Reward")
    ax1.grid(alpha=0.3)

    # Convergence region detection (at the tail)
    if len(x) >= max(convergence_window, 5):
        slope = np.abs(np.gradient(y_smooth, x))
        # Rolling mean slope
        k = max(3, convergence_window)
        kernel = np.ones(k) / k
        mean_slope = np.convolve(slope, kernel, mode="same")
        y_range = max(np.ptp(y_smooth), 1e-9)
        cond = (mean_slope / y_range) < convergence_rel_slope
        # Find last contiguous True segment ending at tail
        tail_idx = len(cond) - 1
        start_idx = None
        if cond[tail_idx]:
            j = tail_idx
            while j >= 0 and cond[j]:
                j -= 1
            start_idx = j + 1
        if start_idx is not None and start_idx < len(x) - 1:
            ax1.axvspan(x[start_idx], x[-1], color="#2ca02c", alpha=0.12)
            ax1.text(x[start_idx], ax1.get_ylim()[1], "Convergence", va="top", ha="left", fontsize=7, color="#2ca02c")

    ax1.legend(loc="best", frameon=False)

    if success_column in training_df.columns:
        ax2.plot(x, training_df[success_column].to_numpy(), color="#9467bd", label="Success Rate")
        ax2.set_ylabel("Success Rate")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="best", frameon=False)
    else:
        ax2.plot(x, y_smooth, color=ER_COLORS.get("DRL", "#d62728"))
        ax2.set_ylabel("Smoothed Reward")
    ax2.set_xlabel("Timesteps")
    ax2.grid(alpha=0.3)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(f"Unable to save training overview figure to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_performance_comparison_figure(
    comparison_data: Dict[str, Dict[str, float]],
    figure_path: Path,
) -> None:
    """Create a grouped bar chart comparing baseline and RL performance."""

    if not comparison_data:
        raise ValueError("comparison_data is empty; cannot create comparison figure.")

    metrics: List[tuple] = []
    if "baseline" in comparison_data and "rl" in comparison_data:
        b = comparison_data["baseline"]
        r = comparison_data["rl"]
        imp = comparison_data.get("improvements", {})
        metrics = [
            ("Total Cost (CNY)", b["cost"]["total_cost"], r["cost"]["total_cost"], True, imp.get("cost_savings_percent")),
            ("Storage Efficiency (%)", b["energy"]["storage_efficiency"] * 100, r["energy"]["storage_efficiency"] * 100, False, None),
            ("TES Coverage (%)", b["energy"]["demand_satisfaction_rate"] * 100, r["energy"]["demand_satisfaction_rate"] * 100, False, None),
            ("Temp Violation Rate (%)", b["temperature"]["violation_rate"] * 100, r["temperature"]["violation_rate"] * 100, True, None),
        ]
    else:
        melted_records: List[Dict[str, object]] = []
        for metric_name, values in comparison_data.items():
            if not isinstance(values, dict):
                raise ValueError(
                    "comparison_data entries must map to dictionaries of controller values"
                )
            baseline_v = values.get("Baseline")
            rl_v = values.get("DRL")
            if baseline_v is None or rl_v is None:
                raise ValueError("Each metric must include 'Baseline' and 'DRL' values.")
            is_lower_better = "cost" in metric_name.lower() or "violation" in metric_name.lower()
            metrics.append((metric_name, baseline_v, rl_v, is_lower_better, None))

    labels = [m[0] for m in metrics]
    baseline_vals = np.array([m[1] for m in metrics], dtype=float)
    rl_vals = np.array([m[2] for m in metrics], dtype=float)
    lower_is_better = [bool(m[3]) for m in metrics]
    improvements = [m[4] for m in metrics]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.75), constrained_layout=True)
    bars_b = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color=ER_COLORS["Baseline"])
    bars_r = ax.bar(x + width / 2, rl_vals, width, label="DRL", color=ER_COLORS["DRL"])

    ax.set_title("Performance Metrics Comparison")
    ax.set_ylabel("Value")
    ax.set_xlabel("Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars_b, bars_r):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    for i, (b, r, low_better, imp_pct) in enumerate(zip(baseline_vals, rl_vals, lower_is_better, improvements)):
        if imp_pct is None:
            if low_better:
                delta = (b - r) / b * 100 if b != 0 else 0.0
            else:
                delta = (r - b) / b * 100 if b != 0 else 0.0
        else:
            delta = imp_pct
        y = max(b, r)
        ax.text(x[i] + width / 2, y * 1.02 if y != 0 else 0.02, f"Δ {delta:+.1f}%", ha="center", va="bottom", fontsize=7, color="#444444")

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(
            f"Unable to save performance comparison figure to {figure_path}"
        ) from exc
    finally:
        plt.close(fig)


def create_time_series_figure(step_df: pd.DataFrame, figure_path: Path) -> None:
    """Generate a four-panel time-series overview highlighting TES behaviour."""

    required = {"time_index", "temperature", "soc", "power_command", "cumulative_cost"}
    missing = required.difference(step_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create time-series figure because the following columns "
            f"are missing: {missing_str}."
        )

    fig, axes = plt.subplots(2, 2, figsize=_figsize_cm(12.0, nrows=2, ncols=2, height_scale=1.0), constrained_layout=True, sharex=True)

    # Stable controller order
    controllers = [c for c in ["Baseline", "DRL"] if c in step_df["controller"].unique()]
    for controller, group in step_df.groupby("controller"):
        color = ER_COLORS.get(controller, None)
        ls = LINESTYLES.get(controller, "-")
        axes[0, 0].plot(group["time_index"], group["temperature"], label=controller, color=color, linestyle=ls)
        axes[0, 1].plot(group["time_index"], group["soc"], label=controller, color=color, linestyle=ls)
        axes[1, 0].plot(group["time_index"], group["power_command"], label=controller, color=color, linestyle=ls)
        axes[1, 1].plot(group["time_index"], group["cumulative_cost"], label=controller, color=color, linestyle=ls)

    axes[0, 0].set_title("Storage Temperature Trajectory")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].axhspan(40, 50, color="lightgrey", alpha=0.3)

    axes[0, 1].set_title("State of Charge Evolution")
    axes[0, 1].set_ylabel("State of Charge (-)")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].set_title("Charging / Discharging Power")
    axes[1, 0].set_ylabel("Power (kW)")
    axes[1, 0].set_xlabel("Timestep (index)")

    axes[1, 1].set_title("Cumulative Operational Cost")
    axes[1, 1].set_ylabel("Cost (CNY)")
    axes[1, 1].set_xlabel("Timestep (index)")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(set(labels)), frameon=False)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(f"Unable to save time-series figure to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_heat_balance_figure(step_df: pd.DataFrame, figure_path: Path) -> None:
    """Plot heat demand coverage to showcase comfort compliance."""

    required = {"time_index", "heat_demand", "heat_delivered", "controller"}
    missing = required.difference(step_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create heat balance figure because the following columns "
            f"are missing: {missing_str}."
        )

    fig, ax = plt.subplots(figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.8), constrained_layout=True)
    ax2 = ax.twinx()

    controllers_present = step_df["controller"].unique().tolist()
    df_b = step_df.loc[step_df["controller"] == "Baseline"].sort_values("time_index")
    df_r = step_df.loc[step_df["controller"] == "DRL"].sort_values("time_index")

    if len(df_b) > 0:
        t = df_b["time_index"].to_numpy()
    elif len(df_r) > 0:
        t = df_r["time_index"].to_numpy()
    else:
        raise ValueError("No data rows available to plot heat balance figure.")

    demand_b = df_b["heat_demand"].to_numpy() if len(df_b) > 0 else None
    demand_r = df_r["heat_demand"].to_numpy() if len(df_r) > 0 else None

    if demand_b is not None and demand_r is not None and len(demand_b) == len(demand_r):
        if not np.allclose(demand_b, demand_r, rtol=1e-4, atol=1e-6):
            ref_demand = 0.5 * (demand_b + demand_r)
        else:
            ref_demand = demand_b
    else:
        ref_demand = demand_b if demand_b is not None else demand_r

    ax.plot(t, ref_demand, color=ER_COLORS["Demand"], label="Heat Demand", linestyle="-", linewidth=1.8)

    delivered_baseline = df_b["heat_delivered"].to_numpy() if len(df_b) > 0 else None
    delivered_rl = df_r["heat_delivered"].to_numpy() if len(df_r) > 0 else None

    if delivered_baseline is not None:
        ax.plot(t, delivered_baseline, color=ER_COLORS.get("Baseline"), linestyle=LINESTYLES.get("Baseline", "-"), label="Baseline Delivered")
        ax.fill_between(t, np.minimum(ref_demand, delivered_baseline), np.maximum(ref_demand, delivered_baseline), color=ER_COLORS.get("Baseline"), alpha=0.12, label="Gap (Baseline)")

    if delivered_rl is not None:
        ax.plot(t, delivered_rl, color=ER_COLORS.get("DRL"), linestyle=LINESTYLES.get("DRL", "-"), label="DRL Delivered")
        ax.fill_between(t, np.minimum(ref_demand, delivered_rl), np.maximum(ref_demand, delivered_rl), color=ER_COLORS.get("DRL"), alpha=0.10, label="Gap (DRL)")

    eps = 1e-9
    if delivered_baseline is not None and demand_b is not None:
        sat_baseline = np.clip(delivered_baseline / (demand_b + eps) * 100.0, 0, 150)
        ax2.plot(t, sat_baseline, color=ER_COLORS.get("Baseline"), linestyle=":", label="Baseline Satisfaction (%)")
    if delivered_rl is not None and demand_r is not None:
        sat_rl = np.clip(delivered_rl / (demand_r + eps) * 100.0, 0, 150)
        ax2.plot(t, sat_rl, color=ER_COLORS.get("DRL"), linestyle=":", label="DRL Satisfaction (%)")
    ax2.axhline(100, color="#555555", linestyle="--", linewidth=1.2)
    ax2.set_ylim(0, 150)

    ax.set_title("Heat Demand vs. Delivered (with Satisfaction)")
    ax.set_xlabel("Timestep (index)")
    ax.set_ylabel("Heat Power (kW)")
    ax2.set_ylabel("Satisfaction (%)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, ncol=2)
    ax.grid(alpha=0.3)

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(f"Unable to save heat balance figure to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_temperature_compliance_boxplot(step_df: pd.DataFrame, figure_path: Path) -> None:
    """Summarise temperature violations to underline control robustness."""

    required = {"temperature", "controller"}
    missing = required.difference(step_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create temperature compliance figure because the following "
            f"columns are missing: {missing_str}."
        )

    fig, ax = plt.subplots(figsize=_figsize_cm(12.0, nrows=1, ncols=1, height_scale=0.7), constrained_layout=True)

    cat_order = [c for c in ["Baseline", "DRL"] if c in step_df["controller"].unique()]
    pal = {k: ER_COLORS[k] for k in ER_COLORS if k in (cat_order or step_df["controller"].unique())}
    sns.boxplot(
        data=step_df,
        x="controller",
        y="temperature",
        hue="controller",
        dodge=False,
        order=cat_order if cat_order else None,
        ax=ax,
        palette=pal,
        fliersize=2,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    sns.stripplot(
        data=step_df,
        x="controller",
        y="temperature",
        hue="controller",
        dodge=False,
        order=cat_order if cat_order else None,
        ax=ax,
        palette=pal,
        alpha=0.35,
        size=2,
        jitter=0.1,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.axhline(40, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Temperature Distribution and Compliance")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Controller")
    ax.grid(alpha=0.3)

    bounds_low, bounds_high = 40.0, 50.0
    rates = (
        step_df.assign(violation=(step_df["temperature"].lt(bounds_low) | step_df["temperature"].gt(bounds_high)))
        .groupby("controller")["violation"].mean()
    )
    ymax = ax.get_ylim()[1]
    cats = cat_order if cat_order else list(step_df["controller"].unique())
    for i, ctrl in enumerate(cats):
        rate = float(rates.get(ctrl, 0.0)) * 100.0
        ax.text(i, ymax * 0.98, f"{rate:.1f}%", ha="center", va="top", fontsize=7, color="#444444")

    try:
        axins = inset_axes(ax, width="35%", height="45%", loc="upper left", borderpad=1)
        sns.boxplot(
            data=step_df,
            x="controller",
            y="temperature",
            hue="controller",
            dodge=False,
            order=cat_order if cat_order else None,
            ax=axins,
            palette=pal,
            fliersize=2,
        )
        leg2 = axins.get_legend()
        if leg2 is not None:
            leg2.remove()
        sns.stripplot(
            data=step_df,
            x="controller",
            y="temperature",
            hue="controller",
            dodge=False,
            order=cat_order if cat_order else None,
            ax=axins,
            palette=pal,
            alpha=0.35,
            size=2,
            jitter=0.1,
        )
        leg2 = axins.get_legend()
        if leg2 is not None:
            leg2.remove()
        axins.set_ylim(48, 50.5)
        axins.axhline(40, color="red", linestyle="--", linewidth=1.0)
        axins.axhline(50, color="red", linestyle="--", linewidth=1.0)
        axins.set_xlabel("")
        axins.set_ylabel("")
        axins.grid(alpha=0.3)
    except Exception:
        pass

    try:
        _savefig(fig, figure_path)
    except OSError as exc:
        raise IOError(
            f"Unable to save temperature compliance figure to {figure_path}"
        ) from exc
    finally:
        plt.close(fig)


def generate_publication_figures(
    step_df: pd.DataFrame,
    figure_dir: Path | str = Path("project/figure"),
) -> Dict[str, Path]:
    """Create the set of required figures and return their file paths."""

    figure_path = Path(figure_dir)
    _ensure_directory(figure_path)

    plotting_df = _prepare_plotting_frame(step_df)

    outputs = {
        "time_series": figure_path / "tes_time_series.png",
        "heat_balance": figure_path / "tes_heat_balance.png",
        "temperature_box": figure_path / "tes_temperature_compliance.png",
    }

    create_time_series_figure(plotting_df, outputs["time_series"])
    create_heat_balance_figure(plotting_df, outputs["heat_balance"])
    create_temperature_compliance_boxplot(plotting_df, outputs["temperature_box"])

    return outputs


def generate_training_figures(
    training_df: pd.DataFrame,
    figure_dir: Path | str = Path("project/figure"),
    smoothing_window: int = 20,
) -> Dict[str, Path]:
    """Produce publication-ready figures summarising the training process."""

    figure_path = Path(figure_dir)
    _ensure_directory(figure_path)

    outputs: Dict[str, Path] = {}

    reward_curve_path = figure_path / "tes_training_reward.png"
    create_reward_timestep_curve(training_df, reward_curve_path)
    outputs["reward_curve"] = reward_curve_path

    learning_curve_path = figure_path / "tes_learning_curve.png"
    create_learning_curve(training_df, learning_curve_path, smoothing_window=smoothing_window)
    outputs["learning_curve"] = learning_curve_path

    if "episode_length" in training_df.columns:
        episode_length_path = figure_path / "tes_episode_length.png"
        create_episode_length_curve(training_df, episode_length_path)
        outputs["episode_length_curve"] = episode_length_path

    if "success_rate" in training_df.columns:
        success_rate_path = figure_path / "tes_success_rate.png"
        create_success_rate_curve(training_df, success_rate_path)
        outputs["success_rate_curve"] = success_rate_path

    iteration_curve_path = figure_path / "tes_training_iterations.png"
    create_training_iteration_curve(training_df, iteration_curve_path)
    outputs["iteration_curve"] = iteration_curve_path

    overview_path = figure_path / "tes_training_overview.png"
    create_training_overview_figure(
        training_df,
        overview_path,
        reward_column="episode_reward",
        success_column="success_rate" if "success_rate" in training_df.columns else "success_rate",
        smoothing_window=smoothing_window,
    )
    outputs["training_overview"] = overview_path

    return outputs


def load_steps_from_csvs(baseline_csv: Path | str, rl_csv: Path | str) -> pd.DataFrame:

    base_df = pd.read_csv(baseline_csv)
    base_df["controller"] = "Baseline"
    rl_df = pd.read_csv(rl_csv)
    rl_df["controller"] = "DRL"

    combined = pd.concat([base_df, rl_df], ignore_index=True)
    if "step" in combined.columns:
        combined = combined.sort_values(["controller", "step"]).reset_index(drop=True)
    return _prepare_plotting_frame(combined)


def generate_figures_from_csvs(
    baseline_csv: Path | str,
    rl_csv: Path | str,
    figure_dir: Path | str,
) -> Dict[str, Path]:
    df = load_steps_from_csvs(baseline_csv, rl_csv)
    return generate_publication_figures(df, figure_dir=figure_dir)


def generate_outputs(
    baseline_steps: Sequence[Dict],
    rl_steps: Sequence[Dict],
    comparison_metrics: Optional[Dict] = None,
    data_output_dir: Path | str = Path("data"),
    figure_output_dir: Path | str = Path("project/figure"),
) -> Dict[str, Path]:
    """Convenience orchestrator for CSV exports and figure generation."""

    exported = {}

    step_csv = export_simulation_steps(
        baseline_steps=baseline_steps,
        rl_steps=rl_steps,
        output_dir=data_output_dir,
    )
    exported["step_data_csv"] = step_csv

    if comparison_metrics is not None:
        summary_csv = export_metric_summary(
            comparison_metrics=comparison_metrics,
            output_dir=data_output_dir,
        )
        exported["metric_summary_csv"] = summary_csv
        # Also generate a grouped performance comparison figure
        comparison_figure_path = Path(figure_output_dir) / "tes_performance_comparison.png"
        create_performance_comparison_figure(comparison_metrics, comparison_figure_path)
        exported["performance_comparison_figure"] = comparison_figure_path

    # Rebuild dataframe for plotting using the just-exported CSV for reproducibility.
    step_df = pd.read_csv(step_csv)
    figures = generate_publication_figures(step_df, figure_dir=figure_output_dir)
    exported.update(figures)

    return exported


def generate_training_outputs(
    training_history: Sequence[Dict],
    hyperparameters: Optional[Dict] = None,
    performance_metrics: Optional[Dict] = None,
    comparison_data: Optional[Dict[str, Dict[str, float]]] = None,
    data_output_dir: Path | str = Path("data"),
    figure_output_dir: Path | str = Path("project/figure"),
    smoothing_window: int = 20,
) -> Dict[str, Path]:
    """Pipeline for exporting training data and creating associated figures."""

    outputs: Dict[str, Path] = {}

    history_paths = export_training_history(
        training_history=training_history,
        output_dir=data_output_dir,
    )
    outputs.update({f"training_history_{key}": path for key, path in history_paths.items()})

    if hyperparameters is not None:
        hyper_path = export_hyperparameters(
            hyperparameters=hyperparameters,
            output_dir=data_output_dir,
        )
        outputs["hyperparameters_json"] = hyper_path

    if performance_metrics is not None:
        performance_path = export_performance_metrics(
            performance_metrics=performance_metrics,
            output_dir=data_output_dir,
        )
        outputs["performance_metrics_json"] = performance_path

    training_df = pd.read_csv(outputs["training_history_csv"])
    figure_paths = generate_training_figures(
        training_df=training_df,
        figure_dir=figure_output_dir,
        smoothing_window=smoothing_window,
    )
    outputs.update(figure_paths)

    if comparison_data is not None:
        comparison_figure_path = Path(figure_output_dir) / "tes_performance_comparison.png"
        create_performance_comparison_figure(comparison_data, comparison_figure_path)
        outputs["performance_comparison_figure"] = comparison_figure_path

    return outputs

