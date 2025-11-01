"""Visualization and data export utilities for TES heat exchanger studies.

Provides helpers to convert simulation outputs to structured CSV datasets and
generate publication-quality figures that align with the Energy Reports styling
guidelines referenced in the DRL-to-TES co-simulation paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Adopt a publication-ready aesthetic inspired by Energy Reports figures.
sns.set_theme(style="whitegrid", context="talk", palette="Paired")


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
        "Demand Satisfaction Rate",
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

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(training_df["timesteps"], training_df[reward_column], linewidth=2)
    ax.set_title("Training Reward vs. Timesteps")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
    except OSError as exc:
        raise IOError(f"Unable to save reward curve to {figure_path}") from exc
    finally:
        plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(training_df["timesteps"], smoothed, linewidth=2)
    ax.set_title("Learning Curve (Smoothed Reward)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rolling Mean Reward")
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
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

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(training_df["timesteps"], training_df[length_column], linewidth=2)
    ax.set_title("Episode Length Across Training")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length (steps)")
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
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

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(training_df["timesteps"], training_df[success_column], linewidth=2)
    ax.set_title("Training Success Rate")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
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

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(training_df["timesteps"], training_df[iteration_column], linewidth=2)
    ax.set_title("Training Episodes vs. Timesteps")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Index")
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
    except OSError as exc:
        raise IOError(
            f"Unable to save training iteration curve to {figure_path}"
        ) from exc
    finally:
        plt.close(fig)


def create_performance_comparison_figure(
    comparison_data: Dict[str, Dict[str, float]],
    figure_path: Path,
) -> None:
    """Create a grouped bar chart comparing baseline and RL performance."""

    if not comparison_data:
        raise ValueError("comparison_data is empty; cannot create comparison figure.")

    melted_records: List[Dict[str, object]] = []
    for metric_name, values in comparison_data.items():
        if not isinstance(values, dict):
            raise ValueError(
                "comparison_data entries must map to dictionaries of controller values"
            )
        for controller, value in values.items():
            melted_records.append(
                {
                    "Metric": metric_name,
                    "Controller": controller,
                    "Value": value,
                }
            )

    df = pd.DataFrame(melted_records)
    required_columns = {"Metric", "Controller", "Value"}
    if df.empty or required_columns.difference(df.columns):
        raise ValueError("comparison_data did not yield a valid comparison table.")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.barplot(data=df, x="Metric", y="Value", hue="Controller", ax=ax)
    ax.set_title("Baseline vs. DRL Performance Metrics")
    ax.set_ylabel("Value")
    ax.set_xlabel("Metric")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
    except OSError as exc:
        raise IOError(
            f"Unable to save performance comparison figure to {figure_path}"
        ) from exc
    finally:
        plt.close(fig)


def create_time_series_figure(step_df: pd.DataFrame, figure_path: Path) -> None:
    """Generate a four-panel time-series overview highlighting TES behaviour."""

    # Figure 1: Temporal evolution of temperature, state of charge, power, and cost.
    required = {"time_index", "temperature", "soc", "power_command", "cumulative_cost"}
    missing = required.difference(step_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create time-series figure because the following columns "
            f"are missing: {missing_str}."
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    for controller, group in step_df.groupby("controller"):
        axes[0, 0].plot(group["time_index"], group["temperature"], label=controller, linewidth=2)
        axes[0, 1].plot(group["time_index"], group["soc"], label=controller, linewidth=2)
        axes[1, 0].plot(group["time_index"], group["power_command"], label=controller, linewidth=2)
        axes[1, 1].plot(group["time_index"], group["cumulative_cost"], label=controller, linewidth=2)

    axes[0, 0].set_title("Storage Temperature Trajectory")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].axhspan(40, 50, color="lightgrey", alpha=0.3)
    axes[0, 0].legend(loc="upper right")

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

    fig.suptitle("TES Behaviour under Baseline and DRL Control", fontsize=20)

    try:
        fig.savefig(figure_path, dpi=300)
    except OSError as exc:
        raise IOError(f"Unable to save time-series figure to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_heat_balance_figure(step_df: pd.DataFrame, figure_path: Path) -> None:
    """Plot heat demand coverage to showcase comfort compliance."""

    # Figure 2: Side-by-side heat demand vs delivery for each controller.
    required = {"heat_demand", "heat_delivered", "controller"}
    missing = required.difference(step_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create heat balance figure because the following columns "
            f"are missing: {missing_str}."
        )

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    melted = step_df.melt(
        id_vars=["time_index", "controller"],
        value_vars=["heat_demand", "heat_delivered"],
        var_name="Series",
        value_name="Heat (kW)",
    )

    sns.lineplot(
        data=melted,
        x="time_index",
        y="Heat (kW)",
        hue="Series",
        style="controller",
        linewidth=2,
        ax=ax,
    )

    ax.set_title("Heat Demand Tracking Performance")
    ax.set_xlabel("Timestep (index)")
    ax.set_ylabel("Heat (kW)")
    ax.legend(title="Series / Controller", loc="upper right")
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
    except OSError as exc:
        raise IOError(f"Unable to save heat balance figure to {figure_path}") from exc
    finally:
        plt.close(fig)


def create_temperature_compliance_boxplot(step_df: pd.DataFrame, figure_path: Path) -> None:
    """Summarise temperature violations to underline control robustness."""

    # Figure 3: Boxplot of temperature excursions relative to bounds.
    required = {"temperature", "controller"}
    missing = required.difference(step_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Cannot create temperature compliance figure because the following "
            f"columns are missing: {missing_str}."
        )

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    sns.boxplot(data=step_df, x="controller", y="temperature", ax=ax)
    ax.axhline(40, color="red", linestyle="--", linewidth=1.5, label="Lower Bound")
    ax.axhline(50, color="red", linestyle="-.", linewidth=1.5, label="Upper Bound")
    ax.set_title("Distribution of Storage Temperature")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Controller")
    ax.legend()
    ax.grid(alpha=0.3)

    try:
        fig.savefig(figure_path, dpi=300)
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

    return outputs


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

