"""Generate representative sample data for visualization validation.

This utility rolls out the baseline rule-based controller and a simple
heuristic RL-like controller inside ``TESHeatExEnv`` for a configurable
number of hours. The resulting per-timestep records, along with synthetic
training history, are exported as CSV files that power the visualization
smoke tests documented in ``docs/visualization_guide.md``.

Example
-------
.. code-block:: bash

   python simulate/generate_sample_data.py --duration 168 --output-dir data/sample

The command above produces ``baseline_steps.csv``, ``rl_steps.csv`` and
``training_history.csv`` inside ``data/sample``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from baselines.rule_based import SimpleTOUController
from env.tes_heatex_env import TESHeatExEnv


def _load_config(config_path: Path) -> Dict:
    import yaml

    candidates = []
    if config_path.is_absolute():
        candidates.append(config_path)
    else:
        candidates.append(Path.cwd() / config_path)
        candidates.append(Path(__file__).resolve().parent.parent / config_path)
        candidates.append(Path(__file__).resolve().parents[1] / config_path)

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Unable to locate configuration file. Checked: {tried}")


def _initialise_environment(config: Dict, duration: int, seed: int) -> TESHeatExEnv:
    cfg = config.copy()
    cfg.setdefault("simulation", {})
    cfg["simulation"]["duration"] = duration
    cfg["simulation"]["seed"] = seed
    return TESHeatExEnv(cfg)


def _rollout_controller(
    env: TESHeatExEnv,
    controller,
    seed: int,
) -> List[Dict]:
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    if hasattr(controller, "reset"):
        controller.reset()

    while not (terminated or truncated):
        if hasattr(controller, "select_action"):
            action = controller.select_action(obs, info)
        else:
            action = controller(obs, info)
        obs, _, terminated, truncated, info = env.step(action)

    return env.get_episode_data()


class HeuristicRLController:
    """Lightweight heuristic that mimics RL behaviour for sample data.

    The policy mixes SimpleTOUController's decision with mild stochastic
    exploration so the produced trajectories differ from the baseline data.
    """

    def __init__(self, env: TESHeatExEnv, exploration_prob: float = 0.15) -> None:
        self.env = env
        self.exploration_prob = exploration_prob
        self.baseline = SimpleTOUController()

    def reset(self) -> None:
        self.baseline.reset()

    def select_action(self, obs, info):
        if random.random() < self.exploration_prob:
            if self.env.action_type == "discrete":
                return self.env.action_space.sample()
            return np.array(
                [random.uniform(-self.env.heater_max_power, self.env.heater_max_power)],
                dtype=np.float32,
            )
        return self.baseline.select_action(obs, info)


def _export_episode_data(data: Iterable[Dict], output_path: Path) -> Path:
    frame = pd.DataFrame(list(data))
    frame.to_csv(output_path, index=False)
    return output_path


def _generate_training_history(num_points: int, max_timesteps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timesteps = np.linspace(max_timesteps / num_points, max_timesteps, num_points, dtype=int)
    base_reward = np.linspace(-200, 250, num_points) + rng.normal(0, 20, num_points)
    success_rate = np.clip(
        np.linspace(0.2, 0.92, num_points) + rng.normal(0, 0.03, num_points),
        0,
        1,
    )
    episode_length = rng.integers(80, 160, num_points)

    history = pd.DataFrame(
        {
            "timesteps": timesteps,
            "episode_reward": base_reward,
            "success_rate": success_rate,
            "episode_length": episode_length,
        }
    )
    history["episode"] = np.arange(1, num_points + 1)
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample TES-HeatEx datasets for visualization.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample"),
        help="Directory to store generated CSV files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the base configuration file.",
    )
    parser.add_argument("--duration", type=int, default=168, help="Simulation horizon in hours for each rollout.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--history-points", type=int, default=150, help="Number of entries to synthesise for training history.")
    parser.add_argument("--max-timesteps", type=int, default=50000, help="Maximum timesteps reflected in the training history")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)

    env_baseline = _initialise_environment(config, args.duration, args.seed)
    baseline_controller = SimpleTOUController()
    baseline_steps = _rollout_controller(env_baseline, baseline_controller, args.seed)

    env_rl = _initialise_environment(config, args.duration, args.seed + 1)
    rl_controller = HeuristicRLController(env_rl)
    rl_steps = _rollout_controller(env_rl, rl_controller, args.seed + 1)

    baseline_path = _export_episode_data(baseline_steps, args.output_dir / "baseline_steps.csv")
    rl_path = _export_episode_data(rl_steps, args.output_dir / "rl_steps.csv")

    history_df = _generate_training_history(args.history_points, args.max_timesteps, args.seed)
    history_path = args.output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    print("Sample data written:")
    print(f"  Baseline steps : {baseline_path}")
    print(f"  RL steps       : {rl_path}")
    print(f"  Training hist. : {history_path}")


if __name__ == "__main__":
    main()
