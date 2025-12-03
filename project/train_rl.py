#!/usr/bin/env python
"""
Complete RL Training Script for Carbon-Aware Building Control.

This is the main entry point for training SAC agents with:
- Transformer or MLP feature extractors
- Carbon-aware dual-objective reward
- Multiple baseline comparisons
- TensorBoard logging and checkpointing

Usage:
    # Train with Transformer (default)
    python train_rl.py
    
    # Train with MLP baseline
    python train_rl.py --extractor mlp
    
    # Quick test run
    python train_rl.py --timesteps 10000 --eval-freq 2000
    
    # Custom configuration
    python train_rl.py --config configs/experiment.yaml --name my_experiment

Author: Auto-generated for Applied Energy publication
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import yaml

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports
from envs.eplus_env import EnergyPlusGymEnv, EnvConfig, create_env as create_eplus_env
from models.transformer_policy import create_policy_kwargs


class TensorBoardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_costs = []
        self._episode_carbons = []
    
    def _on_step(self) -> bool:
        # Log episode metrics when episode ends
        if self.locals.get("dones", [False])[0]:
            infos = self.locals.get("infos", [{}])
            if infos:
                info = infos[0]
                self._episode_costs.append(info.get("episode_cost", 0))
                self._episode_carbons.append(info.get("episode_carbon", 0))
                
                # Log to TensorBoard every 10 episodes
                if len(self._episode_costs) >= 10:
                    self.logger.record("custom/mean_episode_cost", np.mean(self._episode_costs))
                    self.logger.record("custom/mean_episode_carbon", np.mean(self._episode_carbons))
                    self._episode_costs = []
                    self._episode_carbons = []
        
        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_env_fn(config: Dict, seed: int = None, use_mock: bool = True):
    """Factory function for creating environments."""
    def _init():
        env_config = EnvConfig(
            episode_length=config.get('environment', {}).get('episode_length', 672),
            timestep_seconds=config.get('environment', {}).get('timestep', 900),
            include_forecasts=True,
            forecast_horizon=config.get('carbon_wrapper', {}).get('forecast_horizon', 4),
            cost_weight=1.0 - config.get('carbon_wrapper', {}).get('lambda_carbon', 0.5),
            carbon_weight=config.get('carbon_wrapper', {}).get('lambda_carbon', 0.5),
            comfort_penalty=config.get('carbon_wrapper', {}).get('lambda_comfort', 10.0),
        )
        env = EnergyPlusGymEnv(config=env_config, use_mock=use_mock)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def train(
    config_path: str = "configs/experiment.yaml",
    experiment_name: Optional[str] = None,
    seed: int = 42,
    total_timesteps: int = 500000,
    extractor_type: str = "transformer",
    eval_freq: int = 10000,
    save_freq: int = 50000,
    use_mock: bool = True,
    verbose: int = 1,
) -> SAC:
    """
    Main training function.
    
    Args:
        config_path: Path to experiment configuration
        experiment_name: Name for this experiment
        seed: Random seed
        total_timesteps: Total training timesteps
        extractor_type: "transformer", "temporal_transformer", or "mlp"
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        use_mock: Use mock environment (True) or real EnergyPlus (False)
        verbose: Verbosity level
        
    Returns:
        Trained SAC model
    """
    print("=" * 70)
    print("CARBON-AWARE BUILDING CONTROL - RL TRAINING")
    print("=" * 70)
    
    # Load configuration
    config_abs = project_root / config_path
    if config_abs.exists():
        config = load_config(str(config_abs))
    else:
        print(f"Config not found: {config_abs}, using defaults")
        config = {}
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Experiment setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"sac_{extractor_type}"
    run_name = f"{experiment_name}_{timestamp}"
    
    # Create directories
    log_dir = project_root / "outputs" / "logs" / run_name
    model_dir = project_root / "outputs" / "models" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Configuration]")
    print(f"  Experiment: {run_name}")
    print(f"  Extractor: {extractor_type}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Seed: {seed}")
    print(f"  Log dir: {log_dir}")
    print(f"  Model dir: {model_dir}")
    
    # Create environments
    print(f"\n[Creating environments...]")
    train_env = DummyVecEnv([create_env_fn(config, seed, use_mock)])
    eval_env = DummyVecEnv([create_env_fn(config, seed + 1000, use_mock)])
    
    # Normalize observations
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Action bounds: [{train_env.action_space.low}, {train_env.action_space.high}]")
    
    # Create policy kwargs
    print(f"\n[Creating {extractor_type} policy...]")
    transformer_config = config.get('transformer', {})
    carbon_config = config.get('carbon_wrapper', {})
    
    policy_kwargs = create_policy_kwargs(
        extractor_type=extractor_type,
        features_dim=transformer_config.get('features_dim', 128),
        n_forecast_steps=carbon_config.get('forecast_horizon', 4),
        d_model=transformer_config.get('d_model', 64),
        n_heads=transformer_config.get('n_heads', 4),
        n_layers=transformer_config.get('n_layers', 2),
        dropout=transformer_config.get('dropout', 0.1),
    )
    
    # Create SAC model
    print(f"\n[Creating SAC agent...]")
    training_config = config.get('training', {})
    
    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=training_config.get('learning_rate', 3e-4),
        buffer_size=training_config.get('buffer_size', 100000),
        batch_size=training_config.get('batch_size', 256),
        gamma=training_config.get('gamma', 0.99),
        tau=training_config.get('tau', 0.005),
        ent_coef=training_config.get('ent_coef', 'auto'),
        learning_starts=training_config.get('learning_starts', 1000),
        tensorboard_log=str(log_dir),
        verbose=verbose,
        seed=seed,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    print(f"\n[Setting up callbacks...]")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(model_dir),
        name_prefix="sac_checkpoint",
        verbose=verbose,
    )
    
    tb_callback = TensorBoardCallback()
    
    callbacks = CallbackList([eval_callback, checkpoint_callback, tb_callback])
    print(f"  EvalCallback: every {eval_freq} steps")
    print(f"  CheckpointCallback: every {save_freq} steps")
    print(f"  TensorBoardCallback: enabled")
    
    # Save configuration
    config_save = {
        "experiment_name": run_name,
        "extractor_type": extractor_type,
        "total_timesteps": total_timesteps,
        "seed": seed,
        **config,
    }
    with open(model_dir / "config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config_save, f, default_flow_style=False, allow_unicode=True)
    
    # Training
    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)
    print(f"\nMonitor with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    print()
    
    try:
        # Check if progress bar is available
        try:
            import rich
            progress_bar = True
        except ImportError:
            progress_bar = False
            print("  (Progress bar disabled - install rich for progress bar)")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
        )
        
        # Save final model
        final_path = model_dir / "final_model"
        model.save(str(final_path))
        train_env.save(str(model_dir / "vec_normalize.pkl"))
        print(f"\n✅ Final model saved: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        interrupt_path = model_dir / "interrupted_model"
        model.save(str(interrupt_path))
        print(f"  Model saved: {interrupt_path}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    
    return model


def evaluate_baselines(
    config_path: str = "configs/experiment.yaml",
    n_episodes: int = 10,
    use_mock: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate baseline controllers for comparison.
    
    Args:
        config_path: Path to configuration
        n_episodes: Number of evaluation episodes
        use_mock: Use mock environment
        
    Returns:
        Dictionary of baseline results
    """
    from baselines.rule_based import create_baseline
    
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    
    # Load config
    config_abs = project_root / config_path
    if config_abs.exists():
        config = load_config(str(config_abs))
    else:
        config = {}
    
    # Create environment
    env_config = EnvConfig(
        episode_length=config.get('environment', {}).get('episode_length', 672),
        include_forecasts=True,
    )
    env = EnergyPlusGymEnv(config=env_config, use_mock=use_mock)
    
    baseline_names = ['rule_based', 'always_on', 'random', 'carbon_aware_rule']
    results = {}
    
    for baseline_name in baseline_names:
        print(f"\n[Evaluating {baseline_name}...]")
        controller = create_baseline(baseline_name, env.action_space)
        
        episode_costs = []
        episode_carbons = []
        episode_rewards = []
        
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            total_reward = 0
            done = False
            
            while not done:
                action, _ = controller.predict(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            metrics = env.get_episode_metrics()
            episode_costs.append(metrics['total_cost'])
            episode_carbons.append(metrics['total_carbon'])
            episode_rewards.append(total_reward)
        
        results[baseline_name] = {
            'mean_cost': np.mean(episode_costs),
            'std_cost': np.std(episode_costs),
            'mean_carbon': np.mean(episode_carbons),
            'std_carbon': np.std(episode_carbons),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
        }
        
        print(f"  Cost: ¥{results[baseline_name]['mean_cost']:.2f} ± {results[baseline_name]['std_cost']:.2f}")
        print(f"  Carbon: {results[baseline_name]['mean_carbon']:.2f} ± {results[baseline_name]['std_carbon']:.2f} kg")
        print(f"  Reward: {results[baseline_name]['mean_reward']:.2f} ± {results[baseline_name]['std_reward']:.2f}")
    
    env.close()
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Carbon-Aware Building Control Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_rl.py                           # Default training
  python train_rl.py --extractor mlp           # MLP baseline
  python train_rl.py --timesteps 100000        # Custom timesteps
  python train_rl.py --eval-baselines          # Evaluate baselines only
        """
    )
    
    parser.add_argument(
        "--config", type=str, default="configs/experiment.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--extractor", type=str, default="transformer",
        choices=["transformer", "temporal_transformer", "mlp"],
        help="Feature extractor type"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10000,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--save-freq", type=int, default=50000,
        help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        choices=[0, 1, 2],
        help="Verbosity level"
    )
    parser.add_argument(
        "--eval-baselines", action="store_true",
        help="Evaluate baselines only (no training)"
    )
    parser.add_argument(
        "--real-eplus", action="store_true",
        help="Use real EnergyPlus (default: mock environment)"
    )
    
    args = parser.parse_args()
    
    if args.eval_baselines:
        evaluate_baselines(
            config_path=args.config,
            use_mock=not args.real_eplus,
        )
    else:
        train(
            config_path=args.config,
            experiment_name=args.name,
            seed=args.seed,
            total_timesteps=args.timesteps,
            extractor_type=args.extractor,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            use_mock=not args.real_eplus,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
