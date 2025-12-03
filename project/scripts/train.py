"""
Main Training Script for Carbon-Aware Building Control.

This script implements the full SAC training pipeline with:
- Carbon-aware reward (Innovation B)
- Transformer feature extractor (Innovation C)
- TensorBoard logging
- Model checkpointing
- Evaluation callbacks

Run from project root:
    python -m scripts.train
    python -m scripts.train --config configs/experiment.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Local imports
from envs.sinergym_env import create_test_env
from models.transformer_policy import create_policy_kwargs


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_directories(config: dict, experiment_name: str) -> dict:
    """Create output directories for logs, models, and results."""
    logging_config = config.get('logging', {})
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"
    
    dirs = {
        'log_dir': project_root / logging_config.get('log_dir', 'outputs/logs') / run_name,
        'model_dir': project_root / logging_config.get('model_dir', 'outputs/models') / run_name,
        'results_dir': project_root / logging_config.get('results_dir', 'outputs/results') / run_name,
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}


def create_env_fn(config: dict, seed: int = None):
    """Factory function for creating environments."""
    def _init():
        env = create_test_env(
            config_path='configs/experiment.yaml',
            use_carbon_wrapper=True,
        )
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def train(
    config_path: str = 'configs/experiment.yaml',
    experiment_name: str = None,
    seed: int = None,
    use_transformer: bool = True,
    total_timesteps: int = None,
    verbose: int = 1,
):
    """
    Main training function.
    
    Args:
        config_path: Path to experiment configuration
        experiment_name: Name for this experiment run
        seed: Random seed for reproducibility
        use_transformer: Whether to use Transformer extractor (vs MLP)
        total_timesteps: Override total training timesteps
        verbose: Verbosity level (0=none, 1=info, 2=debug)
    """
    print("=" * 60)
    print("CARBON-AWARE BUILDING CONTROL - TRAINING")
    print("=" * 60)
    
    # Load configuration
    config_abs = project_root / config_path
    config = load_config(str(config_abs))
    
    # Override settings
    training_config = config.get('training', {})
    if seed is None:
        seed = training_config.get('seed', 42)
    if total_timesteps is None:
        total_timesteps = training_config.get('total_timesteps', 500000)
    if experiment_name is None:
        experiment_name = config.get('logging', {}).get('experiment_name', 'carbon_aware_sac')
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n[Config]")
    print(f"  Experiment: {experiment_name}")
    print(f"  Seed: {seed}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Use Transformer: {use_transformer}")
    
    # Create directories
    dirs = create_directories(config, experiment_name)
    print(f"\n[Directories]")
    for name, path in dirs.items():
        print(f"  {name}: {path}")
    
    # Create environments
    print("\n[Creating environments...]")
    train_env = DummyVecEnv([create_env_fn(config, seed)])
    eval_env = DummyVecEnv([create_env_fn(config, seed + 1000)])
    
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    
    # Create policy kwargs
    print("\n[Creating policy...]")
    transformer_config = config.get('transformer', {})
    carbon_config = config.get('carbon_wrapper', {})
    
    if use_transformer and transformer_config.get('enabled', True):
        policy_kwargs = create_policy_kwargs(
            extractor_type="transformer",
            features_dim=transformer_config.get('features_dim', 128),
            n_forecast_steps=carbon_config.get('forecast_horizon', 4),
            d_model=transformer_config.get('d_model', 64),
            n_heads=transformer_config.get('n_heads', 4),
            n_layers=transformer_config.get('n_layers', 2),
            dropout=transformer_config.get('dropout', 0.1),
        )
        print("  Using Transformer feature extractor")
    else:
        policy_kwargs = create_policy_kwargs(
            extractor_type="mlp",
            features_dim=128,
        )
        print("  Using MLP feature extractor (baseline)")
    
    # Create SAC agent
    print("\n[Creating SAC agent...]")
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
        learning_starts=training_config.get('learning_starts', 10000),
        tensorboard_log=dirs['log_dir'],
        verbose=verbose,
        seed=seed,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create callbacks
    print("\n[Setting up callbacks...]")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirs['model_dir'],
        log_path=dirs['results_dir'],
        eval_freq=training_config.get('eval_freq', 10000),
        n_eval_episodes=training_config.get('n_eval_episodes', 5),
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.get('save_freq', 50000),
        save_path=dirs['model_dir'],
        name_prefix="sac_checkpoint",
        verbose=verbose,
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    print("  EvalCallback: enabled")
    print("  CheckpointCallback: enabled")
    
    # Save configuration
    config_save_path = Path(dirs['model_dir']) / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  Config saved: {config_save_path}")
    
    # Start training
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"TensorBoard: tensorboard --logdir={dirs['log_dir']}")
    print()
    
    try:
        # Check if progress bar is available
        try:
            from rich.progress import Progress
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
        final_model_path = Path(dirs['model_dir']) / 'final_model'
        model.save(str(final_model_path))
        print(f"\n✅ Final model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        interrupt_model_path = Path(dirs['model_dir']) / 'interrupted_model'
        model.save(str(interrupt_model_path))
        print(f"  Model saved: {interrupt_model_path}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    
    return model, dirs


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train Carbon-Aware Building Control Agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: from config)",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Use MLP baseline instead of Transformer",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=none, 1=info, 2=debug)",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        experiment_name=args.name,
        seed=args.seed,
        use_transformer=not args.no_transformer,
        total_timesteps=args.timesteps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
