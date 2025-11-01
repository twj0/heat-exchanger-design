"""
Training script for reinforcement learning agents.

Supports PPO, SAC, and DQN algorithms from Stable-Baselines3.
"""

import os
import yaml
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.append('..')
from env.tes_heatex_env import TESHeatExEnv


def _tensorboard_available() -> bool:
    """Return True if 'tensorboard' package is importable."""
    try:
        import tensorboard  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False

def resolve_config_path(config_arg: Optional[str]) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = []
    if config_arg:
        if os.path.isabs(config_arg):
            candidates.append(config_arg)
        else:
            # relative to current working directory
            candidates.append(os.path.abspath(config_arg))
            # relative to script location
            candidates.append(os.path.abspath(os.path.join(script_dir, config_arg)))
    # fallback: default config in project root
    candidates.append(os.path.join(project_root, "configs", "default.yaml"))
    # fallback: default config relative to CWD
    candidates.append(os.path.abspath(os.path.join("configs", "default.yaml")))
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not resolve configuration file. Tried: {candidates}"
    )


def create_env(config: Dict, seed: Optional[int] = None):
    """
    Create and wrap environment.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        Wrapped environment
    """
    if seed is not None:
        config["simulation"]["seed"] = seed
    
    env = TESHeatExEnv(config)
    env = Monitor(env)
    return env


def train_rl_agent(
    config: Dict,
    algorithm: str = "PPO",
    total_timesteps: int = 200000,
    save_path: str = "models",
    log_path: str = "logs",
    seed: int = 42,
) -> None:
    """
    Train an RL agent.
    
    Args:
        config: Configuration dictionary
        algorithm: RL algorithm ('PPO', 'SAC', or 'DQN')
        total_timesteps: Total training timesteps
        save_path: Directory to save models
        log_path: Directory for logs
        seed: Random seed
    """
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm}_{timestamp}"
    
    print(f"Starting training: {run_name}")
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Seed: {seed}")
    
    # Create environments
    print("\nCreating environments...")
    train_env = create_env(config, seed=seed)
    eval_env = create_env(config, seed=seed + 1)
    
    # Determine tensorboard logging support
    use_tb = _tensorboard_available()
    tb_log_dir = log_path if use_tb else None
    if not use_tb:
        print("\nTensorBoard not found: disabling tensorboard logging for this run.")
    
    # Get training parameters from config
    training_config = config.get("training", {})
    learning_rate = training_config.get("learning_rate", 3e-4)
    batch_size = training_config.get("batch_size", 64)
    gamma = training_config.get("gamma", 0.99)
    
    # Create RL model
    print(f"\nCreating {algorithm} model...")
    # Select device automatically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if algorithm == "PPO":
        n_steps = training_config.get("n_steps", 2048)
        n_epochs = training_config.get("n_epochs", 10)
        
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=1,
            tensorboard_log=tb_log_dir,
            device=device,
        )
        
    elif algorithm == "SAC":
        buffer_size = training_config.get("buffer_size", 100000)
        learning_starts = training_config.get("learning_starts", 1000)
        tau = training_config.get("tau", 0.005)
        
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            verbose=1,
            tensorboard_log=tb_log_dir,
            device=device,
        )
        
    elif algorithm == "DQN":
        buffer_size = training_config.get("buffer_size", 100000)
        learning_starts = training_config.get("learning_starts", 1000)
        
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            verbose=1,
            tensorboard_log=tb_log_dir,
            device=device,
        )
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Setup callbacks
    eval_freq = training_config.get("eval_freq", 10000)
    n_eval_episodes = training_config.get("n_eval_episodes", 10)
    save_freq = training_config.get("save_freq", 50000)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, run_name),
        log_path=os.path.join(log_path, run_name),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(save_path, run_name, "checkpoints"),
        name_prefix="rl_model",
    )
    
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    # Train model
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=run_name,
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, f"{run_name}_final.zip")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    # Save configuration
    config_save_path = os.path.join(save_path, run_name, "config.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    print(f"Configuration saved to: {config_save_path}")
    
    return model


def evaluate_rl_agent(
    model_path: str,
    config: Dict,
    algo: Optional[str] = None,
    n_episodes: int = 10,
    seed: int = 42,
    render: bool = False,
) -> Dict:
    """
    Evaluate a trained RL agent.
    
    Args:
        model_path: Path to saved model
        config: Configuration dictionary
        n_episodes: Number of evaluation episodes
        seed: Random seed
        render: Whether to render episodes
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading model from: {model_path}")
    
    # Determine algorithm
    if algo is not None:
        algo = algo.upper()
        if algo == "PPO":
            model = PPO.load(model_path)
        elif algo == "SAC":
            model = SAC.load(model_path)
        elif algo == "DQN":
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm specified for evaluation: {algo}")
    else:
        # Fallback to heuristic based on path
        if "PPO" in model_path.upper():
            model = PPO.load(model_path)
        elif "SAC" in model_path.upper():
            model = SAC.load(model_path)
        elif "DQN" in model_path.upper():
            model = DQN.load(model_path)
        else:
            # Default to PPO with explicit notice
            print("Warning: Algorithm not specified and not inferred from path; defaulting to PPO.")
            model = PPO.load(model_path)
    
    # Create environment
    env = create_env(config, seed=seed)
    
    # Evaluate
    episode_rewards = []
    episode_costs = []
    episode_violations = []
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        violations = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # Check violations
            if not (40.0 <= info["temperature"] <= 50.0):
                violations += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        # Access base environment behind Monitor wrapper
        base_env = getattr(env, "env", env)
        episode_costs.append(base_env.economic.total_cost)
        episode_violations.append(violations)
        
        print(f"  Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Cost={base_env.economic.total_cost:.2f} CNY, "
              f"Violations={violations}")
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_cost": np.mean(episode_costs),
        "std_cost": np.std(episode_costs),
        "mean_violations": np.mean(episode_violations),
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "episode_violations": episode_violations,
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean cost: {results['mean_cost']:.2f} ± {results['std_cost']:.2f} CNY")
    print(f"  Mean violations: {results['mean_violations']:.1f}")
    
    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train RL agent for TES-HeatEx system")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        choices=["PPO", "SAC", "DQN"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate instead of train",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for evaluation",
    )
    
    args = parser.parse_args()
    
    # Load configuration (robust path resolution)
    cfg_path = resolve_config_path(args.config)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    if args.eval:
        # Evaluation mode
        if args.model_path is None:
            print("Error: --model-path required for evaluation")
            return

        evaluate_rl_agent(
            model_path=args.model_path,
            config=config,
            algo=args.algo,
            n_episodes=10,
            seed=args.seed,
            render=False,
        )
    else:
        # Training mode
        timesteps = args.timesteps or config["training"]["total_timesteps"]

        train_rl_agent(
            config=config,
            algorithm=args.algo,
            total_timesteps=timesteps,
            save_path=args.save_path,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
