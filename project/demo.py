"""
Quick demonstration script for the TES-HeatEx optimization system.

This script provides a simple example of how to use the environment,
train a basic RL agent, and compare with baseline controller.
"""

import yaml
import numpy as np
from pathlib import Path

# Import project modules
from env.tes_heatex_env import TESHeatExEnv
from baselines.rule_based import SimpleTOUController
from rl_algorithms.train import train_rl_agent, evaluate_rl_agent
from metrics.calculator import MetricsCalculator, plot_comparison


def demo_environment():
    """Demonstrate basic environment usage."""
    print("=" * 80)
    print("DEMO 1: Environment Basics")
    print("=" * 80)
    
    # Load configuration
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Shorten simulation for demo
    config["simulation"]["duration"] = 168  # 1 week
    
    # Create environment
    env = TESHeatExEnv(config)
    
    print("\nEnvironment created successfully!")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Episode length: {config['simulation']['duration']} hours")
    
    # Run a few random steps
    print("\nRunning 10 random steps...")
    obs, info = env.reset(seed=42)
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i < 3:  # Print first 3 steps
            print(f"  Step {i+1}: Action={action}, Reward={reward:.2f}, "
                  f"Temp={info['temperature']:.1f}°C, Cost={info.get('cost', 0):.2f} CNY")
        
        if terminated or truncated:
            break
    
    print("\n✓ Environment demo complete!")


def demo_baseline_controller():
    """Demonstrate baseline controller."""
    print("\n" + "=" * 80)
    print("DEMO 2: Baseline Controller")
    print("=" * 80)
    
    # Load configuration
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config["simulation"]["duration"] = 168  # 1 week
    
    # Create environment and controller
    env = TESHeatExEnv(config)
    controller = SimpleTOUController()
    
    print("\nRunning baseline controller for 1 week...")
    
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    violations = 0
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = controller.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        if not (40.0 <= info['temperature'] <= 50.0):
            violations += 1
    
    episode_data = env.get_episode_data()
    
    print(f"\nBaseline Results:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Cost: {env.economic.total_cost:.2f} CNY")
    print(f"  Temperature Violations: {violations} steps ({violations/len(episode_data)*100:.1f}%)")
    print(f"  Mean Temperature: {np.mean([d['temperature'] for d in episode_data]):.1f}°C")
    print(f"  Mean SoC: {np.mean([d['soc'] for d in episode_data]):.2f}")
    
    print("\n✓ Baseline controller demo complete!")
    
    return episode_data


def demo_rl_training():
    """Demonstrate RL training (short version)."""
    print("\n" + "=" * 80)
    print("DEMO 3: RL Training (Short Demo)")
    print("=" * 80)

    algo_choice = input("\nSelect RL algorithm for demo [PPO/DQN] (default: PPO): ").strip().upper()
    if algo_choice not in {"PPO", "DQN"}:
        algo_choice = "PPO"

    config_path = "configs/ppo_config.yaml" if algo_choice == "PPO" else "configs/dqn_config.yaml"

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Shorten for quick demo
    config["simulation"]["duration"] = 168  # 1 week
    config["training"]["total_timesteps"] = 10000  # Reduced for demo
    config["training"]["eval_freq"] = 2000
    config["training"]["save_freq"] = 5000

    print(f"\nTraining {algo_choice} agent (10,000 steps, ~2 minutes)...")
    print("Note: This is a short demo. Real training requires 200,000+ steps.\n")

    model = train_rl_agent(
        config=config,
        algorithm=algo_choice,
        total_timesteps=10000,
        save_path="my_demo_models",
        log_path="demo_logs",
        seed=42,
    )

    print("\n✓ RL training demo complete!")
    print("  Model saved to: my_demo_models/")

    return config, algo_choice


def demo_comparison(baseline_data, config, algo):
    """Demonstrate comparison between controllers."""
    print("\n" + "=" * 80)
    print("DEMO 4: Controller Comparison")
    print("=" * 80)
    
    # Find the trained model
    model_dir = Path("my_demo_models")
    model_files = list(model_dir.glob(f"{algo}_*/**/best_model.zip"))
    
    if not model_files:
        print("\n⚠ No trained model found. Skipping comparison.")
        return
    
    model_path = str(model_files[0])
    print(f"\nEvaluating RL model: {model_path}")
    
    # Evaluate RL agent
    rl_results = evaluate_rl_agent(
        model_path=model_path,
        config=config,
        algo=algo,
        n_episodes=1,
        seed=42,
        render=False,
    )
    
    # Get RL episode data
    from stable_baselines3 import PPO, DQN
    env = TESHeatExEnv(config)
    if algo == "DQN":
        model = DQN.load(model_path)
    else:
        model = PPO.load(model_path)
    
    obs, info = env.reset(seed=42)
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    
    rl_data = env.get_episode_data()
    
    # Calculate cost comparison
    baseline_cost = sum(d['cost'] for d in baseline_data)
    rl_cost = sum(d['cost'] for d in rl_data)
    savings = baseline_cost - rl_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    print(f"\nComparison Results:")
    print(f"  Baseline Cost: {baseline_cost:.2f} CNY")
    print(f"  RL Cost: {rl_cost:.2f} CNY")
    print(f"  Savings: {savings:.2f} CNY ({savings_pct:.1f}%)")
    
    # Create comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(
        baseline_data=baseline_data,
        rl_data=rl_data,
        save_path="demo_comparison.png",
    )
    
    print("\n✓ Comparison complete!")
    print("  Plot saved to: demo_comparison.png")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("TES-HEATEX SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo will:")
    print("  1. Show basic environment usage")
    print("  2. Run a baseline controller")
    print("  3. Train a simple RL agent (short)")
    print("  4. Compare the results")
    print("\nEstimated time: 3-5 minutes")
    
    input("\nPress Enter to start...")
    
    try:
        # Demo 1: Environment
        demo_environment()
        
        # Demo 2: Baseline
        baseline_data = demo_baseline_controller()
        
        # Demo 3: RL Training
        config, algo = demo_rl_training()
        
        # Demo 4: Comparison
        demo_comparison(baseline_data, config, algo)
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review the comparison plot: demo_comparison.png")
        print("  2. Try full training: python rl_algorithms/train.py")
        print("  3. Read technical docs: docs/technical_doc.md")
        print("  4. Run tests: pytest tests/")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check that all dependencies are installed:")
        print("  pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()
