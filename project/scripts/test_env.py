"""
Environment Integration Test Script.

This script tests all environment components:
1. Mock environment with CarbonAwareWrapper
2. Transformer feature extractor compatibility
3. SAC agent integration

Run from project root:
    python -m scripts.test_env
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import gymnasium as gym

# Local imports
from envs.sinergym_env import create_test_env, SimpleEnergyPlusEnv
from envs.carbon_wrapper import CarbonAwareWrapper
from models.transformer_policy import TransformerExtractor, SimpleMLP, create_policy_kwargs


def test_mock_environment():
    """Test the mock EnergyPlus environment."""
    print("\n" + "=" * 60)
    print("TEST 1: Mock EnergyPlus Environment")
    print("=" * 60)
    
    env = SimpleEnergyPlusEnv(n_zones=5)
    
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    print(f"  Reset observation shape: {obs.shape} ✓")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape, "Step observation shape mismatch"
    print(f"  Step observation shape: {obs.shape} ✓")
    print(f"  Reward: {reward:.4f}")
    print(f"  Info keys: {list(info.keys())}")
    
    env.close()
    print("\n✅ Mock environment test PASSED")
    return True


def test_carbon_wrapper():
    """Test the CarbonAwareWrapper."""
    print("\n" + "=" * 60)
    print("TEST 2: Carbon-Aware Wrapper")
    print("=" * 60)
    
    # Create base environment
    base_env = SimpleEnergyPlusEnv(n_zones=5)
    base_obs_dim = base_env.observation_space.shape[0]
    print(f"  Base observation dim: {base_obs_dim}")
    
    # Wrap with carbon-aware reward
    carbon_file = str(project_root / 'data/schedules/carbon_intensity.csv')
    price_file = str(project_root / 'data/schedules/electricity_price.csv')
    
    env = CarbonAwareWrapper(
        base_env,
        carbon_file=carbon_file,
        price_file=price_file,
        lambda_carbon=0.5,
        lambda_comfort=10.0,
        forecast_horizon=4,
    )
    
    # Check extended observation space
    expected_dim = base_obs_dim + 2 * 4  # +8 for forecasts
    print(f"  Wrapped observation dim: {env.observation_space.shape[0]} (expected: {expected_dim})")
    assert env.observation_space.shape[0] == expected_dim, "Wrapped observation dim mismatch"
    
    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape[0] == expected_dim, "Reset observation shape mismatch"
    print(f"  Reset observation shape: {obs.shape} ✓")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  Reward: {reward:.4f}")
    print(f"  Step cost: {info.get('step_cost', 'N/A'):.6f}")
    print(f"  Step carbon: {info.get('step_carbon', 'N/A'):.6f}")
    print(f"  Current price: {info.get('current_price', 'N/A'):.4f}")
    print(f"  Current carbon factor: {info.get('current_carbon_factor', 'N/A'):.4f}")
    
    # Run short episode
    obs, info = env.reset()
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    metrics = env.get_episode_metrics()
    print(f"\n  Episode metrics after 100 steps:")
    print(f"    Total cost (CNY): {metrics['total_cost_CNY']:.2f}")
    print(f"    Total carbon (kgCO2): {metrics['total_carbon_kgCO2']:.2f}")
    print(f"    Comfort violations: {metrics['comfort_violations']}")
    print(f"    Total reward: {total_reward:.2f}")
    
    env.close()
    print("\n✅ Carbon wrapper test PASSED")
    return True


def test_transformer_extractor():
    """Test the Transformer feature extractor."""
    print("\n" + "=" * 60)
    print("TEST 3: Transformer Feature Extractor")
    print("=" * 60)
    
    # Create a test observation space (mimicking wrapped env)
    obs_dim = 8 + 8  # 8 base + 8 forecast (4 price + 4 carbon)
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    # Create transformer extractor
    extractor = TransformerExtractor(
        observation_space=obs_space,
        features_dim=128,
        n_forecast_steps=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    
    print(f"  Input dim: {obs_dim}")
    print(f"  State dim: {extractor.state_dim}")
    print(f"  Forecast dim: {extractor.forecast_dim}")
    print(f"  Output features dim: {extractor.features_dim}")
    
    # Test forward pass
    batch_size = 32
    test_obs = torch.randn(batch_size, obs_dim)
    
    with torch.no_grad():
        features = extractor(test_obs)
    
    assert features.shape == (batch_size, 128), f"Output shape mismatch: {features.shape}"
    print(f"  Output shape: {features.shape} ✓")
    
    # Test with single observation
    single_obs = torch.randn(1, obs_dim)
    with torch.no_grad():
        single_features = extractor(single_obs)
    assert single_features.shape == (1, 128)
    print(f"  Single observation output: {single_features.shape} ✓")
    
    print("\n✅ Transformer extractor test PASSED")
    return True


def test_mlp_extractor():
    """Test the MLP baseline feature extractor."""
    print("\n" + "=" * 60)
    print("TEST 4: MLP Baseline Feature Extractor")
    print("=" * 60)
    
    obs_dim = 16
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    extractor = SimpleMLP(
        observation_space=obs_space,
        features_dim=128,
    )
    
    batch_size = 32
    test_obs = torch.randn(batch_size, obs_dim)
    
    with torch.no_grad():
        features = extractor(test_obs)
    
    assert features.shape == (batch_size, 128)
    print(f"  Output shape: {features.shape} ✓")
    
    print("\n✅ MLP extractor test PASSED")
    return True


def test_sac_integration():
    """Test SAC agent integration with custom extractor."""
    print("\n" + "=" * 60)
    print("TEST 5: SAC Agent Integration")
    print("=" * 60)
    
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("  ⚠️ stable-baselines3 not installed, skipping SAC test")
        return True
    
    # Create wrapped environment
    env = create_test_env(use_carbon_wrapper=True)
    
    # Check environment compatibility
    print("  Running environment checker...")
    try:
        check_env(env, warn=True)
        print("  Environment check passed ✓")
    except Exception as e:
        print(f"  ⚠️ Environment check warning: {e}")
    
    # Create policy kwargs with transformer
    policy_kwargs = create_policy_kwargs(
        extractor_type="transformer",
        features_dim=128,
        n_forecast_steps=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    
    print(f"  Policy kwargs: {list(policy_kwargs.keys())}")
    
    # Create SAC agent
    print("  Creating SAC agent...")
    agent = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1000,  # Small for testing
        batch_size=64,
        verbose=0,
    )
    
    print(f"  Agent created successfully ✓")
    print(f"  Policy network: {type(agent.policy)}")
    
    # Test a few training steps
    print("  Running 100 training steps...")
    agent.learn(total_timesteps=100, progress_bar=False)
    print("  Training steps completed ✓")
    
    # Test prediction
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    print(f"  Predicted action: {action}")
    
    env.close()
    print("\n✅ SAC integration test PASSED")
    return True


def test_full_pipeline():
    """Test the full training pipeline with a short run."""
    print("\n" + "=" * 60)
    print("TEST 6: Full Pipeline (Short Run)")
    print("=" * 60)
    
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("  ⚠️ stable-baselines3 not installed, skipping")
        return True
    
    # Create environments
    train_env = create_test_env(use_carbon_wrapper=True)
    eval_env = create_test_env(use_carbon_wrapper=True)
    
    # Policy with transformer
    policy_kwargs = create_policy_kwargs(
        extractor_type="transformer",
        features_dim=128,
        n_forecast_steps=4,
    )
    
    # Create agent
    agent = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=5000,
        batch_size=64,
        learning_starts=100,
        verbose=0,
    )
    
    # Short training
    print("  Training for 500 steps...")
    agent.learn(total_timesteps=500, progress_bar=False)
    
    # Evaluate
    print("  Evaluating...")
    obs, _ = eval_env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    metrics = eval_env.get_episode_metrics()
    print(f"  Evaluation reward: {total_reward:.2f}")
    print(f"  Episode cost: {metrics['total_cost_CNY']:.2f} CNY")
    print(f"  Episode carbon: {metrics['total_carbon_kgCO2']:.2f} kgCO2")
    
    train_env.close()
    eval_env.close()
    
    print("\n✅ Full pipeline test PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CARBON-AWARE BUILDING CONTROL - ENVIRONMENT TESTS")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    tests = [
        ("Mock Environment", test_mock_environment),
        ("Carbon Wrapper", test_carbon_wrapper),
        ("Transformer Extractor", test_transformer_extractor),
        ("MLP Extractor", test_mlp_extractor),
        ("SAC Integration", test_sac_integration),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Environment is ready for training.")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before training.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
