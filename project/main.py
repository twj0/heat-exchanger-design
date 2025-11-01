#!/usr/bin/env python3
"""
Main entry point for the TES-HeatEx optimization system.

This script provides a unified interface to:
- Train RL agents
- Evaluate controllers
- Run demonstrations
- Perform comparative studies
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="TES-HeatEx Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  train     Train an RL agent
  eval      Evaluate a controller
  demo      Run demonstration
  compare   Compare multiple controllers
  
Examples:
  python main.py demo                           # Run quick demo
  python main.py train --algo PPO               # Train PPO agent
  python main.py eval --baseline                # Evaluate baseline controller
  python main.py eval --model models/best.zip   # Evaluate trained model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo parser
    demo_parser = subparsers.add_parser('demo', help='Run system demonstration')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--algo', choices=['PPO', 'SAC', 'DQN'], 
                             default='SAC', help='RL algorithm to use')
    train_parser.add_argument('--config', default='configs/default.yaml',
                             help='Path to configuration file')
    train_parser.add_argument('--timesteps', type=int, default=None,
                             help='Training timesteps (overrides config)')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Eval parser
    eval_parser = subparsers.add_parser('eval', help='Evaluate controller')
    eval_parser.add_argument('--baseline', action='store_true',
                            help='Evaluate baseline controller')
    eval_parser.add_argument('--model', help='Path to trained model')
    eval_parser.add_argument('--config', default='configs/default.yaml',
                            help='Path to configuration file')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--output', default='results',
                            help='Output directory for results')
    
    # Compare parser
    compare_parser = subparsers.add_parser('compare', help='Compare controllers')
    compare_parser.add_argument('--config', default='configs/default.yaml',
                               help='Path to configuration file')
    compare_parser.add_argument('--baseline', action='store_true',
                               help='Include baseline controller in comparison')
    compare_parser.add_argument('--models', nargs='+', 
                               help='Paths to trained models')
    compare_parser.add_argument('--names', nargs='+',
                               help='Names for each model')
    compare_parser.add_argument('--output', default='results/comparison',
                               help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        from demo import main as run_demo
        run_demo()
        
    elif args.command == 'train':
        from rl_algorithms.train import train_rl_agent
        import yaml
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Override timesteps if specified
        if args.timesteps:
            config['training']['total_timesteps'] = args.timesteps
            
        print(f"Training {args.algo} agent...")
        model = train_rl_agent(
            config=config,
            algorithm=args.algo,
            total_timesteps=config['training']['total_timesteps'],
            save_path=config['training']['model_dir'],
            log_path=config['training']['log_dir'],
            seed=args.seed
        )
        print(f"Training complete! Model saved to {config['training']['model_dir']}")
        
    elif args.command == 'eval':
        import yaml
        
        if args.baseline:
            from baselines.rule_based import evaluate_baseline
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                
            print("Evaluating baseline controller...")
            results = evaluate_baseline(
                config=config,
                n_episodes=args.episodes
            )
            print("Baseline evaluation complete!")
            
        elif args.model:
            from rl_algorithms.train import evaluate_rl_agent
            
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                
            print(f"Evaluating model: {args.model}")
            results = evaluate_rl_agent(
                model_path=args.model,
                config=config,
                n_episodes=args.episodes,
                seed=42
            )
            print("Model evaluation complete!")
            
        else:
            print("Error: Either --baseline or --model must be specified for evaluation")
            sys.exit(1)
            
    elif args.command == 'compare':
        # Placeholder for comparison functionality
        print("Comparison functionality not yet implemented")
        sys.exit(1)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()