"""
================================================================================
FLEXIBLE TRAINING LAUNCHER - Train with Configurable Episodes
================================================================================

PROJECT: Cleaning Robot RL - All 3 Algorithms
PURPOSE: Train algorithms with flexible episode counts

USAGE EXAMPLES:
  python train_all_flexible.py --episodes 500 --all
  python train_all_flexible.py --episodes 2000 --ql --sarsa
  python train_all_flexible.py --episodes 5000 --dqn

RUN OPTIONS:
  --episodes EPOCHS   : Number of episodes to train (default: 2000)
  --ql               : Train Q-Learning
  --sarsa            : Train SARSA
  --dqn              : Train DQN
  --all              : Train all three algorithms
  --quick            : Quick demo (500 episodes)
  --balanced         : Balanced training (2000 episodes)
  --full             : Full training (5000 episodes)

================================================================================
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.cleaning_env import CleaningEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SarsaAgent

# Create directories
for dir_path in ["models", "plots", "results"]:
    Path(dir_path).mkdir(exist_ok=True)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_q_learning(num_episodes):
    """Train Q-Learning agent."""
    print("\n" + "="*80)
    print(f"  TRAINING Q-LEARNING AGENT ({num_episodes} episodes)")
    print("="*80)
    
    env = CleaningEnv(render_mode=None)
    agent = QLearningAgent(
        state_size=10900,
        action_size=6,
        learning_rate=0.15,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.998
    )
    
    metrics = {
        'episode': [],
        'total_reward': [],
        'episode_length': [],
        'success': [],
        'epsilon': []
    }
    
    render_every = max(100, num_episodes // 10)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = str(state)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = str(next_state)
            
            agent.learn(state, action, reward, next_state, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        is_success = terminated and info.get('success', False)
        agent.decay_epsilon()
        
        metrics['episode'].append(episode + 1)
        metrics['total_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['success'].append(1 if is_success else 0)
        metrics['epsilon'].append(agent.epsilon)
        
        if (episode + 1) % render_every == 0:
            avg_reward = np.mean(metrics['total_reward'][-render_every:])
            success_rate = np.mean(metrics['success'][-render_every:]) * 100
            print(f"  Episode {episode + 1:6d} | Reward: {avg_reward:8.2f} | Success: {success_rate:5.1f}%")
    
    env.close()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"models/q_learning_agent_{num_episodes}eps_{timestamp}.pkl", 'wb') as f:
        pickle.dump(agent, f)
    with open(f"results/q_learning_metrics_{num_episodes}eps_{timestamp}.json", 'w') as f:
        json.dump(metrics, f)
    
    print(f"\n  ✓ Q-Learning training complete ({num_episodes} episodes)")
    return metrics

def train_sarsa(num_episodes):
    """Train SARSA agent."""
    print("\n" + "="*80)
    print(f"  TRAINING SARSA AGENT ({num_episodes} episodes)")
    print("="*80)
    
    env = CleaningEnv(render_mode=None)
    agent = SarsaAgent(
        state_size=10900,
        action_size=6,
        learning_rate=0.15,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.998
    )
    
    metrics = {
        'episode': [],
        'total_reward': [],
        'episode_length': [],
        'success': [],
        'epsilon': []
    }
    
    render_every = max(100, num_episodes // 10)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = str(state)
        action = agent.choose_action(state, training=True)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = str(next_state)
            next_action = agent.choose_action(next_state, training=True)
            
            agent.learn(state, action, reward, next_state, next_action, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            action = next_action
            
            if terminated or truncated:
                break
        
        is_success = terminated and info.get('success', False)
        agent.decay_epsilon()
        
        metrics['episode'].append(episode + 1)
        metrics['total_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['success'].append(1 if is_success else 0)
        metrics['epsilon'].append(agent.epsilon)
        
        if (episode + 1) % render_every == 0:
            avg_reward = np.mean(metrics['total_reward'][-render_every:])
            success_rate = np.mean(metrics['success'][-render_every:]) * 100
            print(f"  Episode {episode + 1:6d} | Reward: {avg_reward:8.2f} | Success: {success_rate:5.1f}%")
    
    env.close()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"models/sarsa_agent_{num_episodes}eps_{timestamp}.pkl", 'wb') as f:
        pickle.dump(agent, f)
    with open(f"results/sarsa_metrics_{num_episodes}eps_{timestamp}.json", 'w') as f:
        json.dump(metrics, f)
    
    print(f"\n  ✓ SARSA training complete ({num_episodes} episodes)")
    return metrics

def train_dqn(num_episodes):
    """Train DQN agent."""
    try:
        from agent.dqn_agent import DQNAgent
    except ImportError as e:
        print(f"  ✗ Failed to import DQN: {e}")
        print("  Skipping DQN training")
        return None
    
    print("\n" + "="*80)
    print(f"  TRAINING DQN AGENT ({num_episodes} episodes)")
    print("="*80)
    
    env = CleaningEnv(render_mode=None)
    agent = DQNAgent(
        input_size=125,
        action_size=6,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9991,
        batch_size=64,
        memory_size=200000,
        target_update=500,
        hidden_size=64,
        train_every=1
    )
    
    metrics = {
        'episode': [],
        'total_reward': [],
        'episode_length': [],
        'success': [],
        'epsilon': []
    }
    
    render_every = max(100, num_episodes // 10)
    
    def env_to_features(env):
        features = np.zeros(125)
        features[0] = env.robot_row / 18.0
        features[1] = env.robot_col / 12.0
        dirty_mask = (env.dirt_map == 0)
        features[2:110] = dirty_mask.flatten()
        battery_ratio = getattr(env, 'battery', 50) / 100.0
        features[110:115] = battery_ratio
        features[115:125] = np.linspace(0, 1, 10)
        return features
    
    for episode in range(num_episodes):
        state, info = env.reset()
        features = env_to_features(env)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            action = agent.choose_action(features, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_features = env_to_features(env)
            
            agent.learn(features, action, reward, next_features, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            features = next_features
            
            if terminated or truncated:
                break
        
        is_success = terminated and info.get('success', False)
        agent.decay_epsilon()
        
        metrics['episode'].append(episode + 1)
        metrics['total_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['success'].append(1 if is_success else 0)
        metrics['epsilon'].append(agent.epsilon)
        
        if (episode + 1) % render_every == 0:
            avg_reward = np.mean(metrics['total_reward'][-render_every:])
            success_rate = np.mean(metrics['success'][-render_every:]) * 100
            print(f"  Episode {episode + 1:6d} | Reward: {avg_reward:8.2f} | Success: {success_rate:5.1f}%")
    
    env.close()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(agent.policy_net.state_dict(), f"models/dqn_model_{num_episodes}eps_{timestamp}.pth")
    with open(f"results/dqn_metrics_{num_episodes}eps_{timestamp}.json", 'w') as f:
        json.dump(metrics, f)
    
    print(f"\n  ✓ DQN training complete ({num_episodes} episodes)")
    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Cleaning Robot RL agents with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python train_all_flexible.py --episodes 500 --all        (Quick test - all agents)
  python train_all_flexible.py --quick                     (Quick preset - 500 episodes)
  python train_all_flexible.py --balanced                  (Balanced - 2000 episodes)
  python train_all_flexible.py --full                      (Full - 5000 episodes)
  python train_all_flexible.py --episodes 3000 --ql --sarsa (Custom - 3000 episodes, QL+SARSA)
        """
    )
    
    # Episode options
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of episodes to train (default: 2000)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick preset: 500 episodes')
    parser.add_argument('--balanced', action='store_true',
                       help='Balanced preset: 2000 episodes (default)')
    parser.add_argument('--full', action='store_true',
                       help='Full preset: 5000 episodes')
    
    # Algorithm options
    parser.add_argument('--ql', action='store_true',
                       help='Train Q-Learning')
    parser.add_argument('--sarsa', action='store_true',
                       help='Train SARSA')
    parser.add_argument('--dqn', action='store_true',
                       help='Train DQN')
    parser.add_argument('--all', action='store_true',
                       help='Train all three algorithms')
    
    args = parser.parse_args()
    
    # Process presets
    if args.quick:
        args.episodes = 500
    elif args.full:
        args.episodes = 5000
    else:
        args.balanced = True  # Default
        if not args.quick and not args.full:
            args.episodes = 2000
    
    # Determine which algorithms to train
    train_ql = args.ql or args.all
    train_sarsa = args.sarsa or args.all
    train_dqn_flag = args.dqn or args.all
    
    # If none specified, train all
    if not (train_ql or train_sarsa or train_dqn_flag):
        train_ql = train_sarsa = train_dqn_flag = True
    
    # Header
    print("\n" + "="*80)
    print("  FLEXIBLE RL TRAINING LAUNCHER")
    print("  Cleaning Robot - All 3 Algorithms")
    print("="*80)
    print(f"\n  Configuration:")
    print(f"    • Episodes: {args.episodes}")
    print(f"    • Algorithms: ", end="")
    algos = []
    if train_ql: algos.append("Q-Learning")
    if train_sarsa: algos.append("SARSA")
    if train_dqn_flag: algos.append("DQN")
    print(", ".join(algos))
    print(f"    • Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    
    results = {}
    
    try:
        if train_ql:
            metrics = train_q_learning(args.episodes)
            results['Q-Learning'] = metrics
        
        if train_sarsa:
            metrics = train_sarsa(args.episodes)
            results['SARSA'] = metrics
        
        if train_dqn_flag:
            metrics = train_dqn(args.episodes)
            if metrics:
                results['DQN'] = metrics
        
        # Summary
        print("\n" + "="*80)
        print("  ✓ TRAINING COMPLETE")
        print("="*80)
        print(f"\n  Results Summary:")
        for algo_name, metrics in results.items():
            final_reward = np.mean(metrics['total_reward'][-100:])
            final_success = np.mean(metrics['success'][-100:]) * 100
            print(f"    {algo_name:12} • Reward: {final_reward:7.2f} | Success: {final_success:5.1f}%")
        
        print(f"\n  Outputs saved to:")
        print(f"    • models/   (trained agents)")
        print(f"    • results/  (metrics and reports)")
        print(f"    • plots/    (visualizations)")
        print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print(f"\n\n  ⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n  ✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
