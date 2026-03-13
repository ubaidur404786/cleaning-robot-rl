"""
================================================================================
COMPREHENSIVE TESTING & COMPARISON SCRIPT
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning
FILE: comprehensive_test.py
PURPOSE: Train all three algorithms and generate detailed visualizations

This script:
1. Trains Q-Learning, SARSA, and DQN agents
2. Saves models and metrics
3. Generates individual and comparison plots
4. Creates analysis report

================================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.cleaning_env import CleaningEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SarsaAgent
from agent.dqn_agent import DQNAgent

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "models"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

NUM_EPISODES = 5000  # Can reduce for faster testing
RENDER_EVERY = 500

# Create directories
for dir_path in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    Path(dir_path).mkdir(exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def env_to_features(env):
    """Convert environment state to DQN feature vector."""
    features = np.zeros(125)
    
    # Position features
    features[0] = env.robot_row / 18.0
    features[1] = env.robot_col / 12.0
    
    # Dirt features (all tiles)
    dirty_mask = (env.dirt_map == 0)
    features[2:110] = dirty_mask.flatten()
    
    # Battery and DNUT features
    battery_ratio = getattr(env, 'battery', 50) / 100.0
    features[110:115] = battery_ratio
    features[115:125] = np.linspace(0, 1, 10)
    
    return features

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_q_learning(num_episodes=NUM_EPISODES):
    """Train Q-Learning agent."""
    print_section("TRAINING Q-LEARNING AGENT")
    
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
    
    success_count = 0
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        state = str((observation['position'][0], observation['position'][1]))
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = str((observation['position'][0], observation['position'][1]))
            
            # Update
            agent.learn(state, action, reward, next_state, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        # Record success
        is_success = terminated and info.get('success', False)
        if is_success:
            success_count += 1
        
        agent.decay_epsilon()
        
        metrics['episode'].append(episode + 1)
        metrics['total_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['success'].append(1 if is_success else 0)
        metrics['epsilon'].append(agent.epsilon)
        
        if (episode + 1) % RENDER_EVERY == 0:
            avg_reward = np.mean(metrics['total_reward'][-RENDER_EVERY:])
            success_rate = np.mean(metrics['success'][-RENDER_EVERY:]) * 100
            print(f"  Episode {episode + 1:5d} | Reward: {avg_reward:7.2f} | Success: {success_rate:5.1f}% | Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    
    # Save
    with open(os.path.join(MODELS_DIR, "q_learning_agent.pkl"), 'wb') as f:
        pickle.dump(agent, f)
    with open(os.path.join(RESULTS_DIR, "q_learning_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  ✓ Q-Learning training complete")
    return agent, metrics

def train_sarsa(num_episodes=NUM_EPISODES):
    """Train SARSA agent."""
    print_section("TRAINING SARSA AGENT")
    
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
    
    success_count = 0
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        state = str((observation['position'][0], observation['position'][1]))
        action = agent.choose_action(state, training=True)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = str((observation['position'][0], observation['position'][1]))
            next_action = agent.choose_action(next_state, training=True)
            
            # Update (SARSA uses next_action)
            agent.learn(state, action, reward, next_state, next_action, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            action = next_action
            
            if terminated or truncated:
                break
        
        # Record success
        is_success = terminated and info.get('success', False)
        if is_success:
            success_count += 1
        
        agent.decay_epsilon()
        
        metrics['episode'].append(episode + 1)
        metrics['total_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['success'].append(1 if is_success else 0)
        metrics['epsilon'].append(agent.epsilon)
        
        if (episode + 1) % RENDER_EVERY == 0:
            avg_reward = np.mean(metrics['total_reward'][-RENDER_EVERY:])
            success_rate = np.mean(metrics['success'][-RENDER_EVERY:]) * 100
            print(f"  Episode {episode + 1:5d} | Reward: {avg_reward:7.2f} | Success: {success_rate:5.1f}% | Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    
    # Save
    with open(os.path.join(MODELS_DIR, "sarsa_agent.pkl"), 'wb') as f:
        pickle.dump(agent, f)
    with open(os.path.join(RESULTS_DIR, "sarsa_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  ✓ SARSA training complete")
    return agent, metrics

def train_dqn(num_episodes=NUM_EPISODES):
    """Train DQN agent."""
    print_section("TRAINING DQN AGENT")
    
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
    
    success_count = 0
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        features = env_to_features(env)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            # Choose action
            action = agent.choose_action(features, training=True)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            next_features = env_to_features(env)
            
            # Learn
            agent.learn(features, action, reward, next_features, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            features = next_features
            
            if terminated or truncated:
                break
        
        # Record success
        is_success = terminated and info.get('success', False)
        if is_success:
            success_count += 1
        
        agent.decay_epsilon()
        
        metrics['episode'].append(episode + 1)
        metrics['total_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['success'].append(1 if is_success else 0)
        metrics['epsilon'].append(agent.epsilon)
        
        if (episode + 1) % RENDER_EVERY == 0:
            avg_reward = np.mean(metrics['total_reward'][-RENDER_EVERY:])
            success_rate = np.mean(metrics['success'][-RENDER_EVERY:]) * 100
            print(f"  Episode {episode + 1:5d} | Reward: {avg_reward:7.2f} | Success: {success_rate:5.1f}% | Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    
    # Save
    torch.save(agent.policy_net.state_dict(), os.path.join(MODELS_DIR, "dqn_model.pth"))
    with open(os.path.join(RESULTS_DIR, "dqn_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  ✓ DQN training complete")
    return agent, metrics

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_individual_performance(metrics, algorithm_name):
    """Plot individual algorithm performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{algorithm_name} - Performance Metrics", fontsize=16, fontweight='bold')
    
    # Plot 1: Total Reward per Episode
    ax = axes[0, 0]
    ax.plot(metrics['episode'], metrics['total_reward'], linewidth=0.5, alpha=0.7, label='Episode Reward')
    window = 100
    if len(metrics['total_reward']) >= window:
        moving_avg = np.convolve(metrics['total_reward'], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(metrics['total_reward'])+1), moving_avg, 'r-', linewidth=2, label='100-Episode Moving Avg')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total Reward', fontsize=11)
    ax.set_title('Rewards Over Training', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    ax = axes[0, 1]
    window = 100
    if len(metrics['success']) >= window:
        success_rate = np.convolve(metrics['success'], np.ones(window)/window, mode='valid') * 100
        ax.plot(range(window, len(metrics['success'])+1), success_rate, 'g-', linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate (100-Episode Window)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 3: Episode Length
    ax = axes[1, 0]
    ax.plot(metrics['episode'], metrics['episode_length'], linewidth=0.5, alpha=0.7, label='Episode Length')
    window = 100
    if len(metrics['episode_length']) >= window:
        moving_avg = np.convolve(metrics['episode_length'], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(metrics['episode_length'])+1), moving_avg, 'b-', linewidth=2, label='100-Episode Moving Avg')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Steps', fontsize=11)
    ax.set_title('Episode Length Over Training', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Epsilon
    ax = axes[1, 1]
    ax.plot(metrics['episode'], metrics['epsilon'], 'purple', linewidth=1)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Epsilon (ε)', fontsize=11)
    ax.set_title('Exploration Rate Decay', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def plot_algorithm_comparison(all_metrics):
    """Plot comparison of all algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Algorithm Comparison: Q-Learning vs SARSA vs DQN", fontsize=16, fontweight='bold')
    
    algorithms = list(all_metrics.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Average Reward
    ax = axes[0, 0]
    for (name, metrics), color in zip(all_metrics.items(), colors):
        window = 100
        if len(metrics['total_reward']) >= window:
            moving_avg = np.convolve(metrics['total_reward'], np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(metrics['total_reward'])+1), moving_avg, '-', linewidth=2.5, label=name, color=color)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Average Reward', fontsize=11)
    ax.set_title('Reward Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    ax = axes[0, 1]
    for (name, metrics), color in zip(all_metrics.items(), colors):
        window = 100
        if len(metrics['success']) >= window:
            success_rate = np.convolve(metrics['success'], np.ones(window)/window, mode='valid') * 100
            ax.plot(range(window, len(metrics['success'])+1), success_rate, '-', linewidth=2.5, label=name, color=color)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 3: Total Successes
    ax = axes[1, 0]
    for (name, metrics), color in zip(all_metrics.items(), colors):
        cumulative_success = np.cumsum(metrics['success'])
        ax.plot(metrics['episode'], cumulative_success, '-', linewidth=2, label=name, color=color)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Cumulative Successful Episodes', fontsize=11)
    ax.set_title('Total Successes Over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = "FINAL STATISTICS (Last 100 Episodes)\n" + "=" * 48 + "\n\n"
    for name, metrics in all_metrics.items():
        final_100_reward = np.mean(metrics['total_reward'][-100:])
        final_100_success = np.mean(metrics['success'][-100:]) * 100
        total_success = sum(metrics['success'])
        
        stats_text += f"{name}:\n"
        stats_text += f"  Avg Reward:    {final_100_reward:7.2f}\n"
        stats_text += f"  Success Rate:  {final_100_success:5.1f}%\n"
        stats_text += f"  Total Success: {total_success}/{len(metrics['success'])}\n\n"
    
    ax.text(0.05, 0.95, stats_text, fontsize=11, family='monospace', 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_report(all_metrics):
    """Create analysis report."""
    report = "=" * 80 + "\n"
    report += "COMPREHENSIVE TRAINING REPORT\n"
    report += "=" * 80 + "\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Total episodes per algorithm: {NUM_EPISODES}\n\n"
    
    for name, metrics in all_metrics.items():
        report += f"\n{name} PERFORMANCE\n"
        report += "-" * 80 + "\n"
        report += f"  Final Reward (100 episodes):    {np.mean(metrics['total_reward'][-100:]):8.2f}\n"
        report += f"  Final Success Rate:            {np.mean(metrics['success'][-100:]) * 100:5.1f}%\n"
        report += f"  Total Successes:               {int(sum(metrics['success']))}/{len(metrics['success'])}\n"
        report += f"  Average Episode Length:        {np.mean(metrics['episode_length'][-100:]):7.1f} steps\n"
        report += f"  Best Single Episode Reward:    {np.max(metrics['total_reward']):8.2f}\n"
    
    report += "\n" + "=" * 80 + "\n"
    return report

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE RL TESTING - All Algorithms")
    print("  Cleaning Robot Project")
    print("=" * 80)
    print(f"\n  Total episodes: {NUM_EPISODES}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_metrics = {}
    
    try:
        # Train all algorithms
        ql_agent, ql_metrics = train_q_learning(NUM_EPISODES)
        all_metrics['Q-Learning'] = ql_metrics
        
        sarsa_agent, sarsa_metrics = train_sarsa(NUM_EPISODES)
        all_metrics['SARSA'] = sarsa_metrics
        
        dqn_agent, dqn_metrics = train_dqn(NUM_EPISODES)
        all_metrics['DQN'] = dqn_metrics
        
        # Generate plots
        print_section("GENERATING VISUALIZATIONS")
        
        for name in ['Q-Learning', 'SARSA', 'DQN']:
            print(f"  Plotting {name}...", end=" ", flush=True)
            fig = plot_individual_performance(all_metrics[name], name)
            filename = f"{name.lower().replace('-', '_')}_performance.png"
            fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("✓")
        
        print("  Plotting Comparison...", end=" ", flush=True)
        fig = plot_algorithm_comparison(all_metrics)
        fig.savefig(os.path.join(PLOTS_DIR, "algorithm_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓")
        
        # Generate report
        print_section("GENERATING REPORT")
        report = create_report(all_metrics)
        report_path = os.path.join(RESULTS_DIR, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n{report}")
        
        print_section("COMPLETION SUMMARY")
        print(f"\n  ✓ All training complete!")
        print(f"\n  Models saved to: {MODELS_DIR}/")
        print(f"    - q_learning_agent.pkl")
        print(f"    - sarsa_agent.pkl")
        print(f"    - dqn_model.pth")
        print(f"\n  Metrics saved to: {RESULTS_DIR}/")
        print(f"    - q_learning_metrics.json")
        print(f"    - sarsa_metrics.json")
        print(f"    - dqn_metrics.json")
        print(f"\n  Plots saved to: {PLOTS_DIR}/")
        print(f"    - q_learning_performance.png")
        print(f"    - sarsa_performance.png")
        print(f"    - dqn_performance.png")
        print(f"    - algorithm_comparison.png")
        print(f"\n  Report: {report_path}")
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
