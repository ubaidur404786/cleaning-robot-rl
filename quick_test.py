"""
================================================================================
QUICK COMPREHENSIVE TEST - Train all 3 algorithms with fewer episodes
================================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.cleaning_env import CleaningEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SarsaAgent

print("✓ Imports successful!")

MODELS_DIR = "models"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

NUM_EPISODES = 2000
RENDER_EVERY = 200

Path(MODELS_DIR).mkdir(exist_ok=True)
Path(PLOTS_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

def train_q_learning(num_episodes=NUM_EPISODES):
    """Train Q-Learning agent."""
    print("\n" + "="*80)
    print("  TRAINING Q-LEARNING AGENT")
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
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = str(state)  # Convert state integer to string key
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = str(next_state)  # Convert to string key
            
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
        
        if (episode + 1) % RENDER_EVERY == 0:
            avg_reward = np.mean(metrics['total_reward'][-RENDER_EVERY:])
            success_rate = np.mean(metrics['success'][-RENDER_EVERY:]) * 100
            print(f"  Episode {episode + 1:5d} | Reward: {avg_reward:7.2f} | Success: {success_rate:5.1f}%")
    
    env.close()
    
    with open(os.path.join(MODELS_DIR, "q_learning_agent_v2.pkl"), 'wb') as f:
        pickle.dump(agent, f)
    with open(os.path.join(RESULTS_DIR, "q_learning_metrics_v2.json"), 'w') as f:
        json.dump(metrics, f)
    
    print(f"\n  ✓ Q-Learning training complete")
    return metrics

def train_sarsa(num_episodes=NUM_EPISODES):
    """Train SARSA agent."""
    print("\n" + "="*80)
    print("  TRAINING SARSA AGENT")
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
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = str(state)  # Convert state integer to string key
        action = agent.choose_action(state, training=True)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < 1000:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = str(next_state)  # Convert to string key
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
        
        if (episode + 1) % RENDER_EVERY == 0:
            avg_reward = np.mean(metrics['total_reward'][-RENDER_EVERY:])
            success_rate = np.mean(metrics['success'][-RENDER_EVERY:]) * 100
            print(f"  Episode {episode + 1:5d} | Reward: {avg_reward:7.2f} | Success: {success_rate:5.1f}%")
    
    env.close()
    
    with open(os.path.join(MODELS_DIR, "sarsa_agent_v2.pkl"), 'wb') as f:
        pickle.dump(agent, f)
    with open(os.path.join(RESULTS_DIR, "sarsa_metrics_v2.json"), 'w') as f:
        json.dump(metrics, f)
    
    print(f"\n  ✓ SARSA training complete")
    return metrics

def plot_comparison(ql_metrics, sarsa_metrics):
    """Plot comparison."""
    print("\n" + "="*80)
    print("  GENERATING COMPARISON PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Learning vs SARSA Comparison", fontsize=16, fontweight='bold')
    
    # Rewards
    ax = axes[0, 0]
    window = 50
    ql_ma = np.convolve(ql_metrics['total_reward'], np.ones(window)/window, mode='valid')
    sarsa_ma = np.convolve(sarsa_metrics['total_reward'], np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(ql_metrics['total_reward'])+1), ql_ma, 'b-', linewidth=2, label='Q-Learning')
    ax.plot(range(window, len(sarsa_metrics['total_reward'])+1), sarsa_ma, 'r-', linewidth=2, label='SARSA')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Reward (Moving Avg)', fontsize=11)
    ax.set_title('Reward Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success Rate
    ax = axes[0, 1]
    ql_success = np.convolve(ql_metrics['success'], np.ones(window)/window, mode='valid') * 100
    sarsa_success = np.convolve(sarsa_metrics['success'], np.ones(window)/window, mode='valid') * 100
    ax.plot(range(window, len(ql_metrics['success'])+1), ql_success, 'b-', linewidth=2, label='Q-Learning')
    ax.plot(range(window, len(sarsa_metrics['success'])+1), sarsa_success,  'r-', linewidth=2, label='SARSA')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Cumulative Success
    ax = axes[1, 0]
    ql_cum = np.cumsum(ql_metrics['success'])
    sarsa_cum = np.cumsum(sarsa_metrics['success'])
    ax.plot(ql_metrics['episode'], ql_cum, 'b-', linewidth=2, label='Q-Learning')
    ax.plot(sarsa_metrics['episode'], sarsa_cum, 'r-', linewidth=2, label='SARSA')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Cumulative Successes', fontsize=11)
    ax.set_title('Total Successful Episodes', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stats
    ax = axes[1, 1]
    ax.axis('off')
    
    ql_final_reward = np.mean(ql_metrics['total_reward'][-100:])
    ql_final_success = np.mean(ql_metrics['success'][-100:]) * 100
    sarsa_final_reward = np.mean(sarsa_metrics['total_reward'][-100:])
    sarsa_final_success = np.mean(sarsa_metrics['success'][-100:]) * 100
    
    stats_text = "FINAL STATISTICS (Last 100 Episodes)\n"
    stats_text += "="*45 + "\n\n"
    stats_text += f"Q-Learning:\n"
    stats_text += f"  Avg Reward:   {ql_final_reward:7.2f}\n"
    stats_text += f"  Success Rate: {ql_final_success:5.1f}%\n\n"
    stats_text += f"SARSA:\n"
    stats_text += f"  Avg Reward:   {sarsa_final_reward:7.2f}\n"
    stats_text += f"  Success Rate: {sarsa_final_success:5.1f}%\n\n"
    stats_text += "="*45 + "\n"
    if ql_final_reward > sarsa_final_reward:
        stats_text += "Q-Learning rewards: BETTER\n"
    else:
        stats_text += "SARSA rewards: BETTER\n"
    
    ax.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "ql_vs_sarsa_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Comparison plot saved")

def create_analysis_report(ql_metrics, sarsa_metrics):
    """Create analysis report."""
    report = "="*80 + "\n"
    report += "Q-LEARNING vs SARSA - COMPREHENSIVE ANALYSIS REPORT\n"
    report += "="*80 + "\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Episodes Trained: {NUM_EPISODES}\n\n"
    
    report += "KEY FINDINGS\n"
    report += "-"*80 + "\n"
    
    ql_final_reward = np.mean(ql_metrics['total_reward'][-100:])
    ql_final_success = np.mean(ql_metrics['success'][-100:]) * 100
    sarsa_final_reward = np.mean(sarsa_metrics['total_reward'][-100:])
    sarsa_final_success = np.mean(sarsa_metrics['success'][-100:]) * 100
    
    report += "\nQ-LEARNING PERFORMANCE:\n"
    report += f"  • Final Average Reward (last 100 episodes): {ql_final_reward:.2f}\n"
    report += f"  • Final Success Rate: {ql_final_success:.1f}%\n"
    report += f"  • Total Successful Episodes: {int(sum(ql_metrics['success']))}/{len(ql_metrics['success'])}\n"
    report += f"  • Best Single Episode Reward: {max(ql_metrics['total_reward']):.2f}\n"
    report += f"  • Average Episode Length (last 100): {np.mean(ql_metrics['episode_length'][-100:]):.1f} steps\n"
    
    report += "\nSARSA PERFORMANCE:\n"
    report += f"  • Final Average Reward (last 100 episodes): {sarsa_final_reward:.2f}\n"
    report += f"  • Final Success Rate: {sarsa_final_success:.1f}%\n"
    report += f"  • Total Successful Episodes: {int(sum(sarsa_metrics['success']))}/{len(sarsa_metrics['success'])}\n"
    report += f"  • Best Single Episode Reward: {max(sarsa_metrics['total_reward']):.2f}\n"
    report += f"  • Average Episode Length (last 100): {np.mean(sarsa_metrics['episode_length'][-100:]):.1f} steps\n"
    
    report += "\nALGORITHM COMPARISON:\n"
    report += "-"*80 + "\n"
    reward_diff = ql_final_reward - sarsa_final_reward
    success_diff = ql_final_success - sarsa_final_success
    
    report += f"\nReward Difference: {reward_diff:+.2f} (Q-Learning " + ("better" if reward_diff > 0 else "worse") + ")\n"
    report += f"Success Rate Difference: {success_diff:+.1f}% (Q-Learning " + ("better" if success_diff > 0 else "worse") + ")\n"
    
    report += "\nINSIGHTS:\n"
    report += "-"*80 + "\n"
    report += "1. Q-LEARNING (OFF-POLICY):\n"
    report += "   - Learns the optimal policy while exploring\n"
    report += "   - Updates use the maximum Q-value of next state\n"
    report += "   - Generally converges faster\n"
    report += "   - May be more aggressive in learning\n\n"
    
    report += "2. SARSA (ON-POLICY):\n"
    report += "   - Learns the value of its exploratory policy\n"
    report += "   - Updates use the actual next action chosen\n"
    report += "   - More conservative approach\n"
    report += "   - Better for environments with penalties\n\n"
    
    report += "CONCLUSION:\n"
    report += "-"*80 + "\n"
    if ql_final_reward > sarsa_final_reward:
        report += "Q-Learning achieved better reward performance.\n"
        report += "This suggests the environment benefits from aggressive learning.\n"
    else:
        report += "SARSA achieved better reward performance.\n"
        report += "This suggests the environment benefits from conservative learning.\n"
    
    report += "\n" + "="*80 + "\n"
    return report

def main():
    print("\n" + "="*80)
    print("  COMPREHENSIVE RL TEST - Q-Learning & SARSA")
    print("  Cleaning Robot Project")
    print("="*80)
    print(f"\n  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Episodes: {NUM_EPISODES}")
    
    try:
        # Train algorithms
        ql_metrics = train_q_learning(NUM_EPISODES)
        sarsa_metrics = train_sarsa(NUM_EPISODES)
        
        # Generate visualizations
        plot_comparison(ql_metrics, sarsa_metrics)
        
        # Generate report
        print("\n" + "="*80)
        print("  GENERATING ANALYSIS REPORT")
        print("="*80)
        report = create_analysis_report(ql_metrics, sarsa_metrics)
        report_path = os.path.join(RESULTS_DIR, "ql_vs_sarsa_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n{report}")
        
        print("\n" + "="*80)
        print("  ✓ COMPLETION SUMMARY")
        print("="*80)
        print(f"\n  Models saved:")
        print(f"    • {MODELS_DIR}/q_learning_agent_v2.pkl")
        print(f"    • {MODELS_DIR}/sarsa_agent_v2.pkl")
        print(f"\n  Metrics saved:")
        print(f"    • {RESULTS_DIR}/q_learning_metrics_v2.json")
        print(f"    • {RESULTS_DIR}/sarsa_metrics_v2.json")
        print(f"\n  Analysis:")
        print(f"    • {RESULTS_DIR}/ql_vs_sarsa_report.txt")
        print(f"\n  Visualizations:")
        print(f"    • {PLOTS_DIR}/ql_vs_sarsa_comparison.png")
        print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
