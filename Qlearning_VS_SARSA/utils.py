"""
Utility functions for the Cleaning Robot RL project.

- Training loops (Q-Learning and SARSA)
- Metric collection
- Plotting functions (learning curves, heatmaps, comparisons)
- Statistical analysis helpers
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from environment import CleaningRobotEnv
from agents import QLearningAgent, SARSAAgent, BaseAgent
from config import (
    PHASE1_CONFIG, PHASE2_CONFIG, TRAINING_EPISODES, NUM_SEEDS, SEEDS,
    ALPHA, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    UCB_C, OPTIMISTIC_INIT, NUM_ACTIONS,
)


# =============================================================================
# Training Functions
# =============================================================================

def train_qlearning(env, agent, num_episodes=TRAINING_EPISODES):
    """
    Train a Q-Learning agent.

    Parameters
    ----------
    env : CleaningRobotEnv
    agent : QLearningAgent
    num_episodes : int

    Returns
    -------
    metrics : dict
        Episode-level metrics: rewards, coverages, steps, deaths, events.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
    }

    for ep in range(num_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        died = False

        while not env.done:
            action = agent.choose_action(state_idx)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            agent.update(state_idx, action, reward, next_state_idx, done)

            state_idx = next_state_idx
            total_reward += reward

            if info["event"] == "battery_dead":
                died = True

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])

    return metrics


def train_sarsa(env, agent, num_episodes=TRAINING_EPISODES):
    """
    Train a SARSA agent.

    Parameters
    ----------
    env : CleaningRobotEnv
    agent : SARSAAgent
    num_episodes : int

    Returns
    -------
    metrics : dict
        Episode-level metrics.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
    }

    for ep in range(num_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        action = agent.choose_action(state_idx)
        total_reward = 0.0
        died = False

        while not env.done:
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            if done:
                agent.update(state_idx, action, reward, next_state_idx, done)
            else:
                next_action = agent.choose_action(next_state_idx)
                agent.update(
                    state_idx, action, reward, next_state_idx, done,
                    next_action=next_action,
                )
                action = next_action

            state_idx = next_state_idx
            total_reward += reward

            if info["event"] == "battery_dead":
                died = True

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])

    return metrics


def create_agent(algorithm, exploration, env, **kwargs):
    """
    Factory function to create an agent with the specified algorithm and
    exploration strategy.

    Parameters
    ----------
    algorithm : str
        "qlearning" or "sarsa"
    exploration : str
        "epsilon_greedy", "ucb", or "optimistic"
    env : CleaningRobotEnv
    **kwargs : additional overrides for agent parameters

    Returns
    -------
    agent : BaseAgent subclass
    """
    params = {
        "num_states": env.num_states,
        "num_actions": env.num_actions,
        "alpha": kwargs.get("alpha", ALPHA),
        "gamma": kwargs.get("gamma", GAMMA),
        "exploration": exploration,
        "epsilon_start": kwargs.get("epsilon_start", EPSILON_START),
        "epsilon_min": kwargs.get("epsilon_min", EPSILON_MIN),
        "epsilon_decay": kwargs.get("epsilon_decay", EPSILON_DECAY),
        "ucb_c": kwargs.get("ucb_c", UCB_C),
        "optimistic_init": kwargs.get("optimistic_init", 0.0),
    }

    if algorithm == "qlearning":
        return QLearningAgent(**params)
    elif algorithm == "sarsa":
        return SARSAAgent(**params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_agent(env, agent, num_episodes=TRAINING_EPISODES):
    """Route to the correct training function based on agent type."""
    if isinstance(agent, SARSAAgent):
        return train_sarsa(env, agent, num_episodes)
    else:
        return train_qlearning(env, agent, num_episodes)


def run_experiment(algorithm, exploration, config=None, num_episodes=TRAINING_EPISODES,
                   seeds=SEEDS, **agent_kwargs):
    """
    Run a full experiment: train over multiple seeds, collect metrics.

    Parameters
    ----------
    algorithm : str
        "qlearning" or "sarsa"
    exploration : str
        "epsilon_greedy", "ucb", or "optimistic"
    config : dict, optional
        Environment config. Defaults to PHASE1_CONFIG.
    num_episodes : int
    seeds : list of int
    **agent_kwargs : passed to create_agent

    Returns
    -------
    all_metrics : list of dict
        One metrics dict per seed.
    agents : list of BaseAgent
        Trained agents (one per seed).
    """
    cfg = config or PHASE1_CONFIG
    all_metrics = []
    agents = []

    for seed in seeds:
        np.random.seed(seed)
        env = CleaningRobotEnv(cfg)
        agent = create_agent(algorithm, exploration, env, **agent_kwargs)

        env.reset(seed=seed)
        metrics = train_agent(env, agent, num_episodes)

        all_metrics.append(metrics)
        agents.append(agent)

    return all_metrics, agents


def evaluate_agent(env, agent, num_episodes=100, seed=42):
    """
    Evaluate a trained agent greedily (no exploration).

    Returns
    -------
    metrics : dict
        Same structure as training metrics.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
    }

    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        died = False

        while not env.done:
            action = agent.get_greedy_action(state_idx)
            next_state, reward, done, info = env.step(action)
            state_idx = env.state_to_index(next_state)
            total_reward += reward
            if info["event"] == "battery_dead":
                died = True

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])

    return metrics


# =============================================================================
# Plotting Functions
# =============================================================================

def smooth(data, window=100):
    """Simple moving average for smoothing curves."""
    if len(data) < window:
        window = max(1, len(data))
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def aggregate_metrics(all_metrics, key, window=100):
    """
    Aggregate a metric across seeds: compute mean and std of smoothed curves.

    Returns
    -------
    mean : np.array
    std : np.array
    """
    smoothed = [smooth(m[key], window) for m in all_metrics]
    min_len = min(len(s) for s in smoothed)
    smoothed = np.array([s[:min_len] for s in smoothed])
    return smoothed.mean(axis=0), smoothed.std(axis=0)


def plot_learning_curves(results_dict, metric="rewards", window=100,
                         title=None, ylabel=None, figsize=(12, 5)):
    """
    Plot learning curves for multiple experiments.

    Parameters
    ----------
    results_dict : dict
        {label: all_metrics} where all_metrics is a list of metric dicts (one per seed).
    metric : str
        Key in metrics dict to plot.
    window : int
        Smoothing window.
    title : str
    ylabel : str
    figsize : tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for label, all_metrics in results_dict.items():
        mean, std = aggregate_metrics(all_metrics, metric, window)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"Learning Curves: {metric}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_multi_metric(results_dict, metrics=None, window=100, figsize=(14, 10)):
    """
    Plot multiple metrics in subplots for comparison.

    Parameters
    ----------
    results_dict : dict
        {label: all_metrics}
    metrics : list of str, optional
        Which metrics to plot. Defaults to all main ones.
    figsize : tuple
    """
    if metrics is None:
        metrics = ["rewards", "coverages", "steps", "deaths"]

    ylabels = {
        "rewards": "Total Reward",
        "coverages": "Coverage (%)",
        "steps": "Steps per Episode",
        "deaths": "Battery Death Rate",
        "battery_at_end": "Battery Remaining",
    }

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for label, all_metrics in results_dict.items():
            mean, std = aggregate_metrics(all_metrics, metric, window)
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=label, linewidth=2)
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)

        ax.set_ylabel(ylabels.get(metric, metric), fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode", fontsize=12)
    fig.suptitle("Training Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_coverage_heatmap(env, agent, num_episodes=100, seed=42, figsize=(8, 7)):
    """
    Heatmap of how often the agent visits each tile during evaluation.

    Parameters
    ----------
    env : CleaningRobotEnv
    agent : BaseAgent
    num_episodes : int
    seed : int
    figsize : tuple
    """
    visit_counts = np.zeros((env.rows, env.cols))

    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        state_idx = env.state_to_index(state)
        visit_counts[state[0], state[1]] += 1

        while not env.done:
            action = agent.get_greedy_action(state_idx)
            next_state, reward, done, info = env.step(action)
            state_idx = env.state_to_index(next_state)
            visit_counts[next_state[0], next_state[1]] += 1

    # Normalize
    visit_counts /= num_episodes

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(visit_counts, cmap="YlOrRd", origin="upper")
    ax.set_title("Average Visit Frequency (per episode)", fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, label="Visits per episode")

    # Mark charger
    cr, cc = env.charger_pos
    ax.plot(cc, cr, "s", color="green", markersize=12, label="Charger")
    ax.legend(fontsize=11)

    plt.tight_layout()
    return fig, ax


def plot_battery_analysis(results_dict, window=100, figsize=(12, 5)):
    """
    Plot battery death rate over training for different strategies.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Death rate
    ax = axes[0]
    for label, all_metrics in results_dict.items():
        mean, std = aggregate_metrics(all_metrics, "deaths", window)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean * 100, label=label, linewidth=2)
        ax.fill_between(episodes, (mean - std) * 100, (mean + std) * 100, alpha=0.15)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Death Rate (%)", fontsize=12)
    ax.set_title("Battery Death Rate", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Battery remaining
    ax = axes[1]
    for label, all_metrics in results_dict.items():
        mean, std = aggregate_metrics(all_metrics, "battery_at_end", window)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Battery Remaining", fontsize=12)
    ax.set_title("Battery at Episode End", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def summary_table(results_dict, last_n=500):
    """
    Print a summary table of final performance (averaged over last_n episodes).

    Parameters
    ----------
    results_dict : dict
        {label: all_metrics}
    last_n : int
        Number of final episodes to average over.
    """
    print(f"{'Agent':<35} {'Reward':>10} {'Coverage':>10} {'Steps':>8} "
          f"{'Death%':>8} {'Battery':>8}")
    print("-" * 85)

    for label, all_metrics in results_dict.items():
        rewards = [np.mean(m["rewards"][-last_n:]) for m in all_metrics]
        coverages = [np.mean(m["coverages"][-last_n:]) for m in all_metrics]
        steps = [np.mean(m["steps"][-last_n:]) for m in all_metrics]
        deaths = [np.mean(m["deaths"][-last_n:]) for m in all_metrics]
        battery = [np.mean(m["battery_at_end"][-last_n:]) for m in all_metrics]

        print(f"{label:<35} "
              f"{np.mean(rewards):>10.1f} "
              f"{np.mean(coverages)*100:>9.1f}% "
              f"{np.mean(steps):>8.1f} "
              f"{np.mean(deaths)*100:>7.1f}% "
              f"{np.mean(battery):>8.1f}")

    print("-" * 85)


def plot_evaluation_comparison(eval_results, figsize=(12, 5)):
    """
    Bar chart comparing evaluation metrics across agents.

    Parameters
    ----------
    eval_results : dict
        {label: eval_metrics} from evaluate_agent().
    """
    labels = list(eval_results.keys())
    n = len(labels)

    avg_reward = [np.mean(eval_results[l]["rewards"]) for l in labels]
    avg_coverage = [np.mean(eval_results[l]["coverages"]) * 100 for l in labels]
    avg_deaths = [np.mean(eval_results[l]["deaths"]) * 100 for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    axes[0].bar(labels, avg_reward, color=colors)
    axes[0].set_title("Average Reward", fontsize=13)
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].bar(labels, avg_coverage, color=colors)
    axes[1].set_title("Average Coverage (%)", fontsize=13)
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis='x', rotation=30)

    axes[2].bar(labels, avg_deaths, color=colors)
    axes[2].set_title("Battery Death Rate (%)", fontsize=13)
    axes[2].set_ylim(0, 105)
    axes[2].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    return fig, axes


def plot_apartment_layout(env, figsize=(10, 10)):
    """
    Visualize the apartment layout: walls, furniture, walkable tiles, charger,
    and room labels.

    Parameters
    ----------
    env : CleaningRobotEnv
        An environment initialized with the apartment config.
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    # Build a grid where:
    #   0 = walkable (white)
    #   1 = furniture (light gray)
    #   2 = wall (dark gray)
    #   3 = charger (green — plotted as marker)
    grid = np.zeros((env.rows, env.cols))
    for (r, c) in env.furniture:
        grid[r, c] = 1
    for (r, c) in env.walls:
        grid[r, c] = 2

    # Custom colormap: walkable=white, furniture=lightgray, wall=darkgray
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#FFFFFF", "#B0B0B0", "#404040"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, origin="upper")

    # Draw grid lines
    for i in range(env.rows + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)
    for j in range(env.cols + 1):
        ax.axvline(j - 0.5, color="black", linewidth=0.5)

    # Mark charger
    cr, cc = env.charger_pos
    ax.plot(cc, cr, "s", color="limegreen", markersize=18, label="Charger",
            markeredgecolor="black", markeredgewidth=1.5)

    # Room labels
    room_labels = [
        (2.5, 3.0, "Living Room"),
        (2.5, 11.0, "Kitchen"),
        (7.5, 7.0, "Hallway"),
        (12.0, 3.0, "Bedroom"),
        (12.0, 9.5, "Bath"),
        (12.0, 13.5, "Storage"),
    ]
    for row_pos, col_pos, name in room_labels:
        ax.text(col_pos, row_pos, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          alpha=0.7, edgecolor="none"))

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FFFFFF", edgecolor="black", label="Walkable"),
        Patch(facecolor="#B0B0B0", edgecolor="black", label="Furniture"),
        Patch(facecolor="#404040", edgecolor="black", label="Wall"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="limegreen",
                   markersize=12, markeredgecolor="black", label="Charger"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    ax.set_title("Apartment Layout (Phase 2)", fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))

    plt.tight_layout()
    return fig, ax
