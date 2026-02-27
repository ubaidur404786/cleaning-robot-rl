"""
================================================================================
MAIN ENTRY POINT - Interactive Menu for Cleaning Robot RL Project
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (Q-Learning)
FILE: main.py
PURPOSE: Interactive command-line interface for training, testing, and watching

================================================================================
📚 OVERVIEW
================================================================================

This is the main entry point for the Cleaning Robot RL project.
It provides an interactive menu with the following options:

    1. Train Agent    - Train a new Q-Learning agent
    2. Test Agent     - Test a trained agent vs random baseline
    3. Watch Agent    - Visual demonstration of trained agent
    4. Quick Demo     - Short training + test demonstration
    5. Exit           - Exit the program

================================================================================
🚀 HOW TO USE
================================================================================

Run from command line:
    python main.py

Or import and call directly:
    from main import main
    main()

================================================================================
"""

import os
import sys
import pickle
import time

# ============================================================================
# IMPORT PROJECT MODULES
# ============================================================================

try:
    # Environment
    from env.cleaning_env import CleaningEnv
    
    # Agent
    from agent.q_learning_agent import QLearningAgent
    
    # Utils
    from utils.helpers import (
        format_time,
        format_duration,
        print_header,
        print_divider,
        print_key_value,
        Timer
    )
    from utils.plotting import plot_training_results, plot_comparison
    
except ImportError as e:
    print(f"\n  ERROR: Could not import required modules: {e}")
    print("  Make sure you're running from the project root directory.")
    print("  Expected structure:")
    print("    cleaning-robot-rl/")
    print("    ├── main.py")
    print("    ├── env/cleaning_env.py")
    print("    ├── agent/q_learning_agent.py")
    print("    └── utils/helpers.py, plotting.py")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default training parameters
DEFAULT_EPISODES = 3000
DEFAULT_MAX_STEPS = 200
DEFAULT_LEARNING_RATE = 0.2
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_END = 0.01
DEFAULT_EPSILON_DECAY = 0.998

# File paths
MODELS_DIR = "models"
PLOTS_DIR = "plots"
Q_TABLE_FILE = os.path.join(MODELS_DIR, "q_table.pkl")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print the application banner."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║      🤖 CLEANING ROBOT - Reinforcement Learning Project 🤖      ║")
    print("║                                                                  ║")
    print("║              Pure Q-Learning (No Hints, No Cheating!)            ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")


def print_menu():
    """Print the main menu options."""
    print("\n  ┌────────────────────────────────────────┐")
    print("  │           MAIN MENU                    │")
    print("  ├────────────────────────────────────────┤")
    print("  │  [1] Train Agent                       │")
    print("  │  [2] Test Agent (vs Random)            │")
    print("  │  [3] Watch Agent (Visual Demo)         │")
    print("  │  [4] Quick Demo (Train + Test)         │")
    print("  │  [5] Show Q-Table Statistics           │")
    print("  │  [0] Exit                              │")
    print("  └────────────────────────────────────────┘")


def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter."""
    input(f"\n  {message}")


def get_integer_input(prompt, default, min_val=1, max_val=100000):
    """
    Get integer input from user with validation.
    
    Parameters:
    -----------
    prompt : str
        Input prompt to show
    default : int
        Default value if user presses Enter
    min_val : int
        Minimum allowed value
    max_val : int
        Maximum allowed value
    
    Returns:
    --------
    int
        User's input or default value
    """
    while True:
        try:
            user_input = input(f"  {prompt} [{default}]: ").strip()
            
            if user_input == "":
                return default
            
            value = int(user_input)
            
            if min_val <= value <= max_val:
                return value
            else:
                print(f"    Please enter a value between {min_val} and {max_val}")
                
        except ValueError:
            print("    Please enter a valid integer")


def get_yes_no_input(prompt, default="y"):
    """
    Get yes/no input from user.
    
    Parameters:
    -----------
    prompt : str
        Prompt to show
    default : str
        Default value ('y' or 'n')
    
    Returns:
    --------
    bool
        True for yes, False for no
    """
    default_str = "Y/n" if default.lower() == "y" else "y/N"
    
    while True:
        user_input = input(f"  {prompt} [{default_str}]: ").strip().lower()
        
        if user_input == "":
            return default.lower() == "y"
        
        if user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            return False
        else:
            print("    Please enter 'y' (yes) or 'n' (no)")


# ============================================================================
# MENU OPTION HANDLERS
# ============================================================================

def train_agent():
    """
    Train a new Q-Learning agent.
    
    This function:
    1. Gets training parameters from user
    2. Creates environment and agent
    3. Runs training loop
    4. Saves the trained Q-table
    5. Generates training plots
    """
    
    print_header("TRAIN Q-LEARNING AGENT", width=60)
    
    # ========================================================================
    # GET TRAINING PARAMETERS
    # ========================================================================
    
    print("\n  Training Parameters:")
    print("  " + "-" * 40)
    
    num_episodes = get_integer_input(
        "Number of episodes",
        DEFAULT_EPISODES,
        min_val=100,
        max_val=50000
    )
    
    max_steps = get_integer_input(
        "Max steps per episode",
        DEFAULT_MAX_STEPS,
        min_val=50,
        max_val=1000
    )
    
    # Ask for advanced parameters
    if get_yes_no_input("Customize advanced parameters?", "n"):
        learning_rate = float(input(f"  Learning rate (alpha) [{DEFAULT_LEARNING_RATE}]: ") or DEFAULT_LEARNING_RATE)
        discount_factor = float(input(f"  Discount factor (gamma) [{DEFAULT_DISCOUNT_FACTOR}]: ") or DEFAULT_DISCOUNT_FACTOR)
        epsilon_decay = float(input(f"  Epsilon decay [{DEFAULT_EPSILON_DECAY}]: ") or DEFAULT_EPSILON_DECAY)
    else:
        learning_rate = DEFAULT_LEARNING_RATE
        discount_factor = DEFAULT_DISCOUNT_FACTOR
        epsilon_decay = DEFAULT_EPSILON_DECAY
    
    # ========================================================================
    # CONFIRM AND START TRAINING
    # ========================================================================
    
    print("\n  Training Configuration:")
    print("  " + "-" * 40)
    print_key_value("Episodes:", num_episodes)
    print_key_value("Max Steps:", max_steps)
    print_key_value("Learning Rate:", learning_rate)
    print_key_value("Discount Factor:", discount_factor)
    print_key_value("Epsilon Decay:", epsilon_decay)
    
    if not get_yes_no_input("Start training?", "y"):
        print("\n  Training cancelled.")
        return
    
    # ========================================================================
    # CREATE ENVIRONMENT AND AGENT
    # ========================================================================
    
    print("\n  Creating environment and agent...")
    
    env = CleaningEnv(render_mode=None)  # No rendering during training
    
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=DEFAULT_EPSILON_START,
        epsilon_end=DEFAULT_EPSILON_END,
        epsilon_decay=epsilon_decay
    )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print(f"\n  Starting training for {num_episodes} episodes...")
    print("  " + "-" * 50)
    
    # Track metrics
    episode_rewards = []
    completion_rates = []
    steps_per_episode = []
    epsilon_history = []
    
    timer = Timer().start()
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state, info = env.reset()
        episode_reward = 0
        
        # Episode loop
        for step in range(max_steps):
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, terminated)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if terminated:
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        completion_rates.append(info.get('completion_rate', 0))
        steps_per_episode.append(step + 1)
        epsilon_history.append(agent.epsilon)
        
        # Print progress every 10% of episodes
        if episode % (num_episodes // 10) == 0 or episode == num_episodes:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            avg_completion = sum(completion_rates[-100:]) / min(100, len(completion_rates))
            elapsed = timer.elapsed()
            
            print(f"  Episode {episode:5d}/{num_episodes} | "
                  f"Reward: {avg_reward:7.1f} | "
                  f"Completion: {avg_completion:5.1f}% | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {format_time(elapsed)}")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    
    training_time = timer.stop()
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    
    # Final statistics
    final_avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
    final_avg_completion = sum(completion_rates[-100:]) / min(100, len(completion_rates))
    initial_avg_reward = sum(episode_rewards[:100]) / min(100, len(episode_rewards))
    
    print(f"\n  Training Time:         {format_duration(training_time)}")
    print(f"  Final Avg Reward:      {final_avg_reward:.1f}")
    print(f"  Final Avg Completion:  {final_avg_completion:.1f}%")
    print(f"  Improvement:           {final_avg_reward - initial_avg_reward:.1f} reward points")
    print(f"  Final Epsilon:         {agent.epsilon:.4f}")
    print(f"  States Explored:       {len(agent.q_table)}")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    agent.save_q_table(Q_TABLE_FILE)
    print(f"\n  Q-table saved to: {Q_TABLE_FILE}")
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    
    if get_yes_no_input("Generate training plots?", "y"):
        print("\n  Generating plots...")
        plot_training_results(
            episode_rewards=episode_rewards,
            completion_rates=completion_rates,
            steps_per_episode=steps_per_episode,
            epsilon_history=epsilon_history,
            save_path=PLOTS_DIR,
            show_plot=True
        )
    
    env.close()
    wait_for_enter()


def test_agent():
    """
    Test a trained agent against random baseline.
    
    This function:
    1. Loads the trained Q-table
    2. Runs test episodes with trained agent
    3. Runs test episodes with random agent
    4. Compares and displays results
    """
    
    print_header("TEST Q-LEARNING AGENT", width=60)
    
    # ========================================================================
    # CHECK FOR TRAINED MODEL
    # ========================================================================
    
    if not os.path.exists(Q_TABLE_FILE):
        print(f"\n  ERROR: No trained model found at {Q_TABLE_FILE}")
        print("  Please train an agent first (option 1).")
        wait_for_enter()
        return
    
    # ========================================================================
    # GET TEST PARAMETERS
    # ========================================================================
    
    num_episodes = get_integer_input(
        "Number of test episodes",
        20,
        min_val=1,
        max_val=1000
    )
    
    max_steps = get_integer_input(
        "Max steps per episode",
        DEFAULT_MAX_STEPS,
        min_val=50,
        max_val=1000
    )
    
    # ========================================================================
    # LOAD MODEL AND CREATE ENVIRONMENT
    # ========================================================================
    
    print("\n  Loading trained model...")
    
    env = CleaningEnv(render_mode=None)
    
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n
    )
    agent.load_q_table(Q_TABLE_FILE)
    agent.epsilon = 0.0  # Disable exploration for testing
    
    print(f"  Loaded Q-table with {len(agent.q_table)} states")
    
    # ========================================================================
    # TEST TRAINED AGENT
    # ========================================================================
    
    print(f"\n  Testing TRAINED agent for {num_episodes} episodes...")
    
    trained_rewards = []
    trained_completions = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            
            if terminated:
                break
        
        trained_rewards.append(episode_reward)
        trained_completions.append(info.get('completion_rate', 0))
    
    # ========================================================================
    # TEST RANDOM AGENT
    # ========================================================================
    
    print(f"  Testing RANDOM agent for {num_episodes} episodes...")
    
    random_rewards = []
    random_completions = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            
            if terminated:
                break
        
        random_rewards.append(episode_reward)
        random_completions.append(info.get('completion_rate', 0))
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    
    import numpy as np
    
    print("\n  ┌────────────────────────────────────────────────────┐")
    print("  │                    COMPARISON                      │")
    print("  ├─────────────────────┬───────────────┬──────────────┤")
    print("  │       Metric        │ Trained Agent │ Random Agent │")
    print("  ├─────────────────────┼───────────────┼──────────────┤")
    print(f"  │ Avg Reward          │ {np.mean(trained_rewards):>11.1f}  │ {np.mean(random_rewards):>10.1f}  │")
    print(f"  │ Avg Completion %    │ {np.mean(trained_completions):>11.1f}% │ {np.mean(random_completions):>10.1f}% │")
    print(f"  │ Best Reward         │ {np.max(trained_rewards):>11.1f}  │ {np.max(random_rewards):>10.1f}  │")
    print(f"  │ Best Completion %   │ {np.max(trained_completions):>11.1f}% │ {np.max(random_completions):>10.1f}% │")
    print("  └─────────────────────┴───────────────┴──────────────┘")
    
    # Calculate improvement
    reward_improvement = np.mean(trained_rewards) - np.mean(random_rewards)
    completion_improvement = np.mean(trained_completions) - np.mean(random_completions)
    
    print(f"\n  Improvement over random:")
    print(f"    Reward:     +{reward_improvement:.1f} points")
    print(f"    Completion: +{completion_improvement:.1f}%")
    
    # Generate comparison plot
    if get_yes_no_input("Generate comparison plot?", "y"):
        plot_comparison(
            trained_rewards=trained_rewards,
            random_rewards=random_rewards,
            trained_completions=trained_completions,
            random_completions=random_completions,
            save_path=PLOTS_DIR,
            show_plot=True
        )
    
    env.close()
    wait_for_enter()


def watch_agent():
    """
    Watch the trained agent clean in real-time with visualization.
    
    Uses Pygame to render the environment and show the agent's actions.
    """
    
    print_header("WATCH TRAINED AGENT", width=60)
    
    # Check for trained model
    if not os.path.exists(Q_TABLE_FILE):
        print(f"\n  ERROR: No trained model found at {Q_TABLE_FILE}")
        print("  Please train an agent first (option 1).")
        wait_for_enter()
        return
    
    # Get parameters
    num_episodes = get_integer_input(
        "Number of episodes to watch",
        3,
        min_val=1,
        max_val=20
    )
    
    step_delay = float(input("  Step delay in seconds [0.3]: ") or 0.3)
    
    # Load model
    print("\n  Loading trained model...")
    
    env = CleaningEnv(render_mode="human")
    
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n
    )
    agent.load_q_table(Q_TABLE_FILE)
    agent.epsilon = 0.0
    
    print(f"  Loaded Q-table with {len(agent.q_table)} states")
    print(f"\n  Starting visualization...")
    print("  Close the Pygame window or press Ctrl+C to stop.\n")
    
    action_names = ['Up', 'Down', 'Left', 'Right', 'Clean']
    
    try:
        for episode in range(1, num_episodes + 1):
            print(f"  Episode {episode}/{num_episodes}")
            print("  " + "-" * 40)
            
            state, info = env.reset()
            episode_reward = 0
            
            for step in range(DEFAULT_MAX_STEPS):
                # Render environment
                env.render()
                
                # Choose action
                action = agent.choose_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Print step info
                print(f"    Step {step+1:3d}: Action={action_names[action]:<6} "
                      f"Reward={reward:>6.1f} "
                      f"Clean={info.get('cleaned_tiles', 0)}/{info.get('total_dirty', 23)}")
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Delay for visualization
                time.sleep(step_delay)
                
                if terminated:
                    break
            
            print(f"\n    Episode {episode} finished!")
            print(f"    Total Reward: {episode_reward:.1f}")
            print(f"    Completion: {info.get('completion_rate', 0):.1f}%")
            print()
            
            if episode < num_episodes:
                time.sleep(1)  # Pause between episodes
    
    except KeyboardInterrupt:
        print("\n\n  Visualization stopped by user.")
    
    finally:
        env.close()
    
    wait_for_enter()


def quick_demo():
    """
    Run a quick demonstration: short training + test.
    
    Good for quickly verifying the system works.
    """
    
    print_header("QUICK DEMO", width=60)
    
    print("\n  This will run a quick demonstration:")
    print("    - Train for 500 episodes (short)")
    print("    - Test for 10 episodes")
    print("    - Compare with random baseline")
    
    if not get_yes_no_input("Continue?", "y"):
        return
    
    # ========================================================================
    # QUICK TRAINING
    # ========================================================================
    
    print("\n  Phase 1: Quick Training (500 episodes)")
    print("  " + "-" * 40)
    
    env = CleaningEnv(render_mode=None)
    
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.3,  # Faster learning for demo
        epsilon_decay=0.99  # Faster decay for demo
    )
    
    episode_rewards = []
    completion_rates = []
    
    timer = Timer().start()
    
    for episode in range(1, 501):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(150):
            action = agent.choose_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            agent.learn(state, action, reward, next_state, terminated)
            state = next_state
            episode_reward += reward
            
            if terminated:
                break
        
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        completion_rates.append(info.get('completion_rate', 0))
        
        if episode % 100 == 0:
            avg = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            print(f"    Episode {episode}: Avg Reward = {avg:.1f}, ε = {agent.epsilon:.3f}")
    
    training_time = timer.stop()
    print(f"\n  Training complete in {format_duration(training_time)}")
    
    # ========================================================================
    # QUICK TEST
    # ========================================================================
    
    print("\n  Phase 2: Testing (10 episodes each)")
    print("  " + "-" * 40)
    
    agent.epsilon = 0.0  # No exploration
    
    # Test trained agent
    trained_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(150):
            action = agent.choose_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        trained_rewards.append(total_reward)
    
    # Test random agent
    random_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(150):
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        random_rewards.append(total_reward)
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    import numpy as np
    
    print("\n  Results:")
    print(f"    Trained Agent Avg Reward: {np.mean(trained_rewards):.1f}")
    print(f"    Random Agent Avg Reward:  {np.mean(random_rewards):.1f}")
    print(f"    Improvement: +{np.mean(trained_rewards) - np.mean(random_rewards):.1f}")
    
    env.close()
    wait_for_enter()


def show_q_table_stats():
    """
    Display statistics about the trained Q-table.
    """
    
    print_header("Q-TABLE STATISTICS", width=60)
    
    if not os.path.exists(Q_TABLE_FILE):
        print(f"\n  ERROR: No trained model found at {Q_TABLE_FILE}")
        wait_for_enter()
        return
    
    # Load Q-table
    with open(Q_TABLE_FILE, 'rb') as f:
        q_table = pickle.load(f)
    
    import numpy as np
    
    # Calculate statistics
    num_states = len(q_table)
    all_q_values = []
    
    for state, q_values in q_table.items():
        all_q_values.extend(q_values)
    
    all_q_values = np.array(all_q_values)
    
    print(f"\n  Q-Table Statistics:")
    print("  " + "-" * 40)
    print(f"    Number of states explored: {num_states}")
    print(f"    Total Q-values stored:     {len(all_q_values)}")
    print(f"    Max Q-value:               {np.max(all_q_values):.2f}")
    print(f"    Min Q-value:               {np.min(all_q_values):.2f}")
    print(f"    Mean Q-value:              {np.mean(all_q_values):.2f}")
    print(f"    Std Q-value:               {np.std(all_q_values):.2f}")
    
    # Show best actions for some states
    print("\n  Sample Best Actions:")
    print("  " + "-" * 40)
    
    action_names = ['Up', 'Down', 'Left', 'Right', 'Clean']
    
    for i, (state, q_values) in enumerate(list(q_table.items())[:5]):
        best_action = np.argmax(q_values)
        pos_idx = state // 2
        is_dirty = state % 2
        dirty_str = "Dirty" if is_dirty else "Clean"
        print(f"    State {state} (Pos {pos_idx}, {dirty_str}): "
              f"Best={action_names[best_action]} (Q={max(q_values):.1f})")
    
    wait_for_enter()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main entry point - runs the interactive menu loop.
    
    This function:
    1. Displays the main menu
    2. Gets user choice
    3. Calls appropriate handler function
    4. Repeats until user exits
    """
    
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        try:
            choice = input("\n  Enter your choice [0-5]: ").strip()
            
            if choice == "1":
                train_agent()
            elif choice == "2":
                test_agent()
            elif choice == "3":
                watch_agent()
            elif choice == "4":
                quick_demo()
            elif choice == "5":
                show_q_table_stats()
            elif choice == "0":
                print("\n  Thank you for using Cleaning Robot RL!")
                print("  Goodbye!\n")
                break
            else:
                print("\n  Invalid choice. Please enter 0-5.")
                wait_for_enter()
                
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n  Error: {e}")
            wait_for_enter()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the main function
    main()
