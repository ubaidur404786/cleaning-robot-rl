# 🤖 Cleaning Robot Reinforcement Learning

A beginner-friendly project that demonstrates **Q-Learning** by training a robot to clean a house using Reinforcement Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Project Overview

This project teaches you Reinforcement Learning through a practical, visual example:

- **A robot** learns to navigate a house
- **Clean dirty tiles** in three rooms (Kitchen, Living Room, Hallway)
- **Maximize rewards** by prioritizing high-value rooms (Kitchen first!)
- **Watch the robot improve** from random behavior to optimal cleaning

---

## 📚 What is Reinforcement Learning?

Imagine training a dog:

- Dog does something good → Give treat (positive reward)
- Dog does something bad → No treat (negative reward)
- Over time, the dog learns what behaviors lead to treats!

**Reinforcement Learning (RL)** works the same way with computer agents.

### Key Concepts

| Concept         | Description                        | In This Project                |
| --------------- | ---------------------------------- | ------------------------------ |
| **Agent**       | The learner that takes actions     | Cleaning robot                 |
| **Environment** | The world the agent interacts with | House with 3 rooms             |
| **State**       | Current situation/position         | Robot position + dirty tiles   |
| **Action**      | What the agent can do              | Move, Wait, Clean              |
| **Reward**      | Feedback for actions               | +30 clean kitchen, -1 move     |
| **Q-Table**     | "Cheat sheet" of action values     | Stored in `models/q_table.npy` |

---

## 🧠 What is Q-Learning?

Q-Learning is a **tabular reinforcement learning algorithm** that learns the expected value (Q-value) of taking each action in each state.

### The Q-Table

Think of it as a lookup table:

```
                    | Forward | Backward | Left | Right | Wait | Clean |
--------------------|---------|----------|------|-------|------|-------|
Kitchen (dirty)     |   5.2   |   3.1    | 2.0  |  1.5  | -1.0 | 25.0  |
Kitchen (clean)     |   8.5   |   4.2    | 6.0  |  3.2  |  0.5 | -5.0  |
Living Room (dirty) |   4.1   |   2.8    | 3.5  |  2.0  | -0.5 | 18.0  |
```

Higher Q-value = Better action!

### The Learning Formula

```
Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
```

Where:

- `Q(s,a)` = Current Q-value for state s, action a
- `α` = Learning rate (how fast to learn)
- `r` = Reward received
- `γ` = Discount factor (how much to value future rewards)
- `max Q(s',a')` = Best Q-value in the next state

### Example Update

```
Current Q(Kitchen_dirty, Clean) = 10.0
Robot cleans, gets reward = 30
Next state's best Q-value = 8.5
α = 0.1, γ = 0.95

Target = 30 + 0.95 * 8.5 = 38.075
New Q = 10.0 + 0.1 * (38.075 - 10.0) = 12.8

The Q-value increased! Robot learned cleaning in kitchen is good.
```

---

## 🏠 The Environment

### Room Layout

```
┌──────────────────────────────────┐
│                                  │
│  ┌─────────┬─────────────────┐  │
│  │         │                 │  │
│  │ KITCHEN │   LIVING ROOM   │  │
│  │ (Yellow)│   (Blue)        │  │
│  │         │                 │  │
│  ├─────────┴─────────────────┤  │
│  │                           │  │
│  │        HALLWAY            │  │
│  │        (Gray)             │  │
│  │                           │  │
│  └───────────────────────────┘  │
│                                  │
└──────────────────────────────────┘
```

### Room Priorities (Reward Values)

| Room        | Clean Reward | Priority |
| ----------- | ------------ | -------- |
| Kitchen     | +30          | Highest  |
| Living Room | +20          | Medium   |
| Hallway     | +10          | Lowest   |

### Actions

| Action   | ID  | Description        |
| -------- | --- | ------------------ |
| Forward  | 0   | Move up            |
| Backward | 1   | Move down          |
| Left     | 2   | Move left          |
| Right    | 3   | Move right         |
| Wait     | 4   | Stay in place      |
| Clean    | 5   | Clean current tile |

### Rewards

| Event                     | Reward |
| ------------------------- | ------ |
| Clean kitchen tile        | +30    |
| Clean living room tile    | +20    |
| Clean hallway tile        | +10    |
| Move                      | -1     |
| Wait                      | -2     |
| Clean already clean tile  | -5     |
| Hit wall                  | -3     |
| All tiles cleaned (bonus) | +100   |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download** this project

2. **Navigate** to the project directory:

   ```bash
   cd cleaning-robot-rl
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Option 1: Use the Main Menu

```bash
python main.py
```

This will show an interactive menu with all options.

#### Option 2: Train Directly

```bash
python train.py
```

This starts training immediately.

#### Option 3: Test a Trained Model

```bash
python test.py
```

This loads a trained model and shows the robot cleaning.

---

## 📁 Project Structure

```
cleaning_robot_rl/
│
├── main.py                 # Entry point with interactive menu
├── train.py                # Training logic
├── test.py                 # Testing saved model
│
├── env/
│   ├── __init__.py
│   └── cleaning_env.py     # Custom Gymnasium environment
│
├── agent/
│   ├── __init__.py
│   └── q_learning_agent.py # Q-Learning agent implementation
│
├── utils/
│   ├── __init__.py
│   ├── plotting.py         # Training visualization
│   └── helpers.py          # Utility functions
│
├── models/
│   └── q_table.npy         # Saved Q-table (after training)
│
├── plots/
│   ├── training_rewards.png
│   └── training_summary.png
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📊 Training Process

### What Happens During Training?

1. **Episode Start**: Robot at start position, all tiles dirty
2. **Action Selection**: Epsilon-greedy (random or best action)
3. **Environment Step**: Robot moves/cleans, gets reward
4. **Learning**: Q-table updated with new experience
5. **Episode End**: All tiles clean OR max steps reached
6. **Epsilon Decay**: Less exploration over time
7. **Repeat**: For thousands of episodes

### Training Parameters (Hyperparameters)

| Parameter         | Default | Description                      |
| ----------------- | ------- | -------------------------------- |
| `num_episodes`    | 2000    | Number of training episodes      |
| `learning_rate`   | 0.1     | How fast to update Q-values      |
| `discount_factor` | 0.95    | How much to value future rewards |
| `epsilon_start`   | 1.0     | Starting exploration rate        |
| `epsilon_end`     | 0.01    | Minimum exploration rate         |
| `epsilon_decay`   | 0.998   | Epsilon multiplier per episode   |

### Expected Output

During training, you'll see progress like:

```
Episode  100/2000 | Reward:   -52.0 | Avg(100):  -65.3 | ε: 0.8187 | Best: +45.0
Episode  200/2000 | Reward:   +35.0 | Avg(100):  -25.1 | ε: 0.6703 | Best: +85.0
Episode  500/2000 | Reward:  +125.0 | Avg(100):  +45.8 | ε: 0.3679 | Best: +145.0
Episode 1000/2000 | Reward:  +165.0 | Avg(100):  +95.3 | ε: 0.1353 | Best: +185.0
Episode 2000/2000 | Reward:  +195.0 | Avg(100): +155.7 | ε: 0.0183 | Best: +210.0
```

Notice how:

- Rewards **increase** over time (robot improves!)
- Epsilon **decreases** (less exploration, more exploitation)
- Average reward **grows** (consistent improvement)

---

## 🖥️ Visualization

The project includes a Pygame-based 2D visualization:

### Legend

- 🟨 **Yellow**: Kitchen (highest priority)
- 🟦 **Blue**: Living Room
- ⬜ **Gray**: Hallway
- 🟤 **Brown spots**: Dirty tiles
- 🟢 **Green circle**: Robot

### How to Enable Visualization

During training (slower):

```python
CONFIG['render_during_training'] = True
```

During testing (always enabled by default):

```python
python test.py
```

---

## 📈 How the Robot Improves

### Early Training (Episodes 0-100)

- Robot takes mostly **random actions**
- Often walks into walls
- Cleans randomly, may miss tiles
- Low rewards (often negative)

### Mid Training (Episodes 100-500)

- Robot starts **recognizing patterns**
- Learns to avoid walls
- Begins targeting dirty tiles
- Rewards become positive

### Late Training (Episodes 500-2000)

- Robot **efficiently navigates**
- Prioritizes high-reward rooms (kitchen first!)
- Minimal wasted movements
- Consistently high rewards

### Visual Proof

After training, check `plots/training_rewards.png` to see the improvement curve!

---

## 🧪 Testing the Model

After training, test the robot:

```bash
python test.py
```

Options:

1. **Watch trained robot** - Visual demonstration
2. **Compare trained vs random** - See the improvement
3. **Quick test** - No visualization, just statistics

### Expected Results

A well-trained robot should:

- Clean all tiles in 50-80 steps
- Achieve 90%+ success rate
- Score 150+ average reward
- Prioritize kitchen over other rooms

---

## 🔧 Customization

### Modify Training Parameters

Edit `train.py`:

```python
CONFIG = {
    'num_episodes': 3000,        # More training
    'learning_rate': 0.15,       # Faster learning
    'discount_factor': 0.99,     # Value future more
    'epsilon_decay': 0.999,      # Slower exploration decay
}
```

### Modify Environment

Edit `env/cleaning_env.py`:

- Change room layout in `_create_room_layout()`
- Adjust rewards (constants at top of file)
- Modify grid size
- Add obstacles

---

## 🎓 Learning Resources

This project follows educational approaches similar to:

- "A Beginner's Guide to Q-Learning"
- OpenAI Spinning Up
- Sutton & Barto's RL textbook

### Further Reading

1. **Q-Learning**: https://en.wikipedia.org/wiki/Q-learning
2. **Gymnasium**: https://gymnasium.farama.org/
3. **Epsilon-Greedy**: https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/

---

## ❓ Troubleshooting

### "Module not found" error

```bash
pip install -r requirements.txt
```

### "Pygame not initialized" error

Make sure pygame is installed:

```bash
pip install pygame
```

### Training is too slow

- Disable visualization: `CONFIG['render_during_training'] = False`
- Reduce episodes: `CONFIG['num_episodes'] = 1000`

### Robot not learning

Try:

- Increase episodes to 3000+
- Lower epsilon decay (0.997)
- Adjust learning rate (0.05-0.2)

---

## 📄 License

This project is for educational purposes. Feel free to use, modify, and share!

---

## 🙏 Acknowledgments

- OpenAI Gym / Gymnasium for the RL framework
- Pygame for visualization
- The RL community for educational resources

---

**Happy Learning! 🤖🏠✨**

_Created as a beginner-friendly introduction to Reinforcement Learning_
