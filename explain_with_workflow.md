# 🤖 Cleaning Robot using Reinforcement Learning (Q-Learning)

> A beginner-friendly project demonstrating **Pure Reinforcement Learning** with a robot that learns to clean a house through trial and error!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-red.svg)

---

## 📋 Table of Contents

1. [What is This Project?](#-what-is-this-project)
2. [How Reinforcement Learning Works](#-how-reinforcement-learning-works)
3. [Project Architecture](#-project-architecture)
4. [Complete Code Workflow](#-complete-code-workflow)
5. [Step-by-Step Flow Diagrams](#-step-by-step-flow-diagrams)
6. [Understanding the Q-Learning Algorithm](#-understanding-the-q-learning-algorithm)
7. [The House Environment](#-the-house-environment)
8. [File-by-File Explanation](#-file-by-file-explanation)
9. [How to Run](#-how-to-run)
10. [Expected Results](#-expected-results)
11. [Common Questions](#-common-questions)

---

## 🎯 What is This Project?

This project creates a **virtual cleaning robot** that learns to clean a house by itself. Unlike traditional programming where we tell the robot exactly what to do, here we use **Reinforcement Learning (RL)** - the robot learns by trying different actions and seeing what works!

### Key Concept: Learning by Experience

```
Traditional Programming:        Reinforcement Learning:
────────────────────────       ──────────────────────────
IF floor is dirty              Robot tries random actions
  THEN clean it                Robot receives rewards/penalties
IF wall ahead                  Robot learns which actions
  THEN turn                    give best rewards over time
```

### What Makes This "Pure" RL?

- ❌ No hardcoded rules like "if dirty, then clean"
- ❌ No pre-programmed navigation paths
- ✅ Robot discovers EVERYTHING through trial and error
- ✅ Uses only rewards to guide learning

---

## 🧠 How Reinforcement Learning Works

### The Basic Cycle

Reinforcement Learning follows a simple cycle that repeats thousands of times:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE RL LEARNING CYCLE                        │
└─────────────────────────────────────────────────────────────────┘

       ┌──────────┐                         ┌─────────────────┐
       │          │  (1) Observes State     │                 │
       │  AGENT   │◄────────────────────────│   ENVIRONMENT   │
       │ (Robot)  │                         │    (House)      │
       │          │  (2) Takes Action       │                 │
       │          │────────────────────────►│                 │
       │          │                         │                 │
       │          │  (3) Receives Reward    │                 │
       │          │◄────────────────────────│                 │
       └──────────┘  + New State            └─────────────────┘
            │
            ▼
     (4) Updates Q-Table
     (Remembers what worked)
```

### What Each Term Means

| Term            | What It Is                         | Example in This Project                       |
| --------------- | ---------------------------------- | --------------------------------------------- |
| **Agent**       | The learner/decision-maker         | Our cleaning robot                            |
| **Environment** | The world the agent interacts with | The house with rooms                          |
| **State**       | Current situation of the agent     | Robot's position + came from direction        |
| **Action**      | What the agent can do              | Move, Clean, Wait                             |
| **Reward**      | Feedback (positive or negative)    | +50 for cleaning kitchen, -5 for hitting wall |
| **Policy**      | The learned strategy               | Which action to take in each state            |
| **Q-Table**     | Memory of learned values           | Table storing "how good" each action is       |

---

## 📁 Project Architecture

```
cleaning-robot-rl/
│
├── 📄 main.py                  # 🚀 START HERE - Interactive menu
│
├── 📁 env/                     # The House (Environment)
│   ├── __init__.py
│   └── cleaning_env.py         # Custom Gymnasium environment
│
├── 📁 agent/                   # The Robot (Agent)
│   ├── __init__.py
│   └── q_learning_agent.py     # Q-Learning algorithm
│
├── 📁 utils/                   # Helper Tools
│   ├── __init__.py
│   ├── helpers.py              # Utility functions
│   └── plotting.py             # Visualization graphs
│
├── 📁 models/                  # Saved Q-Tables
│   └── q_table.pkl             # Trained robot's "brain"
│
├── 📁 plots/                   # Training graphs
│
├── 📄 train.py                 # Direct training script
├── 📄 test.py                  # Direct testing script
├── 📄 requirements.txt         # Required Python packages
└── 📄 README.md                # This file
```

---

## 🔄 Complete Code Workflow

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE PROJECT WORKFLOW                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  1. TRAINING   │────►│  2. SAVE       │────►│  3. TESTING    │
│                │     │                │     │                │
│  Robot learns  │     │  Save Q-table  │     │  Robot shows   │
│  by playing    │     │  to file       │     │  learned       │
│  3000+ games   │     │  (models/)     │     │  behavior      │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
        │                                              │
        │         ┌────────────────────────┐          │
        └────────►│  4. VISUALIZATION      │◄─────────┘
                  │                        │
                  │  Pygame window shows   │
                  │  robot cleaning in     │
                  │  real-time             │
                  │                        │
                  └────────────────────────┘
```

### Detailed Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DETAILED EXECUTION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

User runs: python main.py
            │
            ▼
┌─────────────────────────────────────────┐
│         MAIN MENU DISPLAYED             │
│                                         │
│   1. Train Agent                        │
│   2. Test Agent                         │
│   3. Watch Agent (Visual Demo)          │
│   4. Quick Demo                         │
│   5. Exit                               │
└─────────────────────────────────────────┘
            │
            │ User selects option
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │  TRAIN      │  │  TEST       │  │  WATCH      │  │  QUICK DEMO         ││
│  │             │  │             │  │             │  │                     ││
│  │ Creates new │  │ Loads saved │  │ Loads saved │  │ Short training +   ││
│  │ Q-table     │  │ Q-table     │  │ Q-table     │  │ visual test         ││
│  │             │  │             │  │             │  │                     ││
│  │ Runs 3000   │  │ Runs test   │  │ Shows      │  │ Great for first-   ││
│  │ episodes    │  │ episodes    │  │ Pygame UI   │  │ time users          ││
│  │             │  │             │  │             │  │                     ││
│  │ Saves to    │  │ Compares to │  │ Robot      │  │                     ││
│  │ models/     │  │ random      │  │ cleans     │  │                     ││
│  │ q_table.pkl │  │ baseline    │  │ visually    │  │                     ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Step-by-Step Flow Diagrams

### 1. Training Phase - How the Robot Learns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING FLOW DIAGRAM                               │
└─────────────────────────────────────────────────────────────────────────────┘

START TRAINING
      │
      ▼
┌─────────────────────┐
│ Initialize Q-Table  │
│ (all values = 0)    │
│                     │
│ 230 states ×        │
│ 6 actions =         │
│ 1,380 values        │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ FOR episode = 1     │◄──────────────────────────────────────────┐
│ to 3000:            │                                           │
└─────────────────────┘                                           │
      │                                                           │
      ▼                                                           │
┌─────────────────────┐                                           │
│ Reset Environment   │                                           │
│ • Robot at start    │                                           │
│ • All tiles dirty   │                                           │
│ • Steps = 0         │                                           │
└─────────────────────┘                                           │
      │                                                           │
      ▼                                                           │
┌─────────────────────┐                                           │
│ Get current state   │◄──────────────────────────────────┐      │
│ (position + dir)    │                                   │      │
└─────────────────────┘                                   │      │
      │                                                   │      │
      ▼                                                   │      │
┌─────────────────────────────────────────────────────┐  │      │
│           EPSILON-GREEDY ACTION SELECTION           │  │      │
│                                                     │  │      │
│   Generate random number (0 to 1)                   │  │      │
│              │                                      │  │      │
│   ┌──────────┴──────────┐                          │  │      │
│   │                     │                          │  │      │
│   ▼                     ▼                          │  │      │
│ If random < ε        If random ≥ ε                 │  │      │
│ (exploration)        (exploitation)                │  │      │
│   │                     │                          │  │      │
│   ▼                     ▼                          │  │      │
│ Choose RANDOM       Choose BEST action             │  │      │
│ action (0-5)        from Q-table                   │  │      │
│                     (highest Q-value)              │  │      │
└─────────────────────────────────────────────────────┘  │      │
      │                                                   │      │
      ▼                                                   │      │
┌─────────────────────┐                                   │      │
│ Execute action in   │                                   │      │
│ environment         │                                   │      │
│ • Robot moves/cleans│                                   │      │
│ • Get new state     │                                   │      │
│ • Get reward        │                                   │      │
└─────────────────────┘                                   │      │
      │                                                   │      │
      ▼                                                   │      │
┌─────────────────────────────────────────────────────┐  │      │
│              Q-TABLE UPDATE (LEARNING)              │  │      │
│                                                     │  │      │
│  Q(state, action) ← Q(state, action) +              │  │      │
│                     α × [reward +                    │  │      │
│                          γ × max(Q(next_state)) -   │  │      │
│                          Q(state, action)]          │  │      │
│                                                     │  │      │
│  Where:                                             │  │      │
│  • α (alpha) = 0.1 (learning rate)                  │  │      │
│  • γ (gamma) = 0.99 (discount factor)               │  │      │
└─────────────────────────────────────────────────────┘  │      │
      │                                                   │      │
      ▼                                                   │      │
┌─────────────────────┐    NO     ┌─────────────────┐    │      │
│ Episode done?       │──────────►│ Continue episode│────┘      │
│ (all clean OR       │           │ (more steps)    │           │
│  max steps)         │           └─────────────────┘           │
└─────────────────────┘                                         │
      │ YES                                                      │
      ▼                                                          │
┌─────────────────────┐                                         │
│ Decay epsilon       │                                         │
│ ε = ε × 0.9995      │                                         │
│ (less random over   │                                         │
│  time)              │                                         │
└─────────────────────┘                                         │
      │                                                          │
      ▼                                                          │
┌─────────────────────┐    NO                                   │
│ All episodes done?  │─────────────────────────────────────────┘
│ (episode >= 3000)   │
└─────────────────────┘
      │ YES
      ▼
┌─────────────────────┐
│ SAVE Q-TABLE        │
│ to models/          │
│ q_table.pkl         │
└─────────────────────┘
      │
      ▼
   END TRAINING
```

### 2. Testing Phase - Robot Shows What It Learned

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TESTING FLOW DIAGRAM                               │
└─────────────────────────────────────────────────────────────────────────────┘

START TESTING
      │
      ▼
┌─────────────────────┐
│ Load saved Q-Table  │
│ from models/        │
│ q_table.pkl         │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Create environment  │
│ with visual mode    │
│ (render_mode=       │
│  'human')           │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ FOR test_episode    │◄──────────────────────────────────┐
│ = 1 to 5:           │                                   │
└─────────────────────┘                                   │
      │                                                   │
      ▼                                                   │
┌─────────────────────┐                                   │
│ Reset Environment   │                                   │
│ • Show Pygame window│                                   │
│ • Robot at start    │                                   │
└─────────────────────┘                                   │
      │                                                   │
      ▼                                                   │
┌─────────────────────┐                                   │
│ Get current state   │◄────────────────────────┐        │
└─────────────────────┘                         │        │
      │                                         │        │
      ▼                                         │        │
┌─────────────────────────────────────────────┐│        │
│        SOFTMAX ACTION SELECTION             ││        │
│        (testing mode - no randomness)       ││        │
│                                             ││        │
│  Convert Q-values to probabilities:         ││        │
│  P(action) = exp(Q/temp) / Σexp(Q/temp)     ││        │
│                                             ││        │
│  Sample action from probability             ││        │
│  (mostly picks best, but with variety)      ││        │
└─────────────────────────────────────────────┘│        │
      │                                         │        │
      ▼                                         │        │
┌─────────────────────┐                         │        │
│ Execute action      │                         │        │
│ • Robot moves       │                         │        │
│ • Render frame      │                         │        │
│ • Small delay       │                         │        │
│   (for visibility)  │                         │        │
└─────────────────────┘                         │        │
      │                                         │        │
      ▼                                         │        │
┌─────────────────────┐   NO                   │        │
│ Episode done?       │────────────────────────┘        │
└─────────────────────┘                                  │
      │ YES                                              │
      ▼                                                  │
┌─────────────────────┐                                  │
│ Record results:     │                                  │
│ • Tiles cleaned     │                                  │
│ • Total reward      │                                  │
│ • Steps taken       │                                  │
└─────────────────────┘                                  │
      │                                                  │
      ▼                                                  │
┌─────────────────────┐    NO                           │
│ All tests done?     │────────────────────────────────┘
└─────────────────────┘
      │ YES
      ▼
┌─────────────────────┐
│ Print Summary:      │
│ • Average tiles     │
│ • Coverage %        │
│ • Compare to random │
└─────────────────────┘
      │
      ▼
   END TESTING
```

### 3. Single Episode Flow (Detailed)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SINGLE EPISODE - STEP BY STEP                           │
└─────────────────────────────────────────────────────────────────────────────┘

EPISODE START
     │
     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  INITIAL STATE                                                            │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                               │
│  │WALL│WALL│WALL│WALL│WALL│WALL│WALL│WALL│  Row 0                        │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤                               │
│  │WALL│ 🟫 │ 🟫 │ 🟫 │WALL│ 🟫 │ 🟫 │WALL│  Row 1  🟫 = Dirty tile      │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤                               │
│  │WALL│ 🟫 │ 🤖 │ 🟫 │ 🟫 │ 🟫 │ 🟫 │WALL│  Row 2  🤖 = Robot           │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤                               │
│  │WALL│ 🟫 │ 🟫 │ 🟫 │ 🟫 │ 🟫 │ 🟫 │WALL│  Row 3                        │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤                               │
│  │WALL│ 🟫 │ 🟫 │ 🟫 │ 🟫 │ 🟫 │ 🟫 │WALL│  Row 4                        │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤                               │
│  │WALL│WALL│WALL│WALL│WALL│WALL│WALL│WALL│  Row 5                        │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                               │
│  Robot position: (2,2), Direction: None, Tiles dirty: 23                  │
└──────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STATE ENCODING                                                           │
│                                                                           │
│  State = position_index + (is_current_dirty × 23) + (came_from × 46)     │
│                                                                           │
│  position_index = 7 (robot's position on cleanable tiles)                │
│  is_current_dirty = 1 (yes, current tile is dirty)                       │
│  came_from = 4 (none - just started)                                     │
│                                                                           │
│  State = 7 + (1 × 23) + (4 × 46) = 7 + 23 + 184 = 214                   │
│                                                                           │
│  Total possible states: 23 × 2 × 5 = 230                                 │
└──────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Robot CLEANS current tile                                      │
│                                                                          │
│  Action chosen: 5 (Clean)                                               │
│  Result: Tile (2,2) is now clean!                                       │
│  Reward: +50 (Kitchen tile cleaned)                                     │
│                                                                          │
│  Q-table update:                                                        │
│  Q[214, 5] = Q[214, 5] + 0.1 × (50 + 0.99×max(Q[new_state]) - Q[214,5])│
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Robot moves RIGHT                                              │
│                                                                          │
│  Action chosen: 3 (Move Right)                                          │
│  Robot moves from (2,2) to (2,3)                                        │
│  Reward: -0.1 (small step penalty)                                      │
│  New direction: came_from = WEST                                        │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
     ... (continues for up to 300 steps)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  EPISODE END                                                             │
│                                                                          │
│  Ended because:                                                         │
│  □ All 23 tiles cleaned (SUCCESS!)  OR                                  │
│  □ Max 300 steps reached (TIMEOUT)                                      │
│                                                                          │
│  Results:                                                               │
│  • Tiles cleaned: 19/23                                                 │
│  • Total reward: 850.5                                                  │
│  • Steps taken: 245                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📖 Understanding the Q-Learning Algorithm

### The Q-Table Explained

The Q-Table is like the robot's memory - it remembers how good each action is in each situation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Q-TABLE STRUCTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

                     ACTIONS
              ┌──────┬──────┬──────┬──────┬──────┬──────┐
              │  ↑   │  ↓   │  ←   │  →   │ WAIT │CLEAN │
              │Forwd │Backwd│ Left │Right │      │      │
      ┌───────┼──────┼──────┼──────┼──────┼──────┼──────┤
      │State 0│ -2.3 │ 1.5  │ -5.0 │ 3.2  │ -3.0 │ 45.2 │ ◄─ Best action: CLEAN
S     ├───────┼──────┼──────┼──────┼──────┼──────┼──────┤
T     │State 1│ 5.1  │ 2.3  │ 4.8  │ -1.2 │ -3.0 │ 0.5  │ ◄─ Best action: Forward
A     ├───────┼──────┼──────┼──────┼──────┼──────┼──────┤
T     │State 2│ 2.1  │ 6.7  │ 3.3  │ 4.5  │ -3.0 │ 38.1 │ ◄─ Best action: CLEAN
E     ├───────┼──────┼──────┼──────┼──────┼──────┼──────┤
S     │ ...   │ ...  │ ...  │ ...  │ ...  │ ...  │ ...  │
      ├───────┼──────┼──────┼──────┼──────┼──────┼──────┤
      │St. 229│ 1.2  │ 3.4  │ 5.6  │ 7.8  │ -3.0 │ 2.1  │ ◄─ Best action: Right
      └───────┴──────┴──────┴──────┴──────┴──────┴──────┘

      Total: 230 states × 6 actions = 1,380 Q-values to learn
```

### The Bellman Equation (Learning Rule)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BELLMAN EQUATION EXPLAINED                          │
└─────────────────────────────────────────────────────────────────────────────┘

Q(s, a) ← Q(s, a) + α × [r + γ × max(Q(s', a')) - Q(s, a)]
           │          │   │   │   │                  │
           │          │   │   │   │                  └── Current estimate
           │          │   │   │   │
           │          │   │   │   └── Best future value
           │          │   │   │
           │          │   │   └── Discount (γ = 0.99)
           │          │   │       "How much future matters"
           │          │   │
           │          │   └── Immediate reward
           │          │       "+50 for cleaning"
           │          │
           │          └── Learning rate (α = 0.1)
           │              "How fast to update"
           │
           └── Old Q-value


SIMPLIFIED EXPLANATION:

  New Value = Old Value + Small Step Toward (What We Really Got)

  Where "What We Really Got" =
    Immediate Reward + Discounted Future Reward - What We Thought We'd Get


EXAMPLE:

  Robot is at State 50, takes action CLEAN:
  - Current Q[50, CLEAN] = 10.0 (old estimate)
  - Got reward: +50 (cleaned kitchen)
  - New state: 51
  - Best Q-value in state 51: max(Q[51, :]) = 8.0

  Update:
  Q[50, CLEAN] = 10.0 + 0.1 × [50 + 0.99×8.0 - 10.0]
               = 10.0 + 0.1 × [50 + 7.92 - 10.0]
               = 10.0 + 0.1 × 47.92
               = 10.0 + 4.792
               = 14.792

  The Q-value increased because cleaning gave a good reward!
```

### Epsilon-Greedy Exploration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPLORATION VS EXPLOITATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

Episode 1 (ε = 1.0 = 100% random):
┌────────────────────────────────────────────────────────────┐
│████████████████████████████████████████████████████████████│ 100% Explore
└────────────────────────────────────────────────────────────┘
Robot tries RANDOM actions - exploring the environment

Episode 500 (ε = 0.60):
┌────────────────────────────────────────────────────────────┐
│████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░│ 60% Explore
└────────────────────────────────────────────────────────────┘
Robot mixes random and learned behavior

Episode 1500 (ε = 0.30):
┌────────────────────────────────────────────────────────────┐
│██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 30% Explore
└────────────────────────────────────────────────────────────┘
Robot mostly uses learned behavior, sometimes explores

Episode 3000 (ε = 0.02 = 2% random):
┌────────────────────────────────────────────────────────────┐
│█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 2% Explore
└────────────────────────────────────────────────────────────┘
Robot almost always uses best known actions

█ = Random (Exploration)
░ = Best Action (Exploitation)
```

---

## 🏠 The House Environment

### Room Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HOUSE LAYOUT (8×6 Grid)                          │
└─────────────────────────────────────────────────────────────────────────────┘

         Column: 0    1    2    3    4    5    6    7
               ┌────┬────┬────┬────┬────┬────┬────┬────┐
      Row 0:   │████│████│████│████│████│████│████│████│  █ = WALL
               ├────┼────┼────┼────┼────┼────┼────┼────┤
      Row 1:   │████│🟨  │🟨  │🟨  │████│🟦  │🟦  │████│  🟨 = KITCHEN
               ├────┼────┼────┼────┼────┼────┼────┼────┤
      Row 2:   │████│🟨  │🟨  │🟨  │⬜  │🟦  │🟦  │████│  🟦 = LIVING ROOM
               ├────┼────┼────┼────┼────┼────┼────┼────┤
      Row 3:   │████│🟨  │🟨  │🟨  │⬜  │🟦  │🟦  │████│  ⬜ = HALLWAY
               ├────┼────┼────┼────┼────┼────┼────┼────┤
      Row 4:   │████│⬜  │⬜  │⬜  │⬜  │⬜  │⬜  │████│
               ├────┼────┼────┼────┼────┼────┼────┼────┤
      Row 5:   │████│████│████│████│████│████│████│████│
               └────┴────┴────┴────┴────┴────┴────┴────┘


ROOM STATISTICS:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│    Room     │   Tiles     │   Reward    │   Color     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│   Kitchen   │  9 tiles    │  +50 each   │   Yellow    │
│ Living Room │  6 tiles    │  +35 each   │    Blue     │
│   Hallway   │  8 tiles    │  +20 each   │    Gray     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│   TOTAL     │  23 tiles   │  +200 bonus │ (if all)    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### Reward Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REWARD STRUCTURE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

POSITIVE REWARDS (Good actions):
┌────────────────────────────────────┬───────────┐
│           Action                   │  Reward   │
├────────────────────────────────────┼───────────┤
│ Clean dirty KITCHEN tile           │   +50     │
│ Clean dirty LIVING ROOM tile       │   +35     │
│ Clean dirty HALLWAY tile           │   +20     │
│ Clean ALL 23 tiles (bonus)         │  +200     │
└────────────────────────────────────┴───────────┘

NEGATIVE REWARDS (Penalties):
┌────────────────────────────────────┬───────────┐
│           Action                   │  Reward   │
├────────────────────────────────────┼───────────┤
│ Clean already clean tile           │   -10     │
│ Hit a wall (invalid move)          │    -5     │
│ Wait action (do nothing)           │    -3     │
│ Each step taken                    │   -0.1    │
└────────────────────────────────────┴───────────┘

WHY THESE REWARDS?
• Kitchen has highest reward → Robot learns to prioritize it
• Step penalty (-0.1) → Robot learns to be efficient
• Wall penalty (-5) → Robot learns to avoid walls
• Wait penalty (-3) → Robot learns not to idle
```

---

## 📂 File-by-File Explanation

### 1. `main.py` - The Entry Point

**Purpose:** Interactive menu to train, test, or watch the robot.

**Key Functions:**

```python
def main():
    """Display menu and handle user choices"""

def train_agent():
    """Train a new Q-Learning agent"""

def test_agent():
    """Test trained agent vs random baseline"""

def watch_agent():
    """Visual demo with Pygame rendering"""
```

**Flow:**

```
User runs main.py
      │
      ▼
Display menu → Get choice → Execute function → Repeat
```

---

### 2. `env/cleaning_env.py` - The House (Environment)

**Purpose:** Defines the house, rooms, and rules of the game.

**Key Components:**

```python
class CleaningEnv(gym.Env):
    """Custom Gymnasium environment"""

    def __init__(self, render_mode=None):
        """Initialize house, rooms, robot position"""

    def reset(self):
        """Reset to initial state (all tiles dirty)"""

    def step(self, action):
        """Execute action, return (next_state, reward, done, info)"""

    def _get_state(self):
        """Encode position + dirt + direction into single integer"""

    def render(self):
        """Draw house using Pygame"""
```

**State Encoding:**

```
State = position_index + (is_dirty × 23) + (came_from × 46)

Where:
• position_index: 0-22 (which tile robot is on)
• is_dirty: 0 or 1 (is current tile dirty?)
• came_from: 0-4 (North, South, East, West, None)

Total states: 23 × 2 × 5 = 230 states
```

---

### 3. `agent/q_learning_agent.py` - The Robot (Agent)

**Purpose:** Implements Q-Learning algorithm.

**Key Components:**

```python
class QLearningAgent:
    """Q-Learning agent with epsilon-greedy policy"""

    def __init__(self, state_size, action_size, ...):
        """Initialize Q-table with zeros"""
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, training=True):
        """
        Training: Epsilon-greedy (random vs best)
        Testing: Softmax (probabilistic selection)
        """

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Bellman equation"""

    def decay_epsilon(self):
        """Reduce exploration over time"""

    def save(self, filepath):
        """Save Q-table to file"""

    def load(self, filepath):
        """Load Q-table from file"""
```

---

### 4. `train.py` - Training Script

**Purpose:** Direct training without the menu.

**Usage:**

```bash
python train.py
```

**What it does:**

1. Creates environment and agent
2. Runs training loop for 3000+ episodes
3. Saves Q-table to `models/q_table.pkl`
4. Generates training plots

---

### 5. `test.py` - Testing Script

**Purpose:** Test trained agent and compare to random.

**Usage:**

```bash
python test.py
```

**What it does:**

1. Loads saved Q-table
2. Runs test episodes (visual)
3. Runs random baseline
4. Compares and reports results

---

### 6. `utils/helpers.py` & `utils/plotting.py`

**Purpose:** Utility functions for time formatting, printing, and visualization.

**Key Functions:**

```python
# helpers.py
format_time()      # Pretty print timestamps
print_header()     # Display section headers
Timer class        # Measure training time

# plotting.py
plot_training_results()  # Graph tiles cleaned over episodes
plot_comparison()        # Compare trained vs random agent
```

---

## 🚀 How to Run

### Prerequisites

```bash
# Install required packages
pip install gymnasium numpy pygame matplotlib
```

### Option 1: Interactive Menu (Recommended)

```bash
cd cleaning-robot-rl
python main.py
```

Then select from:

- **1. Train Agent** - Train new model (takes 5-10 minutes)
- **2. Test Agent** - Test saved model
- **3. Watch Agent** - Visual demo
- **4. Quick Demo** - Short demo for beginners

### Option 2: Direct Scripts

```bash
# Train the robot
python train.py

# Test the trained robot
python test.py
```

---

## 📈 Expected Results

### Training Progress

| Episodes | Average Tiles Cleaned | Epsilon |
| -------- | --------------------- | ------- |
| 500      | ~17-18 / 23           | 0.78    |
| 1000     | ~18-19 / 23           | 0.61    |
| 1500     | ~19-20 / 23           | 0.47    |
| 2000     | ~20-21 / 23           | 0.37    |
| 3000     | ~21-22 / 23           | 0.22    |

### Test Results

```
┌─────────────────────────────────────────┐
│          EXPECTED TEST RESULTS          │
├─────────────────────────────────────────┤
│  Trained Agent:   19-22 tiles (82-95%)  │
│  Random Agent:    5-8 tiles  (22-35%)   │
│                                         │
│  Improvement:     ~3x better!           │
└─────────────────────────────────────────┘
```

---

## ❓ Common Questions

### Q: Why doesn't the robot clean all 23 tiles every time?

**A:** Several factors affect this:

- 300 step limit prevents infinite cleaning
- Some states have suboptimal learned values
- Softmax testing adds slight randomness to prevent loops
- More training improves completion rate

### Q: What is epsilon (ε)?

**A:** Epsilon controls exploration:

- ε = 1.0 → 100% random actions (full exploration)
- ε = 0.0 → 0% random actions (full exploitation)
- We decay ε over time so the robot explores early and exploits later

### Q: Why use softmax during testing instead of greedy?

**A:** Pure greedy (always pick max Q-value) can cause the robot to get stuck in loops. Softmax adds small randomness to break out of repeating patterns.

### Q: How long does training take?

**A:**

- Quick demo: 1-2 minutes (500 episodes)
- Full training: 5-10 minutes (3000 episodes)
- Extended training: 15-20 minutes (10000 episodes)

### Q: Can I modify the reward structure?

**A:** Yes! In `env/cleaning_env.py`, look for `REWARD_*` constants:

```python
REWARD_CLEAN_KITCHEN = 50     # Modify these
REWARD_CLEAN_LIVING = 35
REWARD_CLEAN_HALLWAY = 20
```

### Q: Why 230 states?

**A:** State = position × dirt × direction

- 23 positions (cleanable tiles)
- 2 dirt states (clean or dirty)
- 5 directions (N, S, E, W, None)
- 23 × 2 × 5 = 230 total states

---

## 🎓 Learning Outcomes

By studying this project, you will understand:

1. **Reinforcement Learning Basics** - Agent, Environment, State, Action, Reward
2. **Q-Learning Algorithm** - Q-table, Bellman equation, TD learning
3. **Exploration vs Exploitation** - Why and how epsilon-greedy works
4. **Custom Environments** - How to create Gymnasium environments
5. **State Representation** - Encoding complex information into states
6. **Reward Shaping** - How rewards guide learning behavior
7. **Visualization** - Using Pygame for interactive display

---

## 📜 License

This project is for educational purposes. Feel free to use, modify, and learn from it!

---

<div align="center">

**Made with ❤️ for learning Reinforcement Learning**

_If you found this helpful, give it a ⭐!_

</div>
