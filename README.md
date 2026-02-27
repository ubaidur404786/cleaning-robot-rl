<p align="center">
  <img src="https://img.shields.io/badge/🤖-Cleaning%20Robot-brightgreen?style=for-the-badge&labelColor=black" alt="Cleaning Robot"/>
</p>

<h1 align="center">🧹 Cleaning Robot using Reinforcement Learning</h1>

<p align="center">
  <b>Watch an AI robot learn to clean a house — from scratch, with zero instructions!</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gymnasium-0.29+-00C853?style=flat-square"/>
  <img src="https://img.shields.io/badge/Pygame-2.0+-DD2C00?style=flat-square&logo=pygame"/>
  <img src="https://img.shields.io/badge/NumPy-1.20+-013243?style=flat-square&logo=numpy"/>
  <img src="https://img.shields.io/badge/License-Educational-blue?style=flat-square"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Q--Learning-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Type-Pure%20RL-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Beginner-Friendly-success?style=for-the-badge"/>
</p>

---

<h2 align="center">🎬 What You'll See</h2>

<table align="center">
<tr>
<td align="center" width="33%">
<h3>🏠 Before Training</h3>
<p>Robot moves randomly<br/>Cleans ~5 tiles out of 23<br/>No strategy at all</p>
</td>
<td align="center" width="33%">
<h3>⚡ Training</h3>
<p>3000 episodes<br/>Robot learns patterns<br/>Q-table fills up</p>
</td>
<td align="center" width="33%">
<h3>✨ After Training</h3>
<p>Smart navigation<br/>Cleans 19-22 tiles!<br/>~85% coverage</p>
</td>
</tr>
</table>

---

## 🌟 Why This Project?

<table>
<tr>
<td width="50%">

### 📚 Perfect for Learning

- **No ML background needed** — concepts explained from scratch
- **Visual feedback** — watch the robot learn in real-time
- **Well-commented code** — understand every line
- **Interactive demo** — try it yourself in minutes

</td>
<td width="50%">

### 🚀 What You'll Master

- ✅ Reinforcement Learning fundamentals
- ✅ Q-Learning algorithm
- ✅ Custom Gymnasium environments
- ✅ State & reward design
- ✅ Pygame visualization

</td>
</tr>
</table>

---

## 🧠 The Big Idea: Teaching a Robot Without Instructions

<table>
<tr>
<td width="50%">

### ❌ Traditional Programming

```python
# We write EVERY rule
if floor.is_dirty():
    robot.clean()
if wall_ahead():
    robot.turn_right()
if kitchen_done():
    robot.go_to_living_room()
```

**Problem:** What if the house layout changes? We rewrite everything!

</td>
<td width="50%">

### ✅ Reinforcement Learning

```python
# Robot figures it out!
robot.try_action()
reward = environment.feedback()
robot.remember(action, reward)
# After 3000 tries...
robot.knows_what_works()
```

**Magic:** Robot learns optimal strategy through experience!

</td>
</tr>
</table>

---

## 🎓 Core Concepts Explained

### 1️⃣ What is Reinforcement Learning?

Think of training a puppy 🐕:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   PUPPY TRAINING          =       ROBOT TRAINING            │
│                                                             │
│   🐕 Puppy                 →       🤖 Robot (Agent)          │
│   🏠 House                 →       🏠 Virtual House          │
│   🍖 Treat for good        →       +50 points for cleaning  │
│   📣 "No!" for bad         →       -5 points for hitting    │
│   🧠 Learns over time      →       🧠 Q-Table learns         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> **Just like a puppy learns to sit for treats, our robot learns to clean for rewards!**

---

### 2️⃣ The Learning Loop (Happens 300,000+ Times!)

```
        ╔═══════════════════════════════════════════════════════╗
        ║              THE REINFORCEMENT LEARNING LOOP          ║
        ╚═══════════════════════════════════════════════════════╝

                    ┌─────────────────┐
                    │   ENVIRONMENT   │
                    │    (House)      │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                │                ▼
     ┌──────────┐            │         ┌──────────┐
     │  STATE   │            │         │  REWARD  │
     │Position: │            │         │ +50 👍   │
     │ (2,3)    │            │         │ -5  👎   │
     └────┬─────┘            │         └────┬─────┘
          │                  │              │
          │    ┌─────────────┴─────────┐   │
          └───►│        AGENT          │◄──┘
               │       (Robot)         │
               │                       │
               │  "What should I do?"  │
               └───────────┬───────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    ACTION    │
                    │ Move / Clean │
                    └──────────────┘
                           │
                           ▼
                  Back to Environment
                    (Loop repeats)
```

---

### 3️⃣ Q-Learning: The Robot's Memory

The **Q-Table** is like a cheat sheet the robot builds:

```
╔════════════════════════════════════════════════════════════════════╗
║                        Q-TABLE (The Brain)                         ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  "If I'm at position X, which action gives the best reward?"       ║
║                                                                    ║
║  ┌─────────┬────────┬────────┬────────┬────────┬────────┬────────┐║
║  │  State  │   ↑    │   ↓    │   ←    │   →    │  Wait  │ Clean  │║
║  │         │Forward │Backward│  Left  │ Right  │        │        │║
║  ├─────────┼────────┼────────┼────────┼────────┼────────┼────────┤║
║  │Kitchen_1│  2.3   │  1.5   │ -3.2   │  4.1   │ -3.0   │ ⭐45.2 │║
║  │Kitchen_2│  5.1   │ ⭐8.3  │  2.1   │  1.2   │ -3.0   │  2.5   │║
║  │Hallway_1│ ⭐6.7  │  2.3   │  3.3   │  4.5   │ -3.0   │ 38.1   │║
║  │   ...   │  ...   │  ...   │  ...   │  ...   │  ...   │  ...   │║
║  └─────────┴────────┴────────┴────────┴────────┴────────┴────────┘║
║                                                                    ║
║  ⭐ = Best action for that state (robot picks this!)              ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

---

### 4️⃣ Exploration vs Exploitation

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   Episode 1 (Epsilon = 1.0)                                       │
│   ████████████████████████████████████████ 100% RANDOM            │
│   "I know nothing! Let me try everything!"                        │
│                                                                    │
│   Episode 1000 (Epsilon = 0.5)                                    │
│   ████████████████████░░░░░░░░░░░░░░░░░░░░ 50% EACH               │
│   "I know some things, but still exploring..."                    │
│                                                                    │
│   Episode 3000 (Epsilon = 0.02)                                   │
│   █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 2% random              │
│   "I'm confident! Using my learned strategy."                     │
│                                                                    │
│   ████ = Random/Exploring    ░░░░ = Using Best Known Action       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🏠 The Virtual House

```
        ┌─────────────────────────────────────────────────────────┐
        │                    HOUSE LAYOUT (8×6)                   │
        └─────────────────────────────────────────────────────────┘

           0     1     2     3     4     5     6     7
        ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      0 │█████│█████│█████│█████│█████│█████│█████│█████│  ← Walls
        ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
      1 │█████│ 🟨  │ 🟨  │ 🟨  │█████│ 🟦  │ 🟦  │█████│
        ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
      2 │█████│ 🟨  │ 🤖  │ 🟨  │ ⬜  │ 🟦  │ 🟦  │█████│  🤖 Robot
        ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
      3 │█████│ 🟨  │ 🟨  │ 🟨  │ ⬜  │ 🟦  │ 🟦  │█████│
        ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
      4 │█████│ ⬜  │ ⬜  │ ⬜  │ ⬜  │ ⬜  │ ⬜  │█████│
        ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
      5 │█████│█████│█████│█████│█████│█████│█████│█████│
        └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

        🟨 KITCHEN (9 tiles)     → +50 points each
        🟦 LIVING ROOM (6 tiles) → +35 points each
        ⬜ HALLWAY (8 tiles)     → +20 points each
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📊 TOTAL: 23 cleanable tiles
```

---

## 🎮 Real-World Use Cases

<table>
<tr>
<td align="center" width="25%">
<h3>🏠 Smart Vacuums</h3>
<p>Roomba, Roborock use similar RL to navigate homes</p>
</td>
<td align="center" width="25%">
<h3>🏭 Warehouse Robots</h3>
<p>Amazon robots learn optimal picking routes</p>
</td>
<td align="center" width="25%">
<h3>🎮 Game AI</h3>
<p>NPCs that learn player patterns</p>
</td>
<td align="center" width="25%">
<h3>🚗 Self-Driving</h3>
<p>Cars learn traffic navigation</p>
</td>
</tr>
</table>

---

## 📈 Results You'll See

<table align="center">
<tr>
<th>Metric</th>
<th>Random Robot</th>
<th>Trained Robot</th>
<th>Improvement</th>
</tr>
<tr>
<td><b>Tiles Cleaned</b></td>
<td>5-8 / 23</td>
<td>19-22 / 23</td>
<td><b>~3x better!</b></td>
</tr>
<tr>
<td><b>Coverage</b></td>
<td>22-35%</td>
<td>82-95%</td>
<td><b>+60%</b></td>
</tr>
<tr>
<td><b>Efficiency</b></td>
<td>Chaotic</td>
<td>Systematic</td>
<td><b>🧠 Smart!</b></td>
</tr>
</table>

---

## 🚀 Quick Start (5 Minutes!)

### 1️⃣ Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/cleaning-robot-rl.git
cd cleaning-robot-rl
pip install -r requirements.txt
```

### 2️⃣ Run Interactive Demo

```bash
python main.py
```

### 3️⃣ Choose from Menu

```
╔═══════════════════════════════════════════════╗
║        CLEANING ROBOT RL - MAIN MENU          ║
╠═══════════════════════════════════════════════╣
║                                               ║
║   1. 🎯 Train Agent    (Train new robot)      ║
║   2. 🧪 Test Agent     (Test vs random)       ║
║   3. 👀 Watch Agent    (Visual demo)          ║
║   4. ⚡ Quick Demo     (Fast showcase)        ║
║   5. 🚪 Exit                                  ║
║                                               ║
╚═══════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
cleaning-robot-rl/
│
├── 📄 main.py                   # 🚀 Start here!
├── 📄 train.py                  # Direct training script
├── 📄 test.py                   # Direct testing script
│
├── 📂 env/                      # 🏠 The House
│   └── cleaning_env.py          # Custom Gymnasium environment
│
├── 📂 agent/                    # 🤖 The Robot
│   └── q_learning_agent.py      # Q-Learning implementation
│
├── 📂 utils/                    # 🔧 Helper Tools
│   ├── helpers.py               # Utility functions
│   └── plotting.py              # Training graphs
│
├── 📂 models/                   # 💾 Saved Q-Tables
├── 📂 plots/                    # 📊 Training visualizations
│
├── 📄 requirements.txt          # Dependencies
├── 📄 README.md                 # You are here!
└── 📄 explain_with_workflow.md  # 📖 DETAILED DOCUMENTATION
```

---

## 📖 Want to Understand EVERYTHING?

<table>
<tr>
<td>

### 📚 Deep Dive Available!

For a **complete, beginner-friendly explanation** with:

- 📊 Step-by-step flowcharts
- 🔢 Bellman equation breakdown with examples
- 📝 Line-by-line code explanation
- ❓ FAQ section
- 🎯 State encoding details

</td>
<td align="center">

### 👇 Read This File 👇

# [`explain_with_workflow.md`](explain_with_workflow.md)

**987 lines of pure documentation!**

</td>
</tr>
</table>

---

## 🎯 Learning Path

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUGGESTED LEARNING PATH                      │
└─────────────────────────────────────────────────────────────────┘

     START
       │
       ▼
  ┌─────────────┐
  │  Run Demo   │  ← python main.py → Option 4
  │   (5 min)   │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Read README │  ← You're here! Understand concepts
  │  (10 min)   │
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │  Deep Dive   │  ← Read explain_with_workflow.md
  │   (30 min)   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Explore Code │  ← Start with env/cleaning_env.py
  │   (1 hour)   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Modify!    │  ← Change rewards, grid size, etc.
  │  (∞ fun!)    │
  └──────────────┘
```

---

## 🛠️ Customize & Experiment

### Easy Modifications

| What to Change        | Where                 | Effect                  |
| --------------------- | --------------------- | ----------------------- |
| **Room rewards**      | `env/cleaning_env.py` | Change room priorities  |
| **Grid size**         | `env/cleaning_env.py` | Bigger/smaller house    |
| **Training episodes** | `train.py`            | More = better learning  |
| **Learning rate**     | `train.py`            | How fast robot learns   |
| **Exploration rate**  | `train.py`            | Random vs learned ratio |

### Try These Experiments

1. **Bigger house** — Does the robot still learn?
2. **Negative room rewards** — Can robot learn to avoid areas?
3. **No step penalty** — Does efficiency matter?
4. **Faster epsilon decay** — What happens with less exploration?

---

## 🙏 Acknowledgments

<table>
<tr>
<td width="70%">

### 💡 About This Project

This project was created for **educational purposes** to help beginners understand Reinforcement Learning through a practical, visual example.

### 🤖 AI Assistance Disclosure

> **Honesty Note:** Parts of this code were developed with assistance from AI tools (GitHub Copilot). The core concepts, architecture decisions, and learning objectives were human-directed, while AI helped with implementation details, debugging, and documentation.

This transparency is important because:

- Learning to work **with** AI is a valuable skill
- Understanding code matters more than writing from scratch
- The focus is on **learning RL concepts**, not coding speed

</td>
<td width="30%" align="center">

### 🔧 Built With

<br/>

**Python** 🐍
<br/><br/>
**Gymnasium** 🏋️
<br/><br/>
**Pygame** 🎮
<br/><br/>
**NumPy** 🔢
<br/><br/>
**Matplotlib** 📊

</td>
</tr>
</table>

---

## 🤝 Contributing

Found a bug? Have an idea? Want to add features?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

All contributions welcome — especially from beginners learning RL!

---

## 📜 License

This project is open source for **educational purposes**. Feel free to:

- ✅ Learn from it
- ✅ Modify it
- ✅ Use it in your portfolio
- ✅ Share it with others

---

<br/>

<h2 align="center">🌟 Star This Repo If It Helped You Learn! 🌟</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️%20and%20🤖-red?style=for-the-badge"/>
</p>

<p align="center">
  <b>Learning RL should be fun, visual, and accessible to everyone!</b>
</p>

<p align="center">
  <a href="explain_with_workflow.md">📖 Detailed Docs</a> •
  <a href="#-quick-start-5-minutes">🚀 Quick Start</a> •
  <a href="#-core-concepts-explained">🧠 Learn Concepts</a>
</p>

---

<p align="center">
  <i>"The best way to learn is by doing. The second best is by watching a robot do it!" 🤖</i>
</p>
