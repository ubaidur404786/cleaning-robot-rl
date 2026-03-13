# 🚀 QUICK START GUIDE - Training the Cleaning Robot RL

## Step 1: Verify System Setup

Before training, check that everything is configured correctly:

```bash
python verify_setup.py
```

**Output:** Shows:

- ✓ All directories present
- ✓ All files readable
- ✓ All imports working
- ✓ Existing models enumerated
- System READY FOR TRAINING status

---

## Step 2: Use the Interactive Menu (Recommended)

Launch the main application:

```bash
python main.py
```

From the menu you can now:

- Train one algorithm with live grid rendering every **N** episodes
- Test one trained algorithm with on-screen Pygame visualization
- Train **all 3** algorithms with shared episodes, then auto-test and auto-compare
- Train one algorithm and test it immediately
- Show optimal paths and generate comparison dashboards

### Recommended Menu Workflows

- **[11] Train All + Auto Test + Compare**  
  Best for full end-to-end evaluation in one run.
- **[12] Train One Then Test**  
  Best for iterative tuning of one algorithm.
- **[10] Compare All Algorithms (Dashboard)**  
  Generates detailed comparison plots and analysis.

---

## Step 3: Optional CLI Mode (`train_all_flexible.py`)

If you prefer non-menu command-line runs, use the options below.

### 🏃 QUICK TEST (Fast Learning - 500 Episodes)

```bash
python train_all_flexible.py --quick --all
```

**Time:** 15-30 minutes | **Output:** Quick baseline performance

---

### ⚖️ BALANCED TRAINING (Standard - 2000 Episodes)

```bash
python train_all_flexible.py --balanced --all
```

**Time:** 1-2 hours | **Output:** Good convergence for all algorithms

---

### 🎯 FULL PRODUCTION (Complete - 5000 Episodes)

```bash
python train_all_flexible.py --full --all
```

**Time:** 3-5 hours | **Output:** Optimal performance, ready for deployment

---

## Advanced CLI Options

### Train Specific Algorithms Only

```bash
# Q-Learning only (2000 episodes)
python train_all_flexible.py --episodes 2000 --ql

# SARSA only (2000 episodes)
python train_all_flexible.py --episodes 2000 --sarsa

# Q-Learning + SARSA (2000 episodes)
python train_all_flexible.py --episodes 2000 --ql --sarsa

# DQN only (500 episodes - faster)
python train_all_flexible.py --quick --dqn
```

### Custom Episode Count

```bash
# Train all algorithms with custom episode count
python train_all_flexible.py --episodes 3000 --all

# Train specific algorithms with custom episodes
python train_all_flexible.py --episodes 1000 --ql --sarsa
```

---

## 📊 Understanding Results

### Files Generated

After training, check:

```
models/
  └─ q_learning_agent_XXXX_TIMESTAMP.pkl
  └─ sarsa_agent_XXXX_TIMESTAMP.pkl
  └─ dqn_model_XXXX_TIMESTAMP.pth

results/
  └─ q_learning_metrics_XXXX_TIMESTAMP.json
  └─ sarsa_metrics_XXXX_TIMESTAMP.json
  └─ dqn_metrics_XXXX_TIMESTAMP.json

plots/
  └─ (Advanced plotting coming soon)
```

### Interpreting Metrics

Each JSON file contains per-episode tracking:

- `episode`: Episode number
- `total_reward`: Sum of rewards for that episode
- `episode_length`: Steps taken before episode ended
- `success`: 1 if all tiles cleaned, 0 otherwise
- `epsilon`: Exploration rate at episode end

---

## 🎯 Recommended Training Path

**For First-Time Testing:**

```bash
# 1. Verify system
python verify_setup.py

# 2. Launch menu
python main.py

# 3. In menu choose [11] Train All + Auto Test + Compare
```

**For Serious Performance Analysis:**

```bash
# 1. Verify system
python verify_setup.py

# 2. Launch menu
python main.py

# 3. Choose [11] for auto pipeline, or [10] to compare existing runs
```

**For Deployment/Publication:**

```bash
# 1. Verify system
python verify_setup.py

# 2. Launch menu
python main.py

# 3. Use [11] with higher episode count and rendering disabled (0)
```

---

## 🔍 Monitoring Progress

During training, you'll see periodic output:

```
================================================================================
  TRAINING Q-LEARNING AGENT (2000 episodes)
================================================================================
  Episode    500 | Reward:   25.43 | Success:  15.2%
  Episode   1000 | Reward:   48.72 | Success:  32.5%
  Episode   1500 | Reward:   72.15 | Success:  54.8%
  Episode   2000 | Reward:   95.33 | Success:  78.3%

 ✓ Q-Learning training complete (2000 episodes)
```

**What This Means:**

- **Reward**: Average episode reward (increasing = learning happening)
- **Success**: % of episodes that cleaned all tiles (increasing = convergence)

---

## ⚡ Algorithm Time Estimates

| Algorithm  | 500 Episodes | 2000 Episodes | 5000 Episodes |
| ---------- | ------------ | ------------- | ------------- |
| Q-Learning | ~5 min       | ~20 min       | ~50 min       |
| SARSA      | ~5 min       | ~20 min       | ~50 min       |
| DQN        | ~15 min      | ~60 min       | ~150 min      |
| **All 3**  | **~25 min**  | **~100 min**  | **~5 hours**  |

---

## 🐛 Troubleshooting

### "Module not found" errors

```bash
# Re-run verification
python verify_setup.py

# Install missing dependencies
pip install torch numpy matplotlib gymnasium pygame
```

### Training hangs on DQN startup

- Normal behavior: PyTorch initialization takes 10-30 seconds
- If it takes >2 minutes, use Ctrl+C and skip DQN:
  ```bash
  python train_all_flexible.py --balanced --ql --sarsa
  ```

### Out of memory with DQN

- Reduce episodes: `python train_all_flexible.py --episodes 500 --dqn`
- Or skip DQN entirely: `python train_all_flexible.py --balanced --ql --sarsa`

---

## 📖 Algorithm Descriptions

### Q-Learning (Off-Policy Tabular)

- **Speed**: Fast ⚡⚡⚡
- **Memory**: ~1MB (state-action lookup table)
- **Convergence**: Good (handles large state spaces via lazy init)
- **Best for**: Quick learning on modest hardware

### SARSA (On-Policy Tabular)

- **Speed**: Fast ⚡⚡⚡
- **Memory**: ~1MB (similar to Q-Learning)
- **Convergence**: Good (more conservative than Q-Learning)
- **Best for**: Safe learning environments

### DQN (Deep Q-Network)

- **Speed**: Slower ⚡⚡
- **Memory**: ~200MB (neural network + replay buffer)
- **Convergence**: Good (but slower than tabular)
- **Best for**: Complex problems, GPU acceleration

---

## 💾 Saving Your Trained Models

Models are automatically saved with timestamps:

```
models/q_learning_agent_2000eps_20240115_143022.pkl
```

To use a trained model:

```python
import pickle

# Load saved model
with open('models/q_learning_agent_2000eps_20240115_143022.pkl', 'rb') as f:
    agent = pickle.load(f)

# Use for inference
action = agent.choose_action(state, training=False)
```

---

## 📞 Support & Next Steps

After training:

1. Check `results/` directory for metrics JSON files
2. Review per-episode performance data
3. Watch for convergence patterns (reward increasing, success rate increasing)
4. Compare algorithms using saved metrics
5. Deploy best-performing model to production

Happy training! 🤖
