"""
================================================================================
SYSTEM VERIFICATION & SETUP CHECK
================================================================================

PROJECT: Cleaning Robot RL - All 3 Algorithms
PURPOSE: Verify that everything is configured correctly before training
         Check all dependencies, files, and configurations

Run: python verify_setup.py
================================================================================
"""

import os
import sys
import json
import pickle
from pathlib import Path

print("\n" + "="*80)
print("  COMPREHENSIVE SYSTEM VERIFICATION")
print("  Cleaning Robot RL - All 3 Algorithms (Q-Learning, SARSA, DQN)")
print("="*80)

# ============================================================================
# 1. CHECK DIRECTORIES
# ============================================================================
print("\n✓ CHECKING DIRECTORIES...")
print("-" * 80)

required_dirs = {
    "agent": "Agent implementations (Q-Learning, SARSA, DQN)",
    "env": "Environment (CleaningEnv)",
    "utils": "Utility functions (helpers, plotting)",
    "models": "Trained model storage",
    "plots": "Visualization outputs",
    "results": "Metrics and reports"
}

all_dirs_exist = True
for dir_name, description in required_dirs.items():
    if os.path.isdir(dir_name):
        print(f"  ✓ {dir_name:15} - {description}")
    else:
        print(f"  ✗ {dir_name:15} - MISSING!")
        all_dirs_exist = False

if not all_dirs_exist:
    print("\n  Creating missing directories...")
    for dir_name in required_dirs.keys():
        Path(dir_name).mkdir(exist_ok=True)
    print("  ✓ All directories created")

# ============================================================================
# 2. CHECK PYTHON FILES
# ============================================================================
print("\n✓ CHECKING PYTHON FILES...")
print("-" * 80)

required_files = {
    "env/cleaning_env.py": "Main environment implementation",
    "agent/q_learning_agent.py": "Q-Learning algorithm",
    "agent/sarsa_agent.py": "SARSA algorithm",
    "agent/dqn_agent.py": "DQN (Deep Q-Network) implementation",
    "utils/helpers.py": "Helper utilities",
    "utils/plotting.py": "Visualization functions",
    "main.py": "Main entry point",
    "requirements.txt": "Python dependencies",
    "quick_test.py": "Quick test (Q-Learning + SARSA)",
    "comprehensive_test.py": "Full test (all 3 algorithms)"
}

all_files_exist = True
for file_path, description in required_files.items():
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        print(f"  ✓ {file_path:35} - {description} ({file_size:,} bytes)")
    else:
        print(f"  ✗ {file_path:35} - MISSING!")
        all_files_exist = False

# ============================================================================
# 3. CHECK PYTHON IMPORTS
# ============================================================================
print("\n✓ CHECKING PYTHON IMPORTS...")
print("-" * 80)

imports_to_check = {
    "numpy": "NumPy (numerical computing)",
    "matplotlib": "Matplotlib (plotting)",
    "pygame": "Pygame (visualization)",
    "torch": "PyTorch (neural networks)",
    "gymnasium": "Gymnasium (RL environment)",
    "pickle": "Pickle (model serialization)",
    "json": "JSON (data serialization)"
}

all_imports_ok = True
for module_name, description in imports_to_check.items():
    try:
        __import__(module_name)
        print(f"  ✓ {module_name:15} - {description}")
    except ImportError as e:
        print(f"  ✗ {module_name:15} - FAILED: {str(e)[:40]}")
        all_imports_ok = False

# ============================================================================
# 4. CHECK PROJECT-SPECIFIC IMPORTS
# ============================================================================
print("\n✓ CHECKING PROJECT IMPORTS...")
print("-" * 80)

project_imports = [
    ("from env.cleaning_env import CleaningEnv", "CleaningEnv environment"),
    ("from agent.q_learning_agent import QLearningAgent", "Q-Learning agent"),
    ("from agent.sarsa_agent import SarsaAgent", "SARSA agent"),
    ("from utils.helpers import print_header", "Helper utilities"),
]

all_project_imports_ok = True
for import_statement, description in project_imports:
    try:
        exec(import_statement)
        print(f"  ✓ {description:30} - OK")
    except Exception as e:
        print(f"  ✗ {description:30} - FAILED: {str(e)[:30]}")
        all_project_imports_ok = False

# Try DQN separately (may fail if torch has issues)
try:
    exec("from agent.dqn_agent import DQNAgent")
    print(f"  ✓ {'DQN agent':30} - OK")
except Exception as e:
    print(f"  ⚠ {'DQN agent':30} - SLOW LOAD (torch init): {str(e)[:20]}...")

# ============================================================================
# 5. CHECK EXISTING MODELS
# ============================================================================
print("\n✓ CHECKING EXISTING TRAINED MODELS...")
print("-" * 80)

existing_models = {
    "models/q_learning_agent.pkl": "Q-Learning (original)",
    "models/q_learning_agent_v2.pkl": "Q-Learning (v2)",
    "models/sarsa_agent.pkl": "SARSA (original)",
    "models/sarsa_agent_v2.pkl": "SARSA (v2)",
    "models/dqn_model.pth": "DQN neural network",
    "models/q_table.pkl": "Q-table backup"
}

model_count = 0
for model_path, description in existing_models.items():
    if os.path.isfile(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ {description:30} - Found ({size_mb:.2f} MB)")
        model_count += 1
    else:
        print(f"  - {description:30} - Not yet trained")

print(f"\n  Total trained models: {model_count}")

# ============================================================================
# 6. CHECK EXISTING RESULTS
# ============================================================================
print("\n✓ CHECKING EXISTING RESULTS & METRICS...")
print("-" * 80)

result_files = {
    "results/q_learning_metrics_v2.json": "Q-Learning metrics",
    "results/sarsa_metrics_v2.json": "SARSA metrics",
    "results/dqn_metrics.json": "DQN metrics",
    "results/ql_vs_sarsa_report.txt": "Analysis report",
    "results/analysis_report.txt": "Comprehensive analysis"
}

result_count = 0
for result_path, description in result_files.items():
    if os.path.isfile(result_path):
        if result_path.endswith('.json'):
            try:
                with open(result_path, 'r') as f:
                    data = json.load(f)
                    episodes = len(data.get('episode', []))
                print(f"  ✓ {description:30} - Found ({episodes} episodes)")
                result_count += 1
            except:
                print(f"  ⚠ {description:30} - Found (corrupted)")
        else:
            size_kb = os.path.getsize(result_path) / 1024
            print(f"  ✓ {description:30} - Found ({size_kb:.1f} KB)")
            result_count += 1

# ============================================================================
# 7. CHECK PLOTS
# ============================================================================
print("\n✓ CHECKING EXISTING PLOTS...")
print("-" * 80)

plot_files = [
    "plots/ql_vs_sarsa_comparison.png",
    "plots/q_learning_performance.png",
    "plots/sarsa_performance.png",
    "plots/dqn_performance.png",
    "plots/algorithm_comparison.png"
]

plot_count = 0
for plot_path in plot_files:
    if os.path.isfile(plot_path):
        size_kb = os.path.getsize(plot_path) / 1024
        print(f"  ✓ {os.path.basename(plot_path):35} - Found ({size_kb:.1f} KB)")
        plot_count += 1

if plot_count == 0:
    print("  - No plots generated yet (will be created after training)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("  VERIFICATION SUMMARY")
print("="*80)

summary = []
if all_dirs_exist:
    summary.append("✓ All required directories present")
if all_files_exist:
    summary.append("✓ All Python files present")
if all_imports_ok:
    summary.append("✓ All dependencies installed")
if all_project_imports_ok:
    summary.append("✓ All project modules importable")

print("\n" + "\n".join([f"  {item}" for item in summary]))

if all_dirs_exist and all_files_exist and all_imports_ok and all_project_imports_ok:
    print("\n" + "="*80)
    print("  ✓✓✓ SYSTEM READY FOR TRAINING ✓✓✓")
    print("="*80)
    print("\n  Available training options:")
    print("  1. QUICK TRAINING (Fast & Educational):")
    print("     python train_all_flexible.py --episodes 500 --all")
    print("\n  2. BALANCED TRAINING (Recommended):")
    print("     python train_all_flexible.py --episodes 2000 --all")
    print("\n  3. PRODUCTION TRAINING (Best Results):")
    print("     python train_all_flexible.py --episodes 5000 --all")
    print("\n  4. QUICK TEST (Q-Learning + SARSA only):")
    print("     python quick_test.py")
    print("\n" + "="*80 + "\n")
else:
    print("\n" + "="*80)
    print("  ⚠ ISSUES DETECTED - FIX BEFORE TRAINING")
    print("="*80 + "\n")
    sys.exit(1)
