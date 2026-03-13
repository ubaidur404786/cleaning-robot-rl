"""
Centralized configuration for the Cleaning Robot RL project.
All hyperparameters, reward values, and environment settings in one place.
"""

# =============================================================================
# Environment Settings
# =============================================================================

# Phase 1: Simple open room
PHASE1_CONFIG = {
    "rows": 10,
    "cols": 10,
    "charger_pos": (0, 0),
    "start_pos": (0, 0),
    "battery_capacity": 50,
    "walls": [],           # No walls in Phase 1
    "furniture": [],       # No furniture in Phase 1
    "dirt_ratio": 1.0,     # All non-charger tiles start dirty
}

# Phase 2: Realistic apartment with walls, doorways, and furniture
# Layout (15x15):
#   Rows  0-5,  Cols 0-6:   Living room (charger at (0,0))
#   Rows  0-5,  Cols 8-14:  Kitchen (wall at col 7-8, doorway at row 2 col 7→8)
#   Row   6:                 Wall row (doorways at cols 3 and 10)
#   Rows  7-8:              Hallway (full width, open)
#   Row   9:                 Wall row (doorways at cols 3 and 10)
#   Rows 10-14, Cols 0-6:   Bedroom (wall at col 7, doorway at row 12)
#   Rows 10-14, Cols 8-11:  Bathroom (wall at col 12, doorway at row 12)
#   Rows 10-14, Cols 13-14: Storage closet
PHASE2_CONFIG = {
    "rows": 15,
    "cols": 15,
    "charger_pos": (0, 0),
    "start_pos": (0, 0),
    "battery_capacity": 80,
    "walls": [
        # Vertical wall between Living room and Kitchen (cols 7-8, rows 0-5)
        # Doorway at row 2, col 7→8 (so row 2 has no wall at col 7)
        (0, 7), (0, 8),
        (1, 7), (1, 8),
                (2, 8),   # row 2: doorway at col 7 (wall only at col 8)
        (3, 7),
        (4, 7), (4, 8),
        (5, 7), (5, 8),
        # Horizontal wall row 6 (top rooms ↔ hallway)
        # Doorways at cols 3 and 10
        (6, 0), (6, 1), (6, 2),         (6, 4), (6, 5), (6, 6),
        (6, 7), (6, 8), (6, 9),         (6, 11), (6, 12), (6, 13), (6, 14),
        # Horizontal wall row 9 (hallway ↔ bottom rooms)
        # Doorways at cols 3 and 10
        (9, 0), (9, 1), (9, 2),         (9, 4), (9, 5), (9, 6),
        (9, 7), (9, 8), (9, 9),         (9, 11), (9, 12), (9, 13), (9, 14),
        # Vertical wall between Bedroom and Bathroom (col 7, rows 10-14)
        # Doorway at row 12
        (10, 7),
        (11, 7),
                          # row 12: doorway
        (13, 7),
        (14, 7),
        # Vertical wall between Bathroom and Storage closet (col 12, rows 10-14)
        # Doorway at row 12
        (10, 12),
        (11, 12),
                          # row 12: doorway
        (13, 12),
        (14, 12),
    ],
    "furniture": [
        # Living room
        (0, 5),                          # TV stand
        (1, 1), (1, 2),                  # Couch
        (2, 2),                          # Coffee table
        # Kitchen
        (0, 9), (0, 10), (0, 11),        # Kitchen counter
        (0, 13),                         # Fridge
        (3, 10), (3, 11),                # Kitchen table
        # Bedroom
        (11, 1), (11, 2), (11, 3), (11, 4),  # Bed + nightstand
        (12, 1), (12, 2), (12, 3),            # Bed (continued)
        (10, 5), (10, 6),                     # Wardrobe
        # Bathroom
        (10, 9), (10, 10),               # Bathtub
        (14, 9), (14, 10),               # Toilet + sink
        # Storage closet
        (10, 13), (10, 14),              # Shelves
        (11, 13),                        # Shelves (continued)
    ],
    "dirt_ratio": 1.0,
}

# Battery discretization bins for state representation
# e.g., 5 bins: [0-10, 11-20, 21-30, 31-40, 41-50]
BATTERY_BINS = 5

# =============================================================================
# Reward Structure
# =============================================================================

REWARDS = {
    "clean_dirty":       +10.0,   # Move onto a dirty tile (auto-clean)
    "step_cost":          -0.1,   # Move onto a clean tile (step penalty)
    "wall_hit":           -2.0,   # Try to move into a wall / out of bounds
    "charge_success":     +5.0,   # Charge action at charger when not full
    "charge_full":        -1.0,   # Charge action at charger when already full
    "charge_away":        -5.0,   # Charge action away from charger
    "battery_dead":      -50.0,   # Battery reaches 0 (episode ends)
}

# =============================================================================
# Agent Hyperparameters
# =============================================================================

# Learning
ALPHA = 0.1              # Learning rate
GAMMA = 0.99             # Discount factor

# Epsilon-greedy
EPSILON_START = 1.0      # Initial exploration rate
EPSILON_MIN = 0.01       # Minimum exploration rate
EPSILON_DECAY = 0.995    # Multiplicative decay per episode

# UCB
UCB_C = 2.0              # Exploration constant for UCB

# Optimistic Initialization
OPTIMISTIC_INIT = 15.0   # Initial Q-value (optimistic, encourages exploration)

# =============================================================================
# Training Settings
# =============================================================================

TRAINING_EPISODES = 5000       # Episodes per experiment run
MAX_STEPS_PER_EPISODE = 500    # Safety cap to prevent infinite loops
NUM_SEEDS = 5                  # Number of random seeds for statistical significance
SEEDS = [42, 123, 256, 789, 1024]

# =============================================================================
# Actions
# =============================================================================

ACTIONS = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "Charge",
}
NUM_ACTIONS = len(ACTIONS)
