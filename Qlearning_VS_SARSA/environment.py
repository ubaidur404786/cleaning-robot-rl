"""
Cleaning Robot Environment.

A Roomba-style vacuum robot that cleans automatically as it moves.
The robot must learn WHERE to go and WHEN to recharge — not whether to clean.

State:  (row, col, battery_bin)  — agent does NOT see the dirt grid
Actions: Up(0), Down(1), Left(2), Right(3), Charge(4)

Phase 1: Simple open 10x10 grid, no walls.
Phase 2: Realistic apartment with walls, doorways, furniture.
"""

import numpy as np
from config import (
    PHASE1_CONFIG, REWARDS, BATTERY_BINS, NUM_ACTIONS, MAX_STEPS_PER_EPISODE,
)


class CleaningRobotEnv:
    """Grid-world environment for a vacuum cleaning robot with battery."""

    def __init__(self, config=None):
        """
        Initialize the environment.

        Parameters
        ----------
        config : dict, optional
            Environment configuration. Defaults to PHASE1_CONFIG.
        """
        cfg = config or PHASE1_CONFIG

        self.rows = cfg["rows"]
        self.cols = cfg["cols"]
        self.charger_pos = cfg["charger_pos"]
        self.start_pos = cfg["start_pos"]
        self.battery_capacity = cfg["battery_capacity"]
        self.walls = set(map(tuple, cfg.get("walls", [])))
        self.furniture = set(map(tuple, cfg.get("furniture", [])))
        self.dirt_ratio = cfg.get("dirt_ratio", 1.0)

        # Blocked cells = walls + furniture (agent can't step on them)
        self.blocked = self.walls | self.furniture

        # All walkable tiles (excluding blocked cells)
        self.walkable = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.blocked:
                    self.walkable.add((r, c))

        # Tiles that can be dirty (walkable minus charger)
        self.cleanable = self.walkable - {self.charger_pos}

        # Movement deltas: Up, Down, Left, Right
        self._deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }

        self.num_actions = NUM_ACTIONS
        self.battery_bins = BATTERY_BINS
        self.max_steps = MAX_STEPS_PER_EPISODE

        # State space size (for Q-table sizing)
        self.num_states = self.rows * self.cols * self.battery_bins

        # Will be set by reset()
        self.agent_pos = None
        self.battery = None
        self.dirt_grid = None
        self.steps = 0
        self.done = False
        self.total_dirty = 0
        self.cleaned_count = 0

    def reset(self, seed=None):
        """
        Reset the environment to the initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        state : tuple
            Initial state (row, col, battery_bin).
        """
        if seed is not None:
            np.random.seed(seed)

        self.agent_pos = self.start_pos
        self.battery = self.battery_capacity
        self.steps = 0
        self.done = False
        self.cleaned_count = 0

        # Initialize dirt grid: 1 = dirty, 0 = clean
        self.dirt_grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        if self.dirt_ratio >= 1.0:
            # All cleanable tiles are dirty
            for (r, c) in self.cleanable:
                self.dirt_grid[r, c] = 1
        else:
            # Random subset of cleanable tiles are dirty
            cleanable_list = list(self.cleanable)
            n_dirty = int(len(cleanable_list) * self.dirt_ratio)
            dirty_tiles = np.random.choice(
                len(cleanable_list), size=n_dirty, replace=False
            )
            for idx in dirty_tiles:
                r, c = cleanable_list[idx]
                self.dirt_grid[r, c] = 1

        self.total_dirty = int(self.dirt_grid.sum())

        return self._get_state()

    def step(self, action):
        """
        Execute an action in the environment.

        Parameters
        ----------
        action : int
            0=Up, 1=Down, 2=Left, 3=Right, 4=Charge

        Returns
        -------
        state : tuple
            New state (row, col, battery_bin).
        reward : float
            Reward received.
        done : bool
            Whether the episode is over.
        info : dict
            Additional information.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        self.steps += 1
        reward = 0.0
        info = {"event": "none"}

        if action == 4:
            # ---- CHARGE action ----
            if self.agent_pos == self.charger_pos:
                if self.battery < self.battery_capacity:
                    self.battery = self.battery_capacity
                    reward = REWARDS["charge_success"]
                    info["event"] = "charge_success"
                else:
                    reward = REWARDS["charge_full"]
                    info["event"] = "charge_full"
            else:
                reward = REWARDS["charge_away"]
                info["event"] = "charge_away"
                # Charging away still costs 1 battery (wasted action)
                self.battery -= 1
        else:
            # ---- MOVE action ----
            dr, dc = self._deltas[action]
            new_r = self.agent_pos[0] + dr
            new_c = self.agent_pos[1] + dc

            if (
                new_r < 0 or new_r >= self.rows
                or new_c < 0 or new_c >= self.cols
                or (new_r, new_c) in self.blocked
            ):
                # Hit wall / out of bounds / blocked
                reward = REWARDS["wall_hit"]
                info["event"] = "wall_hit"
                # Battery still drains for the attempt
                self.battery -= 1
            else:
                # Successful move
                self.agent_pos = (new_r, new_c)
                self.battery -= 1

                if self.dirt_grid[new_r, new_c] == 1:
                    # Auto-clean: stepped on dirty tile
                    self.dirt_grid[new_r, new_c] = 0
                    self.cleaned_count += 1
                    reward = REWARDS["clean_dirty"]
                    info["event"] = "clean_dirty"
                else:
                    # Stepped on clean tile
                    reward = REWARDS["step_cost"]
                    info["event"] = "step_clean"

        # ---- Check termination conditions ----

        # Battery death
        if self.battery <= 0:
            reward = REWARDS["battery_dead"]
            info["event"] = "battery_dead"
            self.done = True

        # All tiles cleaned
        elif self.dirt_grid.sum() == 0:
            info["event"] = "all_clean"
            self.done = True

        # Max steps reached
        elif self.steps >= self.max_steps:
            info["event"] = "max_steps"
            self.done = True

        # Build info
        info["battery"] = self.battery
        info["cleaned"] = self.cleaned_count
        info["total_dirty"] = self.total_dirty
        info["coverage"] = (
            self.cleaned_count / self.total_dirty if self.total_dirty > 0 else 1.0
        )
        info["steps"] = self.steps

        return self._get_state(), reward, self.done, info

    def _get_state(self):
        """
        Get the current state as (row, col, battery_bin).

        Battery is discretized into bins:
        - bin 0: battery in [0, capacity/bins)
        - bin 1: battery in [capacity/bins, 2*capacity/bins)
        - ...
        - bin (bins-1): battery in [(bins-1)*capacity/bins, capacity]
        """
        bin_size = self.battery_capacity / self.battery_bins
        battery_bin = min(
            int(self.battery / bin_size),
            self.battery_bins - 1,
        )
        return (self.agent_pos[0], self.agent_pos[1], battery_bin)

    def state_to_index(self, state):
        """Convert (row, col, battery_bin) to a flat index for Q-table."""
        r, c, b = state
        return r * self.cols * self.battery_bins + c * self.battery_bins + b

    def get_coverage(self):
        """Return the fraction of dirty tiles that have been cleaned."""
        if self.total_dirty == 0:
            return 1.0
        return self.cleaned_count / self.total_dirty

    def get_dirt_map(self):
        """Return a copy of the current dirt grid."""
        return self.dirt_grid.copy()

    def __repr__(self):
        return (
            f"CleaningRobotEnv({self.rows}x{self.cols}, "
            f"battery={self.battery}/{self.battery_capacity}, "
            f"cleaned={self.cleaned_count}/{self.total_dirty})"
        )
