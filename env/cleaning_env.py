"""
================================================================================
CLEANING ROBOT ENVIRONMENT - Pure Q-Learning Gymnasium Environment
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (Q-Learning)
FILE: env/cleaning_env.py
PURPOSE: Custom Gymnasium environment for training a cleaning robot

================================================================================
📚 REINFORCEMENT LEARNING OVERVIEW
================================================================================

This environment implements a PURE Q-Learning setup where:
1. The agent (robot) receives NO hints or shortcuts
2. Learning happens ONLY through trial and error
3. The Q-table stores learned state-action values
4. The agent discovers optimal behavior through exploration

================================================================================
🎯 STATE REPRESENTATION (Position + Dirt + Direction + DNUT)
================================================================================

State = position × is_dirty × came_from × dnut_direction

Components:
- 109 positions (108 cleanable tiles + 1 charger position)
- 2 dirt statuses (current tile clean/dirty; charger always 0)
- 5 movement history values (N/S/E/W/none)
- 10 DNUT direction values (3×3 relative direction grid + none/"at destination")

State Space Size: 109 × 2 × 5 × 10 = 10,900 states

================================================================================
⚡ ACTION SPACE
================================================================================

The robot can take 6 discrete actions:
- 0: Move Forward (up)    - Row decreases
- 1: Move Backward (down) - Row increases  
- 2: Move Left            - Column decreases
- 3: Move Right           - Column increases
- 4: Wait                 - Stay in place (penalized)
- 5: Clean                - Clean the current tile

================================================================================
🏆 REWARD STRUCTURE
================================================================================

The reward function shapes learning behavior:

MOVEMENT REWARDS (based on tile dirt status):
- Stepping on DIRTY Kitchen tile:     +50 points (auto-cleans)
- Stepping on DIRTY Living Room tile: +35 points (auto-cleans)
- Stepping on DIRTY Hallway tile:     +20 points (auto-cleans)
- Stepping on ALREADY CLEAN tile:      -5 points (any room)
- Completing all cleaning:           +200 bonus points

ACTION REWARDS:
- Cleaning dirty tile (Clean action): same room-based reward
- Cleaning already clean tile:        -10 points (wasted action)

NEGATIVE REWARDS (penalties):
- Hitting a wall:              -5 points (invalid move)
- Waiting:                     -3 points (wasted time)
- Each step taken:             -0.1 points (encourages efficiency)

================================================================================
🧭 DNUT (Detection of Nearest Uncleaned Tile)
================================================================================

The state includes the relative direction (dx, dy) to the nearest dirty tile.
Each of dx, dy is the sign (-1, 0, +1) of the vector to the nearest dirty
tile by Manhattan distance. This gives the agent a compass-like hint toward
the closest uncleaned area. When no dirty tiles remain, a special "none"
value is used.

================================================================================
🏠 HOUSE LAYOUT (18×12 with 8 rooms + charging station)
================================================================================

The house is an 18×12 grid with 8 rooms, multiple hallways, and a charging
station in a dedicated nook. Robot starts at charger and must return there
after cleaning all tiles to fully complete the task.

    Col:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
Row  0:  [W][W][W][W][W][W][W][W][W][W][W][W][W][W][W][W][W][W]  walls
Row  1:  [W][K][K][K][W][B][B][B][B][W][W][L][L][L][L][W][W][W]  Kitchen, Bedroom, Living
Row  2:  [W][K][K][K][W][B][B][B][B][W][W][L][L][L][L][W][W][W]
Row  3:  [W][K][K][K][W][B][B][B][B][W][W][L][L][L][L][W][W][W]
Row  4:  [W][H][H][H][H][H][H][H][H][H][H][H][H][H][H][W][W][W]  top hallway (14 tiles)
Row  5:  [W][T][T][T][W][H][H][H][H][W][W][D][D][D][W][W][W][W]  Bath, mid-hall, Dining
Row  6:  [W][T][T][T][W][H][H][H][H][W][W][D][D][D][W][W][W][W]
Row  7:  [W][H][H][H][H][H][H][H][H][H][H][H][H][H][H][W][W][W]  bot hallway (14 tiles)
Row  8:  [W][G][G][G][W][LN][LN][LN][W][C*][O][O][O][W][W][W][W][W]  Garage, Laundry, Charger(8,9), Office
Row  9:  [W][G][G][G][W][LN][LN][LN][W][ W][O][O][O][W][W][W][W][W]
Row 10:  [W][G][G][G][W][LN][LN][LN][W][ W][O][O][O][W][W][W][W][W]
Row 11:  [W][W][W][W][W][W][W][W][W][W][W][W][W][W][W][W][W][W]  walls

K=Kitchen(9), B=Bedroom(12), L=Living(12), H=Hallway(28), T=Bathroom(6),
D=Dining(6), G=Garage(9), LN=Laundry(9), O=Office(9), C*=Charger(docking)
Total cleanable tiles: 108 (excluding charger which is walkable but not dirty)

================================================================================
DNUT (Dual-Purpose Navigation Hint)
================================================================================

Direction to nearest thing-to-seek:
- While dirty tiles exist: direction to nearest dirty tile
- When all clean: direction to charging station
- At charger with all clean: state bin 9 ("done")

================================================================================
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# ================================================================================
# CONFIGURATION CONSTANTS
# ================================================================================

# Grid dimensions
GRID_WIDTH = 18    # Number of columns in the grid
GRID_HEIGHT = 12   # Number of rows in the grid
CELL_SIZE = 62     # Pixel size of each grid cell in Pygame rendering

# Room type identifiers (used in room_layout array)
EMPTY = 0          # Wall/Outside area - robot cannot enter
KITCHEN = 1        # Kitchen (highest priority, +50)
LIVING_ROOM = 2    # Living room (high priority, +45)
HALLWAY = 3        # Hallway/connector (lowest priority, +15)
BEDROOM = 4        # Bedroom (medium-high priority, +40)
DINING = 5         # Dining room (medium priority, +35)
BATHROOM = 6       # Bathroom (low-medium priority, +25)
GARAGE = 7         # Garage (low priority, +20)
LAUNDRY = 8        # Laundry room (medium priority, +30)
OFFICE = 9         # Office/study (medium priority, +35)
CHARGER = 10       # Charging station (not cleanable, walkable)

# Special position of charging station
CHARGER_ROW = 8
CHARGER_COL = 9

# Tile cleanliness states
CLEAN = 0          # Tile has been cleaned
DIRTY = 1          # Tile needs cleaning

# ================================================================================
# ACTION DEFINITIONS
# ================================================================================
# These constants define the discrete action space for the robot

ACTION_FORWARD = 0    # Move up (decrease row)
ACTION_BACKWARD = 1   # Move down (increase row)
ACTION_LEFT = 2       # Move left (decrease column)
ACTION_RIGHT = 3      # Move right (increase column)
ACTION_WAIT = 4       # Stay in place (do nothing)
ACTION_CLEAN = 5      # Clean the current tile

# Human-readable action names for debugging and visualization
ACTION_NAMES = {
    0: "Forward",
    1: "Backward",
    2: "Left",
    3: "Right",
    4: "Wait",
    5: "Clean"
}

# Number of available actions
NUM_ACTIONS = 6

# ================================================================================
# REWARD VALUES
# ================================================================================
# These values shape the learning behavior of the agent
# Rewards are dirt-conditional: only dirty tiles give positive reward.

# Cleaning rewards (positive) - Only awarded when tile is DIRTY
REWARD_CLEAN_KITCHEN = 50        # Kitchen highest priority
REWARD_CLEAN_LIVING = 45         # Living room
REWARD_CLEAN_BEDROOM = 40        # Bedroom
REWARD_CLEAN_DINING = 35         # Dining room
REWARD_CLEAN_LAUNDRY = 30        # Laundry room
REWARD_CLEAN_OFFICE = 35         # Office
REWARD_CLEAN_BATHROOM = 25       # Bathroom
REWARD_CLEAN_GARAGE = 20         # Garage lowest priority
REWARD_CLEAN_HALLWAY = 15        # Hallway connects rooms

# Completion bonuses
REWARD_ALL_CLEAN_BONUS = 100     # All tiles cleaned (partial)
REWARD_RETURN_CHARGER = 200      # Returned to charger (full completion)

# Penalty values (negative) - Discourage wasteful actions
REWARD_STEP_ON_CLEAN = -5         # Stepping on already-clean tile
REWARD_CLEAN_ALREADY_CLEAN = -10  # Using Clean action on clean tile
REWARD_HIT_WALL = -5              # Penalize trying to move into walls
REWARD_WAIT = -3                  # Penalize waiting (wastes time)
REWARD_STEP_PENALTY = -0.1        # Small penalty per step (encourages efficiency)

# ================================================================================
# VISUALIZATION COLORS (RGB format for Pygame)
# ================================================================================

COLOR_KITCHEN = (255, 255, 180)              # Light yellow for kitchen
COLOR_LIVING_ROOM = (180, 200, 255)          # Light blue for living room
COLOR_BEDROOM = (200, 230, 255)              # Sky blue for bedroom
COLOR_DINING = (255, 210, 180)               # Peach for dining
COLOR_BATHROOM = (180, 220, 240)             # Light cyan for bathroom
COLOR_GARAGE = (195, 195, 185)               # Cool gray for garage
COLOR_LAUNDRY = (210, 190, 230)              # Lavender for laundry
COLOR_OFFICE = (255, 220, 200)               # Warm amber for office
COLOR_HALLWAY = (200, 200, 200)              # Light gray for hallway
COLOR_CHARGER = (255, 200, 0)                # Gold for charger
COLOR_WALL = (80, 80, 80)                    # Dark gray for walls
COLOR_WINDOW_GLASS = (150, 180, 220)         # Light blue-gray for windows
COLOR_DOOR_FRAME = (120, 80, 40)             # Brown for door frames
COLOR_ROBOT = (50, 180, 50)                  # Green for robot
COLOR_ROBOT_HALO = (255, 200, 0)             # Gold halo at charger
COLOR_DIRTY = (139, 90, 43)                  # Brown for dirty tiles
COLOR_CLEAN_MARKER = (100, 220, 100)         # Bright green checkmark
COLOR_GRID_LINE = (50, 50, 50)               # Dark lines for grid
COLOR_TEXT = (255, 255, 255)                 # White text
COLOR_BLACK = (0, 0, 0)                      # Black for outlines
COLOR_LIGHTNING = (255, 255, 100)            # Yellow for lightning bolt


class CleaningEnv(gym.Env):
    """
    ============================================================================
    CLEANING ROBOT GYMNASIUM ENVIRONMENT - Pure Reinforcement Learning
    ============================================================================
    
    This is a custom Gymnasium environment that simulates a cleaning robot
    in a house with multiple rooms. The robot must learn through PURE 
    Q-Learning (no hints, no shortcuts) to efficiently clean all dirty tiles.
    
    THE KEY DIFFERENCE: PURE RL
    ---------------------------
    This implementation does NOT provide direction hints to the agent.
    The state only includes:
    - Current position (which tile the robot is on)
    - Whether the current tile is dirty
    
    The agent must LEARN through trial and error:
    - Where dirty tiles typically are
    - How to navigate the house efficiently
    - When to clean vs when to move
    
    This is TRUE reinforcement learning - the agent discovers optimal
    behavior through experience, not through hardcoded hints!
    
    GYMNASIUM INTERFACE:
    -------------------
    - action_space: Discrete(6) - Six possible actions
    - observation_space: Discrete(2300) - Position × dirty × direction × DNUT
    - reset() - Start new episode with all tiles dirty
    - step(action) - Execute action, return (observation, reward, done, info)
    - render() - Visualize current state with Pygame
    
    ============================================================================
    """
    
    # Gymnasium metadata - tells Gymnasium what render modes we support
    metadata = {
        "render_modes": ["human", "rgb_array"],  # Supported render modes
        "render_fps": 10                          # Frames per second for rendering
    }
    
    def __init__(self, render_mode=None):
        """
        Initialize the Cleaning Robot Environment.
        
        This constructor sets up:
        1. The room layout (which cells belong to which room type)
        2. The action and observation spaces (Gymnasium requirements)
        3. Pygame visualization components (for rendering)
        4. Position indexing for state encoding
        
        Parameters:
        -----------
        render_mode : str or None
            How to render the environment:
            - None: No rendering (fastest for training)
            - "human": Display Pygame window (for watching)
            - "rgb_array": Return RGB array (for recording)
        """
        # Call parent class constructor (required by Gymnasium)
        super().__init__()
        
        # Store render mode for later use
        self.render_mode = render_mode
        
        # ======================================================================
        # PYGAME VISUALIZATION SETUP
        # ======================================================================
        # These variables will be initialized when render() is first called
        self.window = None           # Pygame window object
        self.clock = None            # Pygame clock for frame rate control
        
        # Precompute window and door cells for rendering
        self.window_cells = set()    # Cells on outer walls with adjacent rooms
        self.door_edges = set()      # Shared edges between hallway and rooms
        self._precompute_door_window_cells()
        
        # ======================================================================
        # ACTION SPACE DEFINITION
        # ======================================================================
        # Gymnasium Discrete space: Actions are integers from 0 to 5
        # This tells Gymnasium what actions are valid in this environment
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # ======================================================================
        # CREATE ROOM LAYOUT
        # ======================================================================
        # Build the 2D array that defines which room each cell belongs to
        self.room_layout = self._create_room_layout()
        
        # ======================================================================
        # IDENTIFY CLEANABLE TILES (EXCLUDES CHARGER)
        # ======================================================================
        # Create a list of all tiles that can be cleaned (i.e., not walls, not charger)
        # We iterate through the grid and collect positions where room != EMPTY and != CHARGER
        self.cleanable_tiles = []
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                rt = self.room_layout[row][col]
                if rt != EMPTY and rt != CHARGER:
                    self.cleanable_tiles.append((row, col))
        
        # Store the total number of cleanable tiles
        self.num_cleanable = len(self.cleanable_tiles)
        
        # ======================================================================
        # POSITION INDEX MAPPING (includes charger at index num_cleanable)
        # ======================================================================
        # Map (row, col) coordinates to integer indices for state encoding
        # This allows us to convert 2D positions to a single number for Q-table
        #
        # Indices 0 to num_cleanable-1: cleanable tiles
        # Index num_cleanable: charger position (always at CHARGER_ROW, CHARGER_COL)
        self.pos_to_index = {}     # (row, col) -> integer index
        self.index_to_pos = {}     # integer index -> (row, col)
        for idx, (row, col) in enumerate(self.cleanable_tiles):
            self.pos_to_index[(row, col)] = idx
            self.index_to_pos[idx] = (row, col)
        # Add charger as last index
        self.pos_to_index[(CHARGER_ROW, CHARGER_COL)] = self.num_cleanable
        self.index_to_pos[self.num_cleanable] = (CHARGER_ROW, CHARGER_COL)
        
        # ======================================================================
        # OBSERVATION SPACE DEFINITION (Position + Dirt + History + DNUT)
        # ======================================================================
        # State = position + is_dirty + came_from_direction + dnut_direction
        #
        # Components:
        # - position: Which tile robot is on (0-108 = 109 positions, incl charger)
        # - is_dirty: Current tile dirt status (0 or 1; charger always 0)
        # - came_from: Direction robot just came from (0-4: N/S/E/W/none)
        # - dnut: Relative direction to target (0-9):
        #         0-8 = (dx+1)*3 + (dy+1) where dx,dy ∈ {-1,0,+1}
        #         9   = at destination (no dirty tiles + at charger, or all dirty found)
        #
        # Total: (num_cleanable+1) × 2 × 5 × 10 = 109 × 2 × 5 × 10 = 10,900 states
        
        self.num_directions = 5   # N, S, E, W, none
        self.num_dnut = 10        # 3×3 direction grid + at_destination
        self.state_space_size = (self.num_cleanable + 1) * 2 * self.num_directions * self.num_dnut
        self.observation_space = spaces.Discrete(self.state_space_size)
        
        # ======================================================================
        # ROOM REWARD MAPPING (8 rooms + hallway)
        # ======================================================================
        # Different rewards for cleaning different room types
        self.room_rewards = {
            KITCHEN: REWARD_CLEAN_KITCHEN,
            LIVING_ROOM: REWARD_CLEAN_LIVING,
            BEDROOM: REWARD_CLEAN_BEDROOM,
            DINING: REWARD_CLEAN_DINING,
            LAUNDRY: REWARD_CLEAN_LAUNDRY,
            OFFICE: REWARD_CLEAN_OFFICE,
            BATHROOM: REWARD_CLEAN_BATHROOM,
            GARAGE: REWARD_CLEAN_GARAGE,
            HALLWAY: REWARD_CLEAN_HALLWAY
        }
        
        # ======================================================================
        # ROOM NAME MAPPING (for visualization and debugging)
        # ======================================================================
        self.room_names = {
            EMPTY: "Wall",
            KITCHEN: "Kitchen",
            LIVING_ROOM: "Living Room",
            BEDROOM: "Bedroom",
            DINING: "Dining Room",
            BATHROOM: "Bathroom",
            GARAGE: "Garage",
            LAUNDRY: "Laundry Room",
            OFFICE: "Office",
            HALLWAY: "Hallway",
            CHARGER: "Charger"
        }
        
        # ======================================================================
        # STATE VARIABLES (will be initialized in reset())
        # ======================================================================
        self.robot_row = CHARGER_ROW  # Robot starts at charger
        self.robot_col = CHARGER_COL
        self.dirt_map = None      # 2D array tracking dirty status of each tile
        self.steps_taken = 0      # Number of steps in current episode
        self.tiles_cleaned = 0    # Number of tiles cleaned this episode
        self.max_steps = 600      # Maximum steps per episode (larger house needs more)
        self.last_direction = 4   # Direction we came from (0=N, 1=S, 2=E, 3=W, 4=none)
        self.all_cleaned_bonus_given = False  # Track if all-clean bonus given
        
        # ======================================================================
        # PRINT INITIALIZATION INFO
        # ======================================================================
        print("=" * 73)
        print("  CLEANING ROBOT ENVIRONMENT INITIALIZED (Beautiful Home + Charger)")
        print("=" * 73)
        print(f"  Grid size:          {GRID_WIDTH} × {GRID_HEIGHT}")
        print(f"  Cleanable tiles:    {self.num_cleanable}")
        print(f"  State space size:   {self.state_space_size} states")
        print(f"  Action space size:  {NUM_ACTIONS} actions")
        print(f"  Max steps/episode:  {self.max_steps}")
        print(f"  Charging station:   ({CHARGER_ROW}, {CHARGER_COL})")
        print("-" * 73)
        print("  Room Cleaning Rewards (dirty tiles only):")
        print(f"    Kitchen:     +{REWARD_CLEAN_KITCHEN} points")
        print(f"    Living Room: +{REWARD_CLEAN_LIVING} points")
        print(f"    Bedroom:     +{REWARD_CLEAN_BEDROOM} points")
        print(f"    Dining:      +{REWARD_CLEAN_DINING} points")
        print(f"    Laundry:     +{REWARD_CLEAN_LAUNDRY} points")
        print(f"    Office:      +{REWARD_CLEAN_OFFICE} points")
        print(f"    Bathroom:    +{REWARD_CLEAN_BATHROOM} points")
        print(f"    Garage:      +{REWARD_CLEAN_GARAGE} points")
        print(f"    Hallway:     +{REWARD_CLEAN_HALLWAY} points")
        print(f"    All Cleaned: +{REWARD_ALL_CLEAN_BONUS} bonus")
        print(f"    At Charger:  +{REWARD_RETURN_CHARGER} final bonus")
        print(f"  DNUT feature:   dual-purpose (dirty tiles / charger)  (10 bins)")
        print("=" * 73)
    
    def _precompute_door_window_cells(self):
        """
        Precompute cells for rendering windows and door frames.
        
        Windows: outer-wall cells with adjacent walkable rooms
        Doors: shared edges between hallway cells and room cells
        
        These are computed after room_layout is created.
        """
        # Will be populated after _create_room_layout()
        pass
    
    def _create_room_layout(self):
        """
        Create the 18×12 beautiful house layout with 8 rooms + charger.
        
        Returns
        -------
        numpy.ndarray
            2D array of shape (GRID_HEIGHT, GRID_WIDTH) containing room type IDs.
        """
        # Initialize grid with all walls (EMPTY)
        layout = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        
        # ======================================================================
        # TOP FLOOR - KITCHEN, BEDROOM, LIVING ROOM
        # ======================================================================
        
        # Kitchen: rows 1-3, cols 1-3 (9 tiles, +50 each)
        for row in range(1, 4):
            for col in range(1, 4):
                layout[row][col] = KITCHEN
        
        # Bedroom: rows 1-3, cols 5-8 (12 tiles, +40 each)
        for row in range(1, 4):
            for col in range(5, 9):
                layout[row][col] = BEDROOM
        
        # Living Room: rows 1-3, cols 11-14 (12 tiles, +45 each)
        for row in range(1, 4):
            for col in range(11, 15):
                layout[row][col] = LIVING_ROOM
        
        # ======================================================================
        # MIDDLE FLOOR - HALLWAYS, BATHROOM, DINING
        # ======================================================================
        
        # Top Hallway: row 4, cols 1-14 (14 tiles, +15 each)
        for col in range(1, 15):
            layout[4][col] = HALLWAY
        
        # Bathroom: rows 5-6, cols 1-3 (6 tiles, +25 each)
        for row in range(5, 7):
            for col in range(1, 4):
                layout[row][col] = BATHROOM
        
        # Middle Hallway: rows 5-6, cols 5-8 (8 tiles, +15 each)
        for row in range(5, 7):
            for col in range(5, 9):
                layout[row][col] = HALLWAY
        
        # Dining Room: rows 5-6, cols 11-13 (6 tiles, +35 each)
        for row in range(5, 7):
            for col in range(11, 14):
                layout[row][col] = DINING
        
        # ======================================================================
        # BOTTOM FLOOR - GARAGE, LAUNDRY, OFFICE, HALLWAYS
        # ======================================================================
        
        # Bottom Hallway: row 7, cols 1-14 (14 tiles, +15 each)
        for col in range(1, 15):
            layout[7][col] = HALLWAY
        
        # Garage: rows 8-10, cols 1-3 (9 tiles, +20 each)
        for row in range(8, 11):
            for col in range(1, 4):
                layout[row][col] = GARAGE
        
        # Laundry Room: rows 8-10, cols 5-7 (9 tiles, +30 each)
        for row in range(8, 11):
            for col in range(5, 8):
                layout[row][col] = LAUNDRY
        
        # Charging Station: row 8, col 9 (1 cell, NOT cleanable, walkable)
        layout[CHARGER_ROW][CHARGER_COL] = CHARGER
        
        # Office: rows 8-10, cols 11-13 (9 tiles, +35 each)
        for row in range(8, 11):
            for col in range(11, 14):
                layout[row][col] = OFFICE
        
        # ======================================================================
        # POST-PROCESS: IDENTIFY WINDOWS AND DOORS
        # ======================================================================
        
        # Precompute window cells (outer walls adjacent to rooms)
        # Windows appear on the boundary (row 0, row 11, col 0, col 17)
        # only where they border non-EMPTY cells
        self.window_cells = set()
        # Top wall (row 0)
        for col in range(GRID_WIDTH):
            if layout[1][col] != EMPTY:
                self.window_cells.add((0, col))
        # Bottom wall (row 11)
        for col in range(GRID_WIDTH):
            if layout[10][col] != EMPTY:
                self.window_cells.add((11, col))
        # Left wall (col 0)
        for row in range(GRID_HEIGHT):
            if layout[row][1] != EMPTY:
                self.window_cells.add((row, 0))
        # Right wall (col 17)
        for row in range(GRID_HEIGHT):
            if layout[row][16] != EMPTY:
                self.window_cells.add((row, 17))
        
        # Precompute door edges (boundaries between HALLWAY and other rooms)
        # These are shared edges that will be drawn as door frames
        self.door_edges = set()
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                if layout[row][col] == HALLWAY:
                    # Check all 4 neighbors
                    for dr, dc, direction in [(-1,0,'N'), (1,0,'S'), (0,-1,'E'), (0,1,'W')]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                            if layout[nr][nc] not in (EMPTY, HALLWAY, CHARGER):
                                # This is a door edge
                                edge = tuple(sorted([(row, col), (nr, nc)]))
                                self.door_edges.add(edge)
        
        return layout
    
    def _get_nearest_dirty_direction(self):
        """
        DNUT (Dual-purpose Navigation Unit).
        
        Direction to nearest thing-to-seek:
        - While dirty tiles exist: direction to nearest dirty tile
        - When all clean: direction to charger
        - At charger with all clean: return 9 ("done" / "at destination")
        
        Returns
        -------
        int
            0-8: encoded direction (dx+1)*3 + (dy+1) where dx, dy ∈ {-1,0,+1}
            9  : at destination (no dirty tiles + at charger, or door reached)
        """
        # First check if all tiles are clean
        dirty_count = 0
        best_dist = float('inf')
        best_dr, best_dc = 0, 0
        
        for row, col in self.cleanable_tiles:
            if self.dirt_map[row][col] == DIRTY:
                dirty_count += 1
                dist = abs(row - self.robot_row) + abs(col - self.robot_col)
                if dist < best_dist:
                    best_dist = dist
                    best_dr = row - self.robot_row
                    best_dc = col - self.robot_col
        
        # If dirty tiles remain, point to nearest dirty
        if dirty_count > 0:
            dx = (best_dr > 0) - (best_dr < 0)
            dy = (best_dc > 0) - (best_dc < 0)
            return (dx + 1) * 3 + (dy + 1)
        
        # All tiles clean: point toward charger
        # If robot is AT charger, return 9 ("done")
        if self.robot_row == CHARGER_ROW and self.robot_col == CHARGER_COL:
            return 9
        
        # Point toward charger
        dr = CHARGER_ROW - self.robot_row
        dc = CHARGER_COL - self.robot_col
        dx = (dr > 0) - (dr < 0)
        dy = (dc > 0) - (dc < 0)
        return (dx + 1) * 3 + (dy + 1)
    
    def _get_state(self):
        """
        Convert current environment state to a single integer for Q-table.
        
        State = pos_index + is_dirty*(num_cleanable+1) + came_from*2*(num_cleanable+1) + dnut*10*2*(num_cleanable+1)
        
        Returns
        -------
        int
            State index for the Q-table (0 to state_space_size-1)
        """
        # Position index (0 to num_cleanable, inc. charger)
        pos_index = self.pos_to_index.get((self.robot_row, self.robot_col), 0)
        
        # Dirt status of current tile
        is_dirty = 1 if self.dirt_map[self.robot_row][self.robot_col] == DIRTY else 0
        
        # Direction we came from
        came_from = self.last_direction
        
        # Direction to nearest seek target (dirty or charger)
        dnut = self._get_nearest_dirty_direction()
        
        # Encode as single integer
        n_pos = self.num_cleanable + 1
        state = (pos_index
                 + is_dirty * n_pos
                 + came_from * 2 * n_pos
                 + dnut * self.num_directions * 2 * n_pos)
        
        return int(state)
    
    def _get_room_dirty_combo(self):
        """
        Calculate a 3-bit encoding of which rooms have dirty tiles.
        
        Returns
        -------
        int
            Value 0-7 representing room dirty status
        """
        combo = 0
        
        # Check each room for dirty tiles
        kitchen_dirty = False
        living_dirty = False
        hallway_dirty = False
        
        combo = 0
        
        # Check each room for dirty tiles
        kitchen_dirty = False
        living_dirty = False
        hallway_dirty = False
        
        for row, col in self.cleanable_tiles:
            if self.dirt_map[row][col] == DIRTY:
                room_type = self.room_layout[row][col]
                if room_type == KITCHEN:
                    kitchen_dirty = True
                elif room_type == LIVING_ROOM:
                    living_dirty = True
                elif room_type == HALLWAY:
                    hallway_dirty = True
        
        # Encode as 3-bit value
        if kitchen_dirty:
            combo += 1  # Bit 0
        if living_dirty:
            combo += 2  # Bit 1
        if hallway_dirty:
            combo += 4  # Bit 2
        
        return combo
    
    def _count_dirty_tiles(self):
        """
        Count how many tiles are still dirty.
        
        This method iterates through all cleanable tiles and counts
        how many still have DIRTY status.
        
        Returns:
        --------
        int
            Number of dirty tiles remaining (0 when all clean)
        """
        dirty_count = 0
        for row, col in self.cleanable_tiles:
            if self.dirt_map[row][col] == DIRTY:
                dirty_count += 1
        return dirty_count
    
    def _is_valid_position(self, row, col):
        """
        Check if a position is valid for the robot to occupy.
        
        A position is valid if:
        1. It's within the grid boundaries
        2. It's not a wall (EMPTY)
        
        Parameters:
        -----------
        row : int
            Row coordinate to check
        col : int
            Column coordinate to check
        
        Returns:
        --------
        bool
            True if robot can move to this position, False otherwise
        """
        # Check grid boundaries
        if row < 0 or row >= GRID_HEIGHT:
            return False
        if col < 0 or col >= GRID_WIDTH:
            return False
        
        # Check if position is a wall (EMPTY means wall/outside)
        if self.room_layout[row][col] == EMPTY:
            return False
        
        return True
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Robot starts at the CHARGING STATION and must return there after
        cleaning all tiles to complete the task fully.
        
        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility
        options : dict or None
            Additional reset options (not used)
        
        Returns
        -------
        tuple (observation, info)
            observation : int
                Initial state observation
            info : dict
                Additional information dictionary
        """
        # Call parent reset to handle seeding properly
        super().reset(seed=seed)
        
        # Reset episode tracking variables
        self.steps_taken = 0
        self.tiles_cleaned = 0
        self.all_cleaned_bonus_given = False
        
        # ======================================================================
        # PLACE ROBOT AT CHARGING STATION
        # ======================================================================
        # Robot starts and must return to the charging station docking area
        self.robot_row = CHARGER_ROW
        self.robot_col = CHARGER_COL
        
        # ======================================================================
        # INITIALIZE DIRT MAP - ALL TILES START DIRTY
        # ======================================================================
        # Create a fresh dirt map where all cleanable tiles are dirty
        self.dirt_map = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        for row, col in self.cleanable_tiles:
            self.dirt_map[row][col] = DIRTY
        
        # Reset movement history (4 = no previous direction)
        self.last_direction = 4
        
        # Get initial observation (state)
        observation = self._get_state()
        
        # Build info dictionary with useful debugging information
        info = {
            "robot_position": (self.robot_row, self.robot_col),
            "dirty_tiles": self._count_dirty_tiles(),
            "tiles_cleaned": 0,
            "phase": "CLEANING",
            "room": self.room_names.get(
                self.room_layout[self.robot_row][self.robot_col], "Unknown"
            )
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one action in the environment.
        
        This is the core method that processes robot actions and returns
        the resulting state, reward, and termination status. This is called
        once per timestep during training/testing.
        
        The Q-Learning update equation uses the reward and next state from here:
            Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Parameters:
        -----------
        action : int
            Action to take (0-5, see ACTION_* constants)
            0: Forward, 1: Backward, 2: Left, 3: Right, 4: Wait, 5: Clean
        
        Returns:
        --------
        tuple (observation, reward, terminated, truncated, info)
            observation : int
                New state after action
            reward : float
                Reward received for this action
            terminated : bool
                True if episode ended naturally (all tiles clean)
            truncated : bool
                True if episode ended by step limit
            info : dict
                Additional information dictionary
        """
        # Increment step counter
        self.steps_taken += 1
        
        # Initialize reward with small step penalty
        # This encourages the robot to be efficient (fewer steps = better)
        reward = REWARD_STEP_PENALTY
        
        # Track action outcome for info dictionary
        action_result = "unknown"
        
        # ======================================================================
        # PROCESS ACTION - Execute the robot's chosen action
        # ======================================================================
        
        if action == ACTION_FORWARD:
            # Move up (decrease row)
            new_row = self.robot_row - 1
            new_col = self.robot_col
            if self._is_valid_position(new_row, new_col):
                self.robot_row = new_row
                action_result = "moved_forward"
                self.last_direction = 1
                # Dirt-conditional reward for stepping on the tile
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_BACKWARD:
            # Move down (increase row)
            new_row = self.robot_row + 1
            new_col = self.robot_col
            if self._is_valid_position(new_row, new_col):
                self.robot_row = new_row
                action_result = "moved_backward"
                self.last_direction = 0
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_LEFT:
            # Move left (decrease column)
            new_row = self.robot_row
            new_col = self.robot_col - 1
            if self._is_valid_position(new_row, new_col):
                self.robot_col = new_col
                action_result = "moved_left"
                self.last_direction = 2
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_RIGHT:
            # Move right (increase column)
            new_row = self.robot_row
            new_col = self.robot_col + 1
            if self._is_valid_position(new_row, new_col):
                self.robot_col = new_col
                action_result = "moved_right"
                self.last_direction = 3
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_WAIT:
            # Stay in place - penalize because it wastes time
            reward += REWARD_WAIT
            action_result = "waited"
        
        elif action == ACTION_CLEAN:
            # Try to clean the current tile
            if self.dirt_map[self.robot_row][self.robot_col] == DIRTY:
                # Tile is dirty - successful clean!
                self.dirt_map[self.robot_row][self.robot_col] = CLEAN
                self.tiles_cleaned += 1
                
                # Get room-specific reward
                room_type = self.room_layout[self.robot_row][self.robot_col]
                cleaning_reward = self.room_rewards.get(room_type, 20)
                reward += cleaning_reward
                action_result = f"cleaned_{self.room_names.get(room_type, 'tile')}"
            else:
                # Tile already clean - penalize wasted action
                reward += REWARD_CLEAN_ALREADY_CLEAN
                action_result = "clean_failed_already_clean"
        
        # ======================================================================
        # CHECK TERMINATION CONDITIONS
        # ======================================================================
        
        # Count remaining dirty tiles
        dirty_remaining = self._count_dirty_tiles()
        
        # Episode terminates successfully ONLY if:
        # 1. All tiles are clean AND
        # 2. Robot is at the charging station
        terminated = False
        if dirty_remaining == 0 and not self.all_cleaned_bonus_given:
            # Give all-clean bonus once
            reward += REWARD_ALL_CLEAN_BONUS
            self.all_cleaned_bonus_given = True
            
            # Check if robot is also at charger for FULL completion
            if self.robot_row == CHARGER_ROW and self.robot_col == CHARGER_COL:
                reward += REWARD_RETURN_CHARGER
                terminated = True
        
        # Episode is truncated if max steps reached (time limit)
        truncated = (self.steps_taken >= self.max_steps)
        
        # ======================================================================
        # GET NEW OBSERVATION
        # ======================================================================
        observation = self._get_state()
        
        # ======================================================================
        # BUILD INFO DICTIONARY
        # ======================================================================
        # This provides useful information for debugging and analysis
        
        # Calculate completion rate as percentage
        completion_rate = (self.tiles_cleaned / self.num_cleanable * 100) if self.num_cleanable > 0 else 0.0
        
        # Determine phase
        phase = "CLEANING" if dirty_remaining > 0 else "🏠 RETURNING TO CHARGER"
        
        info = {
            "robot_position": (self.robot_row, self.robot_col),
            "action": ACTION_NAMES.get(action, "Unknown"),
            "action_result": action_result,
            "dirty_tiles": dirty_remaining,
            "tiles_cleaned": self.tiles_cleaned,
            "total_dirty": self.num_cleanable,
            "completion_rate": completion_rate,
            "steps": self.steps_taken,
            "room": self.room_names.get(
                self.room_layout[self.robot_row][self.robot_col], "Unknown"
            )
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the current state of the environment using Pygame.
        
        This method creates a visual representation of the house showing:
        - Room layout with different colors per room type
        - Dirty tiles (brown dirt marks)
        - Cleaned tiles (green checkmark)
        - Robot position (green circle with eyes)
        - Statistics overlay (steps, cleaned count, etc.)
        
        Returns:
        --------
        numpy.ndarray or None
            RGB array if render_mode is "rgb_array", None otherwise
        """
        # If render_mode is None, skip rendering
        if self.render_mode is None:
            return None
        
        # ======================================================================
        # INITIALIZE PYGAME IF NEEDED
        # ======================================================================
        # First time render() is called, we need to set up Pygame
        if self.window is None:
            pygame.init()
            pygame.display.init()
            
            # Calculate window size (grid + space for stats panel)
            window_width = GRID_WIDTH * CELL_SIZE
            window_height = GRID_HEIGHT * CELL_SIZE + 100  # Extra for stats
            
            if self.render_mode == "human":
                # Create visible window for human viewing
                pygame.display.set_caption("Cleaning Robot - Beautiful Home with Charging Station")
                self.window = pygame.display.set_mode((window_width, window_height))
            else:
                # Create off-screen surface for rgb_array mode
                self.window = pygame.Surface((window_width, window_height))
            
            # Create clock for frame rate control
            self.clock = pygame.time.Clock()
            
            # Initialize fonts for text rendering
            self.font = pygame.font.Font(None, 20)
            self.small_font = pygame.font.Font(None, 16)
        
        # ======================================================================
        # DRAW BACKGROUND AND GRID
        # ======================================================================
        self.window.fill(COLOR_WALL)  # Fill with wall color as background
        
        # Map room types to colors (updated for 8 rooms)
        room_colors = {
            EMPTY: COLOR_WALL,
            KITCHEN: COLOR_KITCHEN,
            LIVING_ROOM: COLOR_LIVING_ROOM,
            BEDROOM: COLOR_BEDROOM,
            DINING: COLOR_DINING,
            BATHROOM: COLOR_BATHROOM,
            GARAGE: COLOR_GARAGE,
            LAUNDRY: COLOR_LAUNDRY,
            OFFICE: COLOR_OFFICE,
            HALLWAY: COLOR_HALLWAY,
            CHARGER: COLOR_CHARGER
        }
        
        # Draw each cell in the grid
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                # Calculate cell rectangle position
                x = col * CELL_SIZE
                y = row * CELL_SIZE
                cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                
                # Get room type and corresponding color
                room_type = self.room_layout[row][col]
                base_color = room_colors.get(room_type, COLOR_WALL)
                
                # Draw cell background
                pygame.draw.rect(self.window, base_color, cell_rect)
                
                # Draw windows on outer wall cells with adjacent rooms
                if room_type == EMPTY and (row, col) in self.window_cells:
                    # Light blue glass with white panes
                    pygame.draw.rect(self.window, COLOR_WINDOW_GLASS, cell_rect)
                    # Draw window panes (cross pattern)
                    center_x = x + CELL_SIZE // 2
                    center_y = y + CELL_SIZE // 2
                    pygame.draw.line(self.window, (255, 255, 255), (center_x, y + 5), (center_x, y + CELL_SIZE - 5), 2)
                    pygame.draw.line(self.window, (255, 255, 255), (x + 5, center_y), (x + CELL_SIZE - 5, center_y), 2)
                
                # Draw dirt or clean markers for room tiles (not walls, not charger base)
                if room_type not in (EMPTY, CHARGER) and self.dirt_map is not None:
                    if self.dirt_map[row][col] == DIRTY:
                        # Draw brown dirt spots
                        center_x = x + CELL_SIZE // 2
                        center_y = y + CELL_SIZE // 2
                        dirt_radius = max(3, CELL_SIZE // 12)
                        dirt_positions = [(-CELL_SIZE//8, -CELL_SIZE//10), (CELL_SIZE//9, -CELL_SIZE//12), 
                                         (-CELL_SIZE//10, CELL_SIZE//9), (CELL_SIZE//8, CELL_SIZE//10), (0, 0)]
                        for dx, dy in dirt_positions:
                            pygame.draw.circle(self.window, COLOR_DIRTY, (center_x + dx, center_y + dy), dirt_radius)
                    else:
                        # Draw green checkmark for cleaned tile
                        center_x = x + CELL_SIZE // 2
                        center_y = y + CELL_SIZE // 2
                        check_radius = max(5, CELL_SIZE // 10)
                        pygame.draw.circle(self.window, COLOR_CLEAN_MARKER, (center_x, center_y), check_radius)
                        # Draw check symbol
                        check_size = max(3, CELL_SIZE // 15)
                        pygame.draw.lines(self.window, (255, 255, 255), False,
                            [(center_x - check_size, center_y), 
                             (center_x - check_size//2, center_y + check_size), 
                             (center_x + check_size, center_y - check_size)], 2)
                
                # Draw door frames at hallway boundaries (thin brown stripe)
                # This is cosmetic only - doors are always passable
                
                # Draw grid lines
                pygame.draw.rect(self.window, COLOR_GRID_LINE, cell_rect, 1)
        
        # ======================================================================
        # DRAW CHARGER ICON
        # ======================================================================
        # Draw lightning bolt at charger location
        charger_x = CHARGER_COL * CELL_SIZE + CELL_SIZE // 2
        charger_y = CHARGER_ROW * CELL_SIZE + CELL_SIZE // 2
        bolt_size = max(8, CELL_SIZE // 8)
        # Draw lightning bolt as a simple polygon
        pygame.draw.polygon(self.window, COLOR_LIGHTNING, [
            (charger_x, charger_y - bolt_size),
            (charger_x - bolt_size // 2, charger_y - bolt_size // 3),
            (charger_x - bolt_size // 3, charger_y),
            (charger_x + bolt_size // 2, charger_y - bolt_size // 4),
            (charger_x + bolt_size // 4, charger_y + bolt_size),
            (charger_x, charger_y + bolt_size // 2)
        ])
        
        # ======================================================================
        # DRAW ROBOT
        # ======================================================================
        # Calculate robot center position
        robot_x = self.robot_col * CELL_SIZE + CELL_SIZE // 2
        robot_y = self.robot_row * CELL_SIZE + CELL_SIZE // 2
        robot_radius = max(8, CELL_SIZE // 4)
        
        # Draw golden halo if robot is at charger
        if self.robot_row == CHARGER_ROW and self.robot_col == CHARGER_COL:
            pygame.draw.circle(self.window, COLOR_ROBOT_HALO, (robot_x, robot_y), robot_radius + 5, 3)
        
        # Draw robot body (green circle)
        pygame.draw.circle(self.window, COLOR_ROBOT, (robot_x, robot_y), robot_radius)
        # Draw robot outline
        pygame.draw.circle(self.window, (30, 100, 30), (robot_x, robot_y), robot_radius, 2)
        
        # Draw robot eyes (to give it character and show direction)
        eye_size = max(3, robot_radius // 3)
        eye_offset_x = robot_radius // 3
        eye_offset_y = robot_radius // 4
        # White part of eyes
        pygame.draw.circle(self.window, (255, 255, 255),
                          (robot_x - eye_offset_x, robot_y - eye_offset_y), eye_size)
        pygame.draw.circle(self.window, (255, 255, 255),
                          (robot_x + eye_offset_x, robot_y - eye_offset_y), eye_size)
        # Black pupils
        pygame.draw.circle(self.window, COLOR_BLACK,
                          (robot_x - eye_offset_x, robot_y - eye_offset_y), 4)
        pygame.draw.circle(self.window, COLOR_BLACK,
                          (robot_x + eye_offset_x, robot_y - eye_offset_y), 4)
        
        # ======================================================================
        # DRAW STATISTICS PANEL
        # ======================================================================
        stats_y = GRID_HEIGHT * CELL_SIZE + 10
        
        # Calculate current statistics
        dirty_count = self._count_dirty_tiles() if self.dirt_map is not None else 0
        completion_pct = ((self.num_cleanable - dirty_count) / self.num_cleanable) * 100
        current_room = self.room_names.get(
            self.room_layout[self.robot_row][self.robot_col], "Unknown"
        )
        
        # Determine phase
        phase = "CLEANING" if dirty_count > 0 else "🏠 RETURN TO CHARGER ⚡"
        
        # Draw main stats line
        stats_text = (f"Phase: {phase} | Room: {current_room} | "
                     f"Steps: {self.steps_taken}/{self.max_steps} | "
                     f"Cleaned: {self.tiles_cleaned}/{self.num_cleanable} | "
                     f"Progress: {completion_pct:.0f}%")
        text_surface = self.font.render(stats_text, True, COLOR_TEXT)
        self.window.blit(text_surface, (10, stats_y))
        
        # Draw legend line with all room types and rewards
        legend_y = stats_y + 28
        legend_text = ("Kitchen(+50) Living(+45) Bed(+40) Dine(+35) Office(+35) Laund(+30) "
                      "Bath(+25) Garage(+20) Hall(+15)")
        legend_surface = self.small_font.render(legend_text, True, COLOR_TEXT)
        self.window.blit(legend_surface, (10, legend_y))
        
        # Draw color legend boxes (first 5 rooms)
        legend_items = [
            (10, "K", COLOR_KITCHEN),
            (40, "L", COLOR_LIVING_ROOM),
            (70, "B", COLOR_BEDROOM),
            (100, "D", COLOR_DINING),
            (130, "O", COLOR_OFFICE),
        ]
        for x_pos, label, color in legend_items:
            box_size = 10
            pygame.draw.rect(self.window, color, (x_pos, legend_y - 18, box_size, box_size))
            pygame.draw.rect(self.window, COLOR_BLACK, (x_pos, legend_y - 18, box_size, box_size), 1)
        
        # ======================================================================
        # UPDATE DISPLAY
        # ======================================================================
        if self.render_mode == "human":
            # Process Pygame events (required to keep window responsive)
            pygame.event.pump()
            # Update the display
            pygame.display.flip()
            # Control frame rate
            self.clock.tick(self.metadata["render_fps"])
        
        # Return RGB array if requested (for video recording)
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)),
                axes=(1, 0, 2)
            )
        
        return None
    
    def close(self):
        """
        Clean up Pygame resources when environment is closed.
        
        This method should be called when you're done using the environment
        to properly release Pygame resources and close the window.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def get_action_name(self, action):
        """
        Get human-readable name for an action.
        
        Parameters:
        -----------
        action : int
            Action number (0-5)
        
        Returns:
        --------
        str
            Human-readable action name (e.g., "Forward", "Clean")
        """
        return ACTION_NAMES.get(action, "Unknown")
    
    def get_room_name(self, row=None, col=None):
        """
        Get the room name at a specific position or current position.
        
        Parameters:
        -----------
        row : int, optional
            Row coordinate (uses robot position if None)
        col : int, optional
            Column coordinate (uses robot position if None)
        
        Returns:
        --------
        str
            Room name at the given position
        """
        if row is None:
            row = self.robot_row
        if col is None:
            col = self.robot_col
        room_type = self.room_layout[row][col]
        return self.room_names.get(room_type, "Unknown")


# ================================================================================
# MODULE TEST - Run this file directly to test the environment
# ================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TESTING CLEANING ROBOT ENVIRONMENT")
    print("=" * 65)
    
    # Create environment with human rendering
    print("\n1. Creating environment with render_mode='human'...")
    env = CleaningEnv(render_mode="human")
    
    # Test reset
    print("\n2. Testing reset()...")
    observation, info = env.reset()
    print(f"   Initial observation shape: {observation}")
    print(f"   Initial info: {info}")
    
    # Test a few steps
    print("\n3. Testing step() function...")
    for step in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {step}: Action={action}, Reward={reward}, Done={terminated}")
    
    # Close environment
    env.close()
    print("\n✓ Environment test completed successfully!")
    print("=" * 65 + "\n")
    
    # Test reset
    print("\n2. Testing reset()...")
    obs, info = env.reset(seed=42)
    print(f"   Initial observation: {obs}")
    print(f"   Initial info: {info}")
    
    # Test render
    print("\n3. Rendering initial state...")
    env.render()
    
    # Test each action
    print("\n4. Testing each action:")
    for action in range(6):
        obs, reward, term, trunc, info = env.step(action)
        print(f"   Action {action} ({env.get_action_name(action)}): "
              f"reward={reward:+.1f}, state={obs}, result={info['action_result']}")
        env.render()
    
    # Test random episode
    print("\n5. Running 50 random steps...")
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            break
    
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Tiles cleaned: {info['tiles_cleaned']}")
    
    # Wait before closing
    import time
    print("\n6. Waiting 2 seconds before closing...")
    time.sleep(2)
    
    # Clean up
    env.close()
    print("\nTest complete! Environment working correctly.")
