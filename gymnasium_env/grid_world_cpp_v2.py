from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

#
# Coverage Path Planning (CPP) environment - V2
#
# Key improvements over V1:
#
#   FRONTIER (BFS-based):
#     The frontier observation uses BFS through the grid (respecting obstacles but
#     traversing visited cells) to find the actual shortest path to the nearest
#     unvisited cell. The output is the direction of the FIRST STEP of that path.
#     This directly fixes two failure modes:
#       - Dead-end / corridor trap: Euclidean direction pointed through walls;
#         BFS correctly says "backtrack through visited cells to exit".
#       - Oscillation: BFS gives a deterministic, optimal next-step direction
#         instead of an ambiguous Euclidean hint.
#
#   REWARD (no revisit penalty):
#     The explicit revisit penalty (-0.3) discouraged backtracking, which is
#     sometimes the only way to reach a remaining unvisited cell. It has been
#     removed: revisiting a cell now costs only the step penalty (-0.1), making
#     necessary backtracking inexpensive. The BFS frontier is now the primary
#     navigation signal, so the penalty is no longer needed.
#
# Observation space (Dict):
#   "agent"    : [x/size, y/size, coverage_ratio]   - shape (3,)
#   "neighbors": 3x3 matrix of local view           - shape (3,3)
#                  0 = free/unvisited, 1 = wall/obstacle, 2 = visited
#   "frontier" : [first_dx, first_dy, dist_norm]    - shape (3,)
#                  first step direction of BFS path to nearest unvisited cell
#                  first_dx, first_dy ∈ {-1, 0, +1}; dist_norm ∈ [0, 1]
#
# Reward function:
#   +1.0  visit new cell
#    0.0  revisit already-visited cell (only step penalty applies)
#   -0.5  collision (wall or obstacle)
#   -0.1  step penalty (every action)
#   +10.0 bonus for full coverage
#   -5.0  penalty when max steps reached without full coverage
#

class GridWorldCPPEnvV2(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self._obstacle_set: set = set()
        self.count_steps = 0
        self.max_steps = max_steps

        self.visited: set = set()
        self._reachable_cells: set = set()  # free cells reachable from agent start
        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros((3, 3), dtype=int)

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((3, 3), dtype=np.float32),
                high=np.full((3, 3), 2.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "frontier": gym.spaces.Box(
                low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        # Only count cells reachable from the agent's starting position.
        # Random obstacle placement can create isolated free cells that are
        # physically unreachable; including them would make 100% coverage impossible.
        return len(self._reachable_cells) if self._reachable_cells else max(1, self.size * self.size - len(self.obstacles_locations))

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _compute_reachable(self) -> set:
        """Flood fill from agent start to find all free cells reachable via cardinal moves."""
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        reachable: set = {(ax, ay)}
        queue: deque = deque([(ax, ay)])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and (nx, ny) not in self._obstacle_set
                        and (nx, ny) not in reachable):
                    reachable.add((nx, ny))
                    queue.append((nx, ny))
        return reachable

    def _get_frontier(self) -> np.ndarray:
        """
        BFS from the agent's position to the nearest unvisited free cell.
        Traverses visited cells freely; blocked only by obstacles and grid boundaries.

        Returns [first_dx, first_dy, dist_norm]:
          - first_dx, first_dy: direction of the FIRST STEP of the optimal path
            (each ∈ {-1, 0, +1}).  This correctly handles dead-ends: if backtracking
            through visited cells is required, it points backward.
          - dist_norm: BFS path length normalised to [0, 1].
        """
        if len(self.visited) >= self.total_free_cells:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        queue: deque = deque()
        seen: set = {(ax, ay)}

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in self._obstacle_set:
                seen.add((nx, ny))
                queue.append((nx, ny, dx, dy, 1))

        max_dist = 2 * self.size

        while queue:
            x, y, first_dx, first_dy, dist = queue.popleft()

            if (x, y) not in self.visited:
                return np.array([
                    float(first_dx),
                    float(first_dy),
                    float(min(dist / max_dist, 1.0)),
                ], dtype=np.float32)

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and (nx, ny) not in self._obstacle_set
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    queue.append((nx, ny, first_dx, first_dy, dist + 1))

        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _get_obs(self):
        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
            ], dtype=np.float32),
            "neighbors": self._neighbors.astype(np.float32),
            "frontier": self._get_frontier(),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def _set_neighbors(self):
        matrix = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                nx = self._agent_location[0] + (j - 1)
                ny = self._agent_location[1] + (i - 1)
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    matrix[i][j] = 1
                elif (nx, ny) in self._obstacle_set:
                    matrix[i][j] = 1
                elif (nx, ny) in self.visited:
                    matrix[i][j] = 2
        self._neighbors = matrix

    def _has_accessible_neighbor(self) -> bool:
        """Return True if at least one of the 4 movement directions is reachable."""
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in self._obstacle_set:
                return True
        return False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self._obstacle_set = set()
        self.visited = set()

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        for _ in range(self.obs_quantity):
            loc = self._agent_location.copy()
            while np.array_equal(loc, self._agent_location) or tuple(loc) in self._obstacle_set:
                loc = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(loc)
            self._obstacle_set.add(tuple(loc))

        # Guarantee agent is not completely trapped (all 4 directions blocked).
        # If so, relocate the agent to a free, accessible cell.
        for _ in range(100):
            if self._has_accessible_neighbor():
                break
            new_loc = self.np_random.integers(0, self.size, size=2, dtype=int)
            while tuple(new_loc) in self._obstacle_set:
                new_loc = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._agent_location = new_loc

        # Compute reachable cells from this starting position so that isolated
        # free cells (surrounded by obstacles) are excluded from the coverage goal.
        self._reachable_cells = self._compute_reachable()

        self.visited.add(tuple(self._agent_location))
        self._set_neighbors()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if tuple(self._agent_location) in self._obstacle_set:
            self._agent_location = old_location

        self._set_neighbors()
        self.count_steps += 1

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        reward = -0.1  # step penalty

        if stayed_in_place:
            reward -= 0.5
        elif is_new_cell:
            reward += 1.0
            self.visited.add(current_pos)
        # revisiting a cell costs only the step penalty above — backtracking is free

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage
        if full_coverage:
            reward += 10.0

        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0
        else:
            truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size

        for cell in self.visited:
            pygame.draw.rect(canvas, (144, 238, 144),
                             pygame.Rect(pix * cell[0], pix * cell[1], pix, pix))

        for obs in self.obstacles_locations:
            pygame.draw.rect(canvas, (0, 0, 0),
                             pygame.Rect(pix * obs[0], pix * obs[1], pix, pix))

        pygame.draw.circle(
            canvas, (0, 0, 255),
            ((self._agent_location[0] + 0.5) * pix, (self._agent_location[1] + 0.5) * pix),
            pix / 3,
        )

        font = pygame.font.SysFont(None, 24)
        text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}", True, (0, 0, 0)
        )
        canvas.blit(text, (5, 5))

        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix * x), (self.window_size, pix * x), width=3)
            pygame.draw.line(canvas, 0, (pix * x, 0), (pix * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
