"""
Lane Dodge - game logic only (no graphics).

The SAME class is used for:
  * manual play   (play.py)
  * RL training   (DQN, next step)

Grid layout (rows x cols). Row 0 = top, last row = car row:

    col: 0 1 2 3 4
         . . X . .   row 0        <- new blocks spawn here
         . . . . .
         . X . . .
         . C . . .   row rows-1   <- car lives here

One step:
  1. car moves (LEFT / STAY / RIGHT), it cannot leave the road
     -> if it moves INTO a lane where a block is beside it -> side crash
  2. every block falls one row
  3. a new top row spawns: one block in a random column, or empty
  4. if a block lands on the car -> front crash

Spawn rule: at most one block per row, and never in a column right next to the
previous row's block. That removes "staircase" traps, so there is ALWAYS a safe move.
"""
from __future__ import annotations

import numpy as np

LEFT, STAY, RIGHT = 0, 1, 2
ACTION_NAMES = ("LEFT", "STAY", "RIGHT")


class LaneEnv:
    def __init__(
        self,
        cols: int = 5,
        rows: int = 9,
        spawn_prob: float = 0.6,
        max_steps: int | None = 1000,
        reward_survive: float = 1.0,
        reward_crash: float = -10.0,
        reward_move: float = 0.0,
        seed: int | None = None,
    ):
        if cols < 2:
            raise ValueError("cols must be >= 2")
        if rows < 3:
            raise ValueError("rows must be >= 3")
        self.cols = cols
        self.rows = rows
        self.spawn_prob = spawn_prob
        self.max_steps = max_steps
        self.reward_survive = reward_survive
        self.reward_crash = reward_crash
        self.reward_move = reward_move

        self.n_actions = 3
        # what the agent sees: every cell (incl. the car row, for side hits) + one-hot car column
        self.obs_size = rows * cols + cols

        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.exited = np.zeros(self.cols, dtype=np.int8)  # row that just left the screen (for drawing)
        self.car = self.cols // 2
        self.steps = 0
        self.dodged = 0
        self.done = False
        self.crashed = False
        self.crash_type = None      # "side" or "front"
        self.last_shifted = False   # did blocks fall on the last step? (for drawing)
        return self.get_obs()

    def step(self, action: int):
        """Returns (obs, reward, done, info) - same idea as Gym/Gymnasium."""
        if self.done:
            raise RuntimeError("Episode is over - call reset() first.")
        if action not in (LEFT, STAY, RIGHT):
            raise ValueError(f"invalid action {action}")

        # 1. move car (walls stop it)
        old_car = self.car
        if action == LEFT:
            self.car = max(0, self.car - 1)
        elif action == RIGHT:
            self.car = min(self.cols - 1, self.car + 1)
        moved = self.car != old_car
        self.steps += 1

        if moved and self.grid[-1, self.car]:
            # side crash: drove into a block that is still beside the car
            self.crashed, self.crash_type = True, "side"
            self.last_shifted = False
            self.exited = np.zeros(self.cols, dtype=np.int8)
        else:
            # 2. blocks fall one row; bottom row leaves the screen
            self.exited = self.grid[-1].copy()
            self.dodged += int(self.exited.sum())
            self.grid = np.roll(self.grid, 1, axis=0)
            self.last_shifted = True

            # 3. spawn new top row (grid[1] is the previous top row)
            self.grid[0] = self._spawn_row(self.grid[1])

            # 4. front crash: a block landed on the car
            self.crashed = bool(self.grid[-1, self.car])
            self.crash_type = "front" if self.crashed else None
        truncated = False
        if self.crashed:
            reward = self.reward_crash
            self.done = True
        else:
            reward = self.reward_survive + (self.reward_move if moved else 0.0)
            if self.max_steps is not None and self.steps >= self.max_steps:
                self.done = True
                truncated = True  # time limit, NOT a real ending (matters for DQN later)

        info = {
            "crashed": self.crashed,
            "crash_type": self.crash_type,
            "truncated": truncated,
            "steps": self.steps,
            "dodged": self.dodged,
            "moved": moved,
        }
        return self.get_obs(), reward, self.done, info

    def get_obs(self) -> np.ndarray:
        """Flat float vector: rows*cols block cells (1 = block) + cols one-hot car."""
        blocks = self.grid.astype(np.float32).ravel()
        car = np.zeros(self.cols, dtype=np.float32)
        car[self.car] = 1.0
        return np.concatenate([blocks, car])

    def render_text(self) -> str:
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if r == self.rows - 1 and c == self.car:
                    row.append("*" if self.grid[r, c] else "C")
                else:
                    row.append("X" if self.grid[r, c] else ".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------- helpers
    def _spawn_row(self, prev_row: np.ndarray) -> np.ndarray:
        row = np.zeros(self.cols, dtype=np.int8)
        if self.rng.random() < self.spawn_prob:
            prev = np.flatnonzero(prev_row)
            allowed = [c for c in range(self.cols) if not any(abs(c - p) == 1 for p in prev)]
            row[allowed[self.rng.integers(len(allowed))]] = 1
        return row


# ---------------------------------------------------------------- self-test
def _heuristic(env: LaneEnv) -> int:
    """Perfect rule: if a block is about to land on me, step to a side lane that is free."""
    c = env.car
    if env.grid[-2, c]:
        if c > 0 and not env.grid[-1, c - 1]:
            return LEFT
        if c < env.cols - 1 and not env.grid[-1, c + 1]:
            return RIGHT
    return STAY


def _test_side_crash() -> None:
    env = LaneEnv(seed=0)
    env.grid[-1, env.car + 1] = 1          # block right beside the car
    _, r, done, info = env.step(RIGHT)
    assert done and info["crash_type"] == "side" and r == env.reward_crash, info
    env.reset()
    env.grid[-1, env.car + 1] = 1
    _, _, done, info = env.step(STAY)      # staying next to it is fine
    assert not done, info
    env.reset()
    env.grid[-2, env.car] = 1              # block about to land on the car
    _, _, done, info = env.step(STAY)
    assert done and info["crash_type"] == "front", info
    print("side/front crash rules: OK")


if __name__ == "__main__":
    env = LaneEnv(seed=0)
    print("Example state:\n" + env.render_text())
    print("obs size:", env.obs_size)
    _test_side_crash()

    for name, policy in [
        ("random", lambda e: int(e.rng.integers(3))),
        ("heuristic", _heuristic),
    ]:
        lengths = []
        for ep in range(200):
            env.reset(seed=ep)
            done = False
            while not done:
                _, _, done, info = env.step(policy(env))
            lengths.append(info["steps"])
        print(f"{name:>9}: avg steps survived = {np.mean(lengths):7.1f}  (max {env.max_steps})")
