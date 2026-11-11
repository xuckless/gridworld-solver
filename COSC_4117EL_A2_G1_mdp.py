"""MDP formulation of the warehouse task (Part 1.1), built on the provided gridworld framework.

AI use: originally implemented by our group (G1); refactored and bug-fixed with the help of
Claude (Anthropic, Claude Code). Claude-assisted changes in this file: the reward table from
the former reward_system.py was folded into `WarehouseMDP._move`, and the intended-action
probability `p` was made a parameter (`WarehouseMDP.__init__`, `transitions`) for Part 3,
and `states` now skips the two unreachable package-status combinations.

State:  (row, col, has_package, delivered); has_package stays 1 after delivery.
Action: "up", "down", "left", "right".
Transitions are stochastic and known: the intended direction happens with probability p
(0.8 in Parts 1 and 2) and each perpendicular direction with probability (1 - p) / 2.
A movement into a shelf or off the grid leaves the robot in place and costs the invalid-move
penalty. The episode ends on returning to the dock after delivery. The step limit of the
environment is a time cutoff, not part of the MDP.

The provided gridworld.py is used unchanged; slip dynamics live here.
"""

import random
from collections import defaultdict
from typing import Iterator, List, Tuple

from gridworld import (
    ACTION_TO_DELTA,
    ACTIONS,
    CONGESTION,
    CONGESTION_PENALTY,
    DELIVERY_REWARD,
    DOCK_REWARD,
    INVALID_MOVE_PENALTY,
    PICKUP_REWARD,
    SHELF,
    STEP_PENALTY,
    WarehouseGridWorld,
)

State = Tuple[int, int, int, int]
Outcome = Tuple[float, State, float, bool]  # (probability, next_state, reward, done)

DEFAULT_P = 0.8
P_VALUES = (1.0, 0.9, 0.8)  # intended-action probabilities studied in Part 3
PERPENDICULAR = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}
# The three task phases a policy has to cover: (has_package, delivered) -> description.
PHASES = [
    ((0, 0), "Phase 1: heading to pickup (no package)"),
    ((1, 0), "Phase 2: carrying package to packing station"),
    ((1, 1), "Phase 3: delivered, returning to dock"),
]


class WarehouseMDP:
    def __init__(self, world: WarehouseGridWorld, gamma: float = 0.99, p: float = DEFAULT_P):
        self.world = world
        self.gamma = gamma
        self.p = p
        self.actions: List[str] = list(ACTIONS)

    @property
    def start_state(self) -> State:
        row, col = self.world.dock_pos
        return (row, col, 0, 0)

    def is_terminal(self, state: State) -> bool:
        row, col, _, delivered = state
        return (row, col) == self.world.dock_pos and bool(delivered)

    def states(self) -> Iterator[State]:
        """All reachable states: robot on a non-shelf cell with a consistent package status.

        Standing on the pickup cell without the package, or on the packing cell while still
        carrying it, is impossible (entering those cells updates the flags), so they are skipped.
        """
        for row in range(self.world.size):
            for col in range(self.world.size):
                if self.world.grid[row, col] == SHELF:
                    continue
                for has_package, delivered in (phase for phase, _ in PHASES):
                    if (row, col) == self.world.pickup_pos and not has_package:
                        continue
                    if (row, col) == self.world.packing_pos and has_package and not delivered:
                        continue
                    yield (row, col, has_package, delivered)

    def _move(self, state: State, direction: str) -> Tuple[State, float, bool]:
        """Deterministic outcome (next_state, reward, done) of actually moving in `direction`."""
        row, col, has_package, delivered = state
        dr, dc = ACTION_TO_DELTA[direction]
        nr, nc = row + dr, col + dc
        reward = STEP_PENALTY

        if not (self.world.in_bounds(nr, nc) and self.world.passable(nr, nc)):
            nr, nc = row, col
            reward += INVALID_MOVE_PENALTY
        if self.world.grid[nr, nc] == CONGESTION:
            reward += CONGESTION_PENALTY
        if (nr, nc) == self.world.pickup_pos and not has_package:
            has_package = 1
            reward += PICKUP_REWARD
        if (nr, nc) == self.world.packing_pos and has_package and not delivered:
            delivered = 1
            reward += DELIVERY_REWARD
        done = (nr, nc) == self.world.dock_pos and bool(delivered)
        if done:
            reward += DOCK_REWARD
        return (nr, nc, has_package, delivered), reward, done

    def transitions(self, state: State, action: str) -> List[Outcome]:
        """P(s', r | s, a) as a list of outcomes, merging directions that lead to the same result."""
        if self.is_terminal(state):
            raise ValueError("Cannot act in a terminal state.")
        slip = (1.0 - self.p) / 2
        merged = defaultdict(float)
        for direction, prob in ((action, self.p), *((d, slip) for d in PERPENDICULAR[action])):
            if prob > 0:
                merged[self._move(state, direction)] += prob
        return [(prob, ns, reward, done) for (ns, reward, done), prob in merged.items()]

    def step(self, state: State, action: str, rng: random.Random) -> Tuple[State, float, bool]:
        """Sample (next_state, reward, done) from the transition model."""
        outcomes = self.transitions(state, action)
        _, next_state, reward, done = rng.choices(outcomes, weights=[o[0] for o in outcomes])[0]
        return next_state, reward, done
