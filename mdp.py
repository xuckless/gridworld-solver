"""MDP formulation of the warehouse task.

State:  (row, col, has_package, delivered); has_package stays 1 after delivery.
Action: "up", "down", "left", "right".
Transitions are stochastic and known: the intended direction happens with
probability 0.8 and each perpendicular direction with probability 0.1. A movement
into a shelf or off the grid leaves the robot in place (and incurs the invalid-move
penalty). The episode ends on returning to the dock after delivery.
The step limit of the environment is a time cutoff, not part of the MDP.
"""

import random
from collections import defaultdict
from typing import Iterator, List, Tuple

from gridworld import ACTION_TO_DELTA, ACTIONS, CONGESTION, SHELF, WarehouseGridWorld
from reward_system import TransitionEvents, compute_reward

State = Tuple[int, int, int, int]
Outcome = Tuple[float, State, float, bool]  # (probability, next_state, reward, done)

INTENDED_PROB = 0.8
PERPENDICULAR_PROB = 0.1
PERPENDICULAR = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}


class WarehouseMDP:
    def __init__(self, world: WarehouseGridWorld, gamma: float = 0.99):
        self.world = world
        self.gamma = gamma
        self.actions: List[str] = list(ACTIONS)

    @property
    def start_state(self) -> State:
        row, col = self.world.dock_pos
        return (row, col, 0, 0)

    def is_terminal(self, state: State) -> bool:
        row, col, _, delivered = state
        return (row, col) == self.world.dock_pos and bool(delivered)

    def states(self) -> Iterator[State]:
        """All states with the robot on a non-shelf cell and a consistent package status."""
        size = self.world.size
        for row in range(size):
            for col in range(size):
                if self.world.grid[row, col] == SHELF:
                    continue
                for has_package, delivered in ((0, 0), (1, 0), (1, 1)):
                    yield (row, col, has_package, delivered)

    def _move(self, state: State, direction: str) -> Tuple[State, float, bool]:
        """Outcome of actually moving in `direction` from `state`."""
        row, col, has_package, delivered = state
        dr, dc = ACTION_TO_DELTA[direction]
        nr, nc = row + dr, col + dc

        invalid = not self.world.in_bounds(nr, nc) or not self.world.passable(nr, nc)
        if invalid:
            nr, nc = row, col

        pickup = (nr, nc) == self.world.pickup_pos and not has_package
        delivery = (nr, nc) == self.world.packing_pos and bool(has_package) and not delivered
        has_package = has_package or int(pickup)
        delivered = delivered or int(delivery)
        dock_return = (nr, nc) == self.world.dock_pos and bool(delivered)

        events = TransitionEvents(
            invalid_move=invalid,
            congestion=self.world.grid[nr, nc] == CONGESTION,
            pickup=pickup,
            delivery=delivery,
            dock_return=dock_return,
        )
        return (nr, nc, has_package, delivered), compute_reward(events), dock_return

    def transitions(self, state: State, action: str) -> List[Outcome]:
        """P(s', r | s, a) as a list of outcomes, merging directions that coincide."""
        if self.is_terminal(state):
            raise ValueError("Cannot act in a terminal state.")

        left, right = PERPENDICULAR[action]
        merged = defaultdict(float)
        for direction, prob in (
            (action, INTENDED_PROB),
            (left, PERPENDICULAR_PROB),
            (right, PERPENDICULAR_PROB),
        ):
            merged[self._move(state, direction)] += prob
        return [(prob, ns, reward, done) for (ns, reward, done), prob in merged.items()]

    def step(self, state: State, action: str, rng: random.Random) -> Tuple[State, float, bool]:
        """Sample (next_state, reward, done) from the transition model."""
        outcomes = self.transitions(state, action)
        _, next_state, reward, done = rng.choices(outcomes, weights=[o[0] for o in outcomes])[0]
        return next_state, reward, done
