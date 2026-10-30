# -*- coding: utf-8 -*-
"""
COSC-4117EL: Assignment 2 Problem Domain
Warehouse Robot Navigation with Reinforcement Learning

This module provides a warehouse-style grid world environment for Assignment 2.
The robot starts at a charging dock, must pick up one package from a pickup
station, deliver it to the packing station, and then return to the charging dock.

Cell types
----------
0 : empty floor
1 : shelf / wall (blocked)
2 : congestion zone (penalty, but traversable)
3 : pickup station
4 : packing station
5 : charging dock

Suggested state representation for RL:
    (row, col, has_package, delivered)

The environment is intentionally lightweight so you can focus on RL.

Note: The initial code template was generated with the assistance of ChatGPT.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

# Display constants
GRID_SIZE = 10
CELL_SIZE = 56
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE

# Reward constants
PICKUP_REWARD = 25
DELIVERY_REWARD = 40
DOCK_REWARD = 100
CONGESTION_PENALTY = -10
STEP_PENALTY = -1
INVALID_MOVE_PENALTY = -3

# Cell types
EMPTY = 0
SHELF = 1
CONGESTION = 2
PICKUP = 3
PACKING = 4
DOCK = 5

# Colors
EMPTY_COLOR = (245, 245, 245)
SHELF_COLOR = (20, 20, 20)
CONGESTION_COLOR = (225, 60, 60)
PICKUP_COLOR = (70, 130, 255)
PACKING_COLOR = (250, 220, 40)
DOCK_COLOR = (60, 200, 120)
ROBOT_COLOR = (100, 0, 200)
GRID_LINE_COLOR = (200, 200, 200)
TEXT_COLOR = (30, 30, 30)

ACTION_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ACTIONS = list(ACTION_TO_DELTA.keys())


@dataclass
class StepResult:
    state: Tuple[int, int, int, int]
    reward: float
    done: bool
    info: Dict


class WarehouseGridWorld:
    def __init__(self, seed: int = 42, max_steps: int = 120):
        self.size = GRID_SIZE
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.np_random = np.random.default_rng(seed)
        self.seed_value = seed

        self.base_grid = np.array(
            [
                [0, 0, 1, 0, 0, 0, 1, 0, 2, 0],
                [0, 2, 1, 0, 1, 0, 1, 0, 2, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 2, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 2, 1, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 1, 0, 2, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        self.grid = self.base_grid.copy()
        self.place_stations()
        self.reset()

    def place_stations(self) -> None:
        """Place pickup, packing, and dock positions reproducibly using the seed."""
        self.grid = self.base_grid.copy()

        empty_cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r, c] == EMPTY
        ]

        self.pickup_pos = self.random.choice(empty_cells)
        empty_cells.remove(self.pickup_pos)

        self.packing_pos = self.random.choice(empty_cells)
        empty_cells.remove(self.packing_pos)

        self.dock_pos = self.random.choice(empty_cells)

        # Persist markers in the grid
        self.grid[self.pickup_pos] = PICKUP
        self.grid[self.packing_pos] = PACKING
        self.grid[self.dock_pos] = DOCK

    def reset(self) -> Tuple[int, int, int, int]:
        self.robot_pos = self.dock_pos
        self.has_package = False
        self.delivered = False
        self.done = False
        self.steps = 0
        self.score = 0
        return self.get_state()

    def reseed(self, seed: int, regenerate_layout: bool = False) -> None:
        self.random = random.Random(seed)
        self.np_random = np.random.default_rng(seed)
        self.seed_value = seed
        if regenerate_layout:
            self.place_stations()
            self.reset()

    def regenerate_layout(self, seed: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Generate a new seeded layout and reset the episode."""
        if seed is not None:
            self.reseed(seed, regenerate_layout=False)
        self.place_stations()
        return self.reset()

    def get_state(self) -> Tuple[int, int, int, int]:
        return (
            self.robot_pos[0],
            self.robot_pos[1],
            int(self.has_package),
            int(self.delivered),
        )

    def get_state_space_size(self) -> int:
        return self.size * self.size * 2 * 2

    def get_action_space_size(self) -> int:
        return len(ACTIONS)

    def state_to_index(self, state: Tuple[int, int, int, int]) -> int:
        row, col, has_package, delivered = state
        return ((row * self.size + col) * 2 + has_package) * 2 + delivered

    def is_terminal(self) -> bool:
        return self.done

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def passable(self, row: int, col: int) -> bool:
        return self.grid[row, col] != SHELF

    def valid_actions(self, state: Optional[Tuple[int, int, int, int]] = None) -> List[str]:
        if state is None:
            row, col = self.robot_pos
        else:
            row, col = state[0], state[1]

        valid = []
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and self.passable(nr, nc):
                valid.append(action)
        return valid

    def step(self, action: str) -> StepResult:
        if self.done:
            return StepResult(self.get_state(), 0.0, True, {"message": "Episode already finished."})

        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {action}")

        self.steps += 1
        reward = STEP_PENALTY
        info: Dict[str, object] = {"event": "move"}

        row, col = self.robot_pos
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = row + dr, col + dc

        if not self.in_bounds(nr, nc) or not self.passable(nr, nc):
            nr, nc = row, col
            reward += INVALID_MOVE_PENALTY
            info["event"] = "invalid_move"

        self.robot_pos = (nr, nc)
        cell = self.grid[nr, nc]

        if cell == CONGESTION:
            reward += CONGESTION_PENALTY
            info["congestion"] = True

        if (nr, nc) == self.pickup_pos and not self.has_package:
            self.has_package = True
            reward += PICKUP_REWARD
            info["event"] = "pickup"

        if (nr, nc) == self.packing_pos and self.has_package and not self.delivered:
            self.delivered = True
            reward += DELIVERY_REWARD
            info["event"] = "delivery"

        if (nr, nc) == self.dock_pos and self.delivered:
            reward += DOCK_REWARD
            self.done = True
            info["event"] = "return_to_dock"

        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["event"] = "max_steps"

        self.score += reward
        return StepResult(self.get_state(), reward, self.done, info)

    def move(self, direction: str) -> float:
        """Backwards-compatible helper that returns only the reward."""
        result = self.step(direction)
        return result.reward

    def display(self) -> None:
        """Simple text renderer for debugging."""
        symbols = {
            EMPTY: ".",
            SHELF: "#",
            CONGESTION: "!",
            PICKUP: "P",
            PACKING: "D",
            DOCK: "C",
        }
        for i in range(self.size):
            row_symbols = []
            for j in range(self.size):
                if (i, j) == self.robot_pos:
                    row_symbols.append("R")
                else:
                    row_symbols.append(symbols.get(int(self.grid[i, j]), "."))
            print(" ".join(row_symbols))
        print(f"Seed={self.seed_value} State={self.get_state()} Score={self.score} Steps={self.steps}")
        print(f"Pickup={self.pickup_pos} Packing={self.packing_pos} Dock={self.dock_pos}")

    def random_action(self) -> str:
        return self.random.choice(ACTIONS)


def setup_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 56))
    pygame.display.set_caption("Warehouse Grid World")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    small_font = pygame.font.SysFont("arial", 16)
    return screen, clock, font, small_font


def draw_grid(world: WarehouseGridWorld, screen, font, small_font):
    screen.fill((255, 255, 255))

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = int(world.grid[row, col])
            color = EMPTY_COLOR
            if cell == SHELF:
                color = SHELF_COLOR
            elif cell == CONGESTION:
                color = CONGESTION_COLOR
            elif cell == PICKUP:
                color = PICKUP_COLOR
            elif cell == PACKING:
                color = PACKING_COLOR
            elif cell == DOCK:
                color = DOCK_COLOR

            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

            label = None
            if cell == PICKUP:
                label = "P"
            elif cell == PACKING:
                label = "D"
            elif cell == DOCK:
                label = "C"
            elif cell == CONGESTION:
                label = "!"
            if label is not None:
                text = font.render(label, True, TEXT_COLOR if cell != SHELF else (255, 255, 255))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, GRID_LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE * CELL_SIZE))
        pygame.draw.line(screen, GRID_LINE_COLOR, (0, i * CELL_SIZE), (GRID_SIZE * CELL_SIZE, i * CELL_SIZE))

    robot_center = (
        int((world.robot_pos[1] + 0.5) * CELL_SIZE),
        int((world.robot_pos[0] + 0.5) * CELL_SIZE),
    )
    pygame.draw.circle(screen, ROBOT_COLOR, robot_center, int(CELL_SIZE / 3))

    panel_y = GRID_SIZE * CELL_SIZE
    pygame.draw.rect(screen, (245, 247, 250), pygame.Rect(0, panel_y, SCREEN_WIDTH, 56))
    status = (
        f"Seed: {world.seed_value}   "
        f"Score: {world.score:.0f}   "
        f"Steps: {world.steps}/{world.max_steps}   "
        f"Has package: {world.has_package}   "
        f"Delivered: {world.delivered}"
    )
    status_text = small_font.render(status, True, TEXT_COLOR)
    screen.blit(status_text, (10, panel_y + 8))

    help_text = small_font.render("Arrow keys move | R reset | N new layout | ESC quit", True, TEXT_COLOR)
    screen.blit(help_text, (10, panel_y + 30))


def main():
    screen, clock, font, small_font = setup_pygame()
    world = WarehouseGridWorld()
    running = True

    print(f"Initial seeded layout -> Pickup={world.pickup_pos}, Packing={world.packing_pos}, Dock={world.dock_pos}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    world.reset()

                elif event.key == pygame.K_n:
                    world.regenerate_layout(world.seed_value + 1)
                    print(
                        f"New seeded layout (seed={world.seed_value}) -> "
                        f"Pickup={world.pickup_pos}, Packing={world.packing_pos}, Dock={world.dock_pos}"
                    )

                elif event.key == pygame.K_UP:
                    result = world.step("up")
                    print(result)

                elif event.key == pygame.K_DOWN:
                    result = world.step("down")
                    print(result)

                elif event.key == pygame.K_LEFT:
                    result = world.step("left")
                    print(result)

                elif event.key == pygame.K_RIGHT:
                    result = world.step("right")
                    print(result)

        draw_grid(world, screen, font, small_font)
        pygame.display.flip()
        clock.tick(15)

    pygame.quit()


if __name__ == "__main__":
    main()
