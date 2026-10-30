"""Reward function for the warehouse robot task (single source of truth for the reward table)."""

from dataclasses import dataclass

from gridworld import (
    CONGESTION_PENALTY,
    DELIVERY_REWARD,
    DOCK_REWARD,
    INVALID_MOVE_PENALTY,
    PICKUP_REWARD,
    STEP_PENALTY,
)


@dataclass(frozen=True)
class TransitionEvents:
    """Events that can occur on one time step."""

    invalid_move: bool = False  # tried to move into a wall or outside the grid
    congestion: bool = False  # ended the step in a congestion zone
    pickup: bool = False  # picked up the package at the pickup station
    delivery: bool = False  # delivered the package at the packing station while carrying
    dock_return: bool = False  # returned to the charging dock after delivery


def compute_reward(events: TransitionEvents) -> float:
    """Sum the rewards of every event in a step; the time-step penalty always applies."""
    reward = STEP_PENALTY
    if events.invalid_move:
        reward += INVALID_MOVE_PENALTY
    if events.congestion:
        reward += CONGESTION_PENALTY
    if events.pickup:
        reward += PICKUP_REWARD
    if events.delivery:
        reward += DELIVERY_REWARD
    if events.dock_return:
        reward += DOCK_REWARD
    return reward
