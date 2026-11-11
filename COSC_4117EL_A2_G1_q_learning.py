"""Tabular Q-learning for the warehouse task (Part 2).

AI use: originally implemented by our group (G1); refactored and bug-fixed with the help of
Claude (Anthropic, Claude Code). Claude-assisted changes in this file: `run_episode` (new; shared
by `train`, `evaluate` and the UI, and returns the visited path), the `p` parameter of
`StochasticWarehouseEnv`, and the `--p` option in `main`.

The agent only interacts with `StochasticWarehouseEnv` through reset()/step(); it never reads the
transition probabilities P(s'|s,a). The environment samples the same slip dynamics as the MDP
used by value iteration.

Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
"""

import argparse
import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from gridworld import WarehouseGridWorld
from COSC_4117EL_A2_G1_mdp import DEFAULT_P, PHASES, State, WarehouseMDP
from COSC_4117EL_A2_G1_value_iteration import render_policy


class StochasticWarehouseEnv:
    """Black-box environment: the model lives here, hidden from the learning agent."""

    def __init__(self, world: WarehouseGridWorld, seed: int, max_steps: int = 120, p: float = DEFAULT_P):
        self._mdp = WarehouseMDP(world, p=p)
        self._rng = random.Random(seed)
        self.max_steps = max_steps
        self.actions = self._mdp.actions
        self.state: State = self._mdp.start_state
        self.steps = 0

    def reset(self) -> State:
        self.state = self._mdp.start_state
        self.steps = 0
        return self.state

    def step(self, action: str) -> Tuple[State, float, bool, bool]:
        """Return (next_state, reward, terminated, truncated)."""
        self.state, reward, terminated = self._mdp.step(self.state, action, self._rng)
        self.steps += 1
        return self.state, reward, terminated, (not terminated and self.steps >= self.max_steps)


class QLearningAgent:
    def __init__(self, world: WarehouseGridWorld, actions: List[str], alpha: float, gamma: float,
                 epsilon: float, seed: int):
        self.world = world
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        # One row per (row, col, has_package, delivered) index, one column per action.
        self.Q = np.zeros((world.get_state_space_size(), len(actions)))

    def row(self, state: State) -> np.ndarray:
        return self.Q[self.world.state_to_index(state)]

    def greedy_action(self, state: State) -> str:
        q = self.row(state)
        best = np.flatnonzero(q == q.max())  # break ties randomly
        return self.actions[self.rng.choice(list(best))]

    def act(self, state: State) -> str:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.greedy_action(state)

    def update(self, state: State, action: str, reward: float, next_state: State, terminated: bool) -> None:
        a = self.actions.index(action)
        target = reward if terminated else reward + self.gamma * self.row(next_state).max()
        row = self.row(state)
        row[a] += self.alpha * (target - row[a])

    def policy(self, states: Iterable[State]) -> Dict[State, str]:
        return {s: self.actions[int(np.argmax(self.row(s)))] for s in states}


def run_episode(env: StochasticWarehouseEnv, choose: Callable[[State], str],
                learner: Optional[QLearningAgent] = None) -> Tuple[float, bool, List[Tuple[State, str]]]:
    """Play one episode with `choose(state) -> action`; if `learner` is given, Q-update every step.

    Returns (cumulative reward, success, [(state, action), ..., (final_state, "-")]).
    """
    state = env.reset()
    total, terminated, truncated, path = 0.0, False, False, []
    while not (terminated or truncated):
        action = choose(state)
        next_state, reward, terminated, truncated = env.step(action)
        if learner is not None:
            # Bootstrap through time-limit truncation; only a true terminal has no future.
            learner.update(state, action, reward, next_state, terminated)
        path.append((state, action))
        state, total = next_state, total + reward
    path.append((state, "-"))
    return total, terminated, path  # success = back at the dock after delivery


def train(env: StochasticWarehouseEnv, agent: QLearningAgent, episodes: int,
          epsilon_min: float, epsilon_decay: float) -> Tuple[List[float], List[bool]]:
    """Run Q-learning; return per-episode cumulative reward and task success."""
    rewards, successes = [], []
    for _ in range(episodes):
        total, success, _ = run_episode(env, agent.act, agent)
        rewards.append(total)
        successes.append(success)
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
    return rewards, successes


def evaluate(env: StochasticWarehouseEnv, agent: QLearningAgent) -> Tuple[float, bool, List[Tuple[State, str]]]:
    """One run with exploration disabled (epsilon = 0) and no learning."""
    agent.epsilon = 0.0
    return run_episode(env, agent.greedy_action)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-learning on the warehouse grid world")
    parser.add_argument("--seed", type=int, default=42, help="layout and learning seed")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--p", type=float, default=DEFAULT_P, help="intended-action probability")
    args = parser.parse_args()

    world = WarehouseGridWorld(seed=args.seed)
    env = StochasticWarehouseEnv(world, seed=args.seed, max_steps=world.max_steps, p=args.p)
    agent = QLearningAgent(world, env.actions, args.alpha, args.gamma, args.epsilon_start, args.seed)

    rewards, successes = train(env, agent, args.episodes, args.epsilon_min, args.epsilon_decay)

    print(f"Seed={args.seed} Pickup={world.pickup_pos} Packing={world.packing_pos} Dock={world.dock_pos}")
    print(f"p={args.p} alpha={args.alpha} gamma={args.gamma} epsilon: {args.epsilon_start} -> "
          f"{agent.epsilon:.3f} (min {args.epsilon_min}, decay {args.epsilon_decay}/episode)")
    print(f"Q-table shape: {agent.Q.shape}, episodes: {args.episodes}, max steps/episode: {env.max_steps}")
    block = max(1, args.episodes // 10)
    for start in range(0, args.episodes, block):
        print(f"  episodes {start + 1:>5}-{min(start + block, args.episodes):>5}: "
              f"mean reward {np.mean(rewards[start:start + block]):8.2f}, "
              f"success rate {np.mean(successes[start:start + block]):.2%}")

    mdp = WarehouseMDP(world, gamma=args.gamma, p=args.p)
    policy = agent.policy(s for s in mdp.states() if not mdp.is_terminal(s))
    for (has_package, delivered), title in PHASES:
        print(f"\nGreedy policy - {title}")
        print(render_policy(mdp, policy, has_package, delivered))

    total, success, path = evaluate(env, agent)
    print(f"\nEvaluation run (epsilon = 0): {len(path) - 1} steps, cumulative reward {total:.0f}, success: {success}")
    print(" -> ".join(f"({s[0]},{s[1]})" + (f"[{a}]" if a != "-" else "") for s, a in path))


if __name__ == "__main__":
    main()
