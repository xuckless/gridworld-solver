"""Value iteration for the warehouse MDP (Part 1.2).

AI use: originally implemented by our group (G1); refactored and bug-fixed with the help of
Claude (Anthropic, Claude Code). Claude-assisted changes in this file: `value_iteration_sweeps`
(new generator yielding one Bellman sweep at a time, used by the UI animation),
`greedy_policy` (split out of `value_iteration`), `optimal_actions`, and the `--p` option in `main`.

V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]
"""

import argparse
from typing import Dict, Iterator, Tuple

from gridworld import CONGESTION, DOCK, PACKING, PICKUP, SHELF, WarehouseGridWorld
from COSC_4117EL_A2_G1_mdp import DEFAULT_P, PHASES, State, WarehouseMDP

ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">"}
Values = Dict[State, float]
Policy = Dict[State, str]


def q_value(mdp: WarehouseMDP, V: Values, state: State, action: str) -> float:
    return sum(
        prob * (reward + (0.0 if done else mdp.gamma * V[next_state]))
        for prob, next_state, reward, done in mdp.transitions(state, action)
    )


def optimal_actions(mdp: WarehouseMDP, V: Values, state: State, tol: float = 1e-6) -> set:
    """All actions whose Q-value is within `tol` of the best one (ties are common when p = 1)."""
    q = {a: q_value(mdp, V, state, a) for a in mdp.actions}
    best = max(q.values())
    return {a for a, v in q.items() if v >= best - tol}


def greedy_policy(mdp: WarehouseMDP, V: Values) -> Policy:
    """argmax_a Q(s, a) for every non-terminal state."""
    return {
        s: max(mdp.actions, key=lambda a: q_value(mdp, V, s, a))
        for s in V
        if not mdp.is_terminal(s)
    }


def value_iteration_sweeps(mdp: WarehouseMDP, theta: float = 1e-6) -> Iterator[Tuple[Values, Values, float]]:
    """Yield (V_k+1, V_k, delta) after each synchronous Bellman sweep; stops once delta < theta.

    V is initialised to 0 for every valid state. Terminal states stay at 0 (no future reward).
    """
    V = {s: 0.0 for s in mdp.states()}
    nonterminal = [s for s in V if not mdp.is_terminal(s)]
    while True:
        new_V = dict(V)
        for s in nonterminal:
            new_V[s] = max(q_value(mdp, V, s, a) for a in mdp.actions)
        delta = max(abs(new_V[s] - V[s]) for s in nonterminal)
        yield new_V, V, delta
        V = new_V
        if delta < theta:
            return


def value_iteration(mdp: WarehouseMDP, theta: float = 1e-6) -> Tuple[Values, Policy, int]:
    """Return (V*, greedy policy, number of sweeps until convergence)."""
    iterations = 0
    for V, _, _ in value_iteration_sweeps(mdp, theta):
        iterations += 1
    return V, greedy_policy(mdp, V), iterations


def render_policy(mdp: WarehouseMDP, policy: Policy, has_package: int, delivered: int) -> str:
    """Text grid of arrows for one task phase; # shelf, P pickup, D packing, C dock, ! congestion."""
    world = mdp.world
    tags = {PICKUP: "P", PACKING: "D", DOCK: "C", CONGESTION: "!"}
    lines = []
    for r in range(world.size):
        cells = []
        for c in range(world.size):
            cell = int(world.grid[r, c])
            state = (r, c, has_package, delivered)
            if cell == SHELF:
                cells.append("#")
            elif state in policy:
                cells.append(tags.get(cell, "") + ARROWS[policy[state]])
            else:
                cells.append("C*")  # terminal: back at the dock after delivery
        lines.append(" ".join(f"{x:>2}" for x in cells))
    return "\n".join(lines)


EXPLANATION = (
    "Why a known model is required: the Bellman optimality update sums over every possible next\n"
    "state s' weighted by P(s'|s,a) and uses R(s,a,s'). Value iteration therefore has to look up\n"
    "these probabilities and rewards for every state-action pair (a planning method); it never\n"
    "interacts with the environment to sample experience, so without the model it cannot be run."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Value iteration on the warehouse grid world")
    parser.add_argument("--seed", type=int, default=42, help="layout seed")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-6)
    parser.add_argument("--p", type=float, default=DEFAULT_P, help="intended-action probability")
    args = parser.parse_args()

    world = WarehouseGridWorld(seed=args.seed)
    mdp = WarehouseMDP(world, gamma=args.gamma, p=args.p)
    V, policy, iterations = value_iteration(mdp, args.theta)

    print(f"Seed={args.seed} Pickup={world.pickup_pos} Packing={world.packing_pos} Dock={world.dock_pos}")
    print(f"Intended-action probability p: {mdp.p}")
    print(f"Discount factor (gamma): {mdp.gamma}")
    print(f"Convergence threshold (theta): {args.theta}")
    print(f"Iterations to converge: {iterations}")
    print(f"V(start state) = {V[mdp.start_state]:.3f}")
    for (has_package, delivered), title in PHASES:
        print(f"\n{title}")
        print(render_policy(mdp, policy, has_package, delivered))
    print(f"\n{EXPLANATION}")


if __name__ == "__main__":
    main()
