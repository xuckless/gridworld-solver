"""Value iteration for the warehouse MDP.

V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]
"""

import argparse
from typing import Dict, Tuple

from gridworld import CONGESTION, DOCK, PACKING, PICKUP, SHELF, WarehouseGridWorld
from mdp import State, WarehouseMDP

ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">"}
PHASES = [
    ((0, 0), "Phase 1: heading to pickup (no package)"),
    ((1, 0), "Phase 2: carrying package to packing station"),
    ((1, 1), "Phase 3: delivered, returning to dock"),
]


def q_value(mdp: WarehouseMDP, V: Dict[State, float], state: State, action: str) -> float:
    return sum(
        prob * (reward + (0.0 if done else mdp.gamma * V[next_state]))
        for prob, next_state, reward, done in mdp.transitions(state, action)
    )


def value_iteration(mdp: WarehouseMDP, theta: float = 1e-6) -> Tuple[Dict[State, float], Dict[State, str], int]:
    """Return (V, greedy policy, iterations). Terminal states keep value 0 (no future reward)."""
    V = {s: 0.0 for s in mdp.states()}
    nonterminal = [s for s in V if not mdp.is_terminal(s)]

    iterations = 0
    while True:
        iterations += 1
        delta = 0.0
        new_V = dict(V)
        for s in nonterminal:
            new_V[s] = max(q_value(mdp, V, s, a) for a in mdp.actions)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break

    policy = {s: max(mdp.actions, key=lambda a: q_value(mdp, V, s, a)) for s in nonterminal}
    return V, policy, iterations


def render_policy(mdp: WarehouseMDP, policy: Dict[State, str], has_package: int, delivered: int) -> str:
    """Grid of arrows for one task phase; # shelf, P pickup, D packing, C dock, ! congestion."""
    world = mdp.world
    lines = []
    for r in range(world.size):
        cells = []
        for c in range(world.size):
            cell = int(world.grid[r, c])
            if cell == SHELF:
                cells.append("#")
            elif (r, c, has_package, delivered) in policy:
                arrow = ARROWS[policy[(r, c, has_package, delivered)]]
                tag = {PICKUP: "P", PACKING: "D", DOCK: "C", CONGESTION: "!"}.get(cell, "")
                cells.append(f"{tag}{arrow}")
            else:
                cells.append("C*" if cell == DOCK else "?")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-6)
    args = parser.parse_args()

    world = WarehouseGridWorld(seed=args.seed)
    mdp = WarehouseMDP(world, gamma=args.gamma)
    V, policy, iterations = value_iteration(mdp, args.theta)

    print(f"Seed={args.seed} Pickup={world.pickup_pos} Packing={world.packing_pos} Dock={world.dock_pos}")
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
