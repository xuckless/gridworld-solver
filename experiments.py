"""Q-learning hyperparameter / seed study and comparison with the value-iteration policy.

Saves all figures to plots/ and prints the numbers quoted in plots/comparison.md.
"""

import argparse
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import CONGESTION, DOCK, PACKING, PICKUP, SHELF, WarehouseGridWorld
from mdp import State, WarehouseMDP
from q_learning import QLearningAgent, StochasticWarehouseEnv, train
from value_iteration import PHASES, q_value, value_iteration

PLOT_DIR = "plots"
EVAL_GAMMA = 0.99  # every policy is judged under the task discount, whatever gamma it was trained with
SMOOTH = 100
ARROW_VEC = {"up": (0, 0.35), "down": (0, -0.35), "left": (-0.35, 0), "right": (0.35, 0)}

CONFIGS = {
    "baseline (a=0.1, g=0.99, decay=0.999)": dict(alpha=0.1, gamma=0.99, decay=0.999),
    "high alpha (a=0.5)": dict(alpha=0.5, gamma=0.99, decay=0.999),
    "low gamma (g=0.9)": dict(alpha=0.1, gamma=0.9, decay=0.999),
    "fast decay (0.995)": dict(alpha=0.1, gamma=0.99, decay=0.995),
    "slow decay (0.9995)": dict(alpha=0.1, gamma=0.99, decay=0.9995),
}


def moving_average(x, window=SMOOTH):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def convergence_episode(rewards, tolerance=0.05, window=SMOOTH):
    """First episode whose moving average is within `tolerance` of the final level (last 500 episodes)."""
    smooth = moving_average(rewards, window)
    final = np.mean(rewards[-500:])
    close = np.abs(smooth - final) <= tolerance * abs(final)
    # require it to stay close from then on
    stays = np.flip(np.cumprod(np.flip(close)))
    idx = np.flatnonzero(stays)
    return int(idx[0]) + window if len(idx) else None


def evaluate_policy(mdp: WarehouseMDP, policy: Dict[State, str], theta=1e-8) -> Dict[State, float]:
    """Exact V^pi under the true model (analysis only; the learner never sees this)."""
    nonterminal = [s for s in mdp.states() if not mdp.is_terminal(s)]
    V = {s: 0.0 for s in mdp.states()}
    while True:
        delta = 0.0
        for s in nonterminal:
            v = q_value(mdp, V, s, policy[s])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            return V


def visit_weights(mdp: WarehouseMDP, policy: Dict[State, str], horizon: int) -> Dict[State, float]:
    """Expected visits per state over one episode when following `policy` from the start."""
    dist = {mdp.start_state: 1.0}
    visits: Dict[State, float] = {}
    for _ in range(horizon):
        nxt: Dict[State, float] = {}
        for s, p in dist.items():
            visits[s] = visits.get(s, 0.0) + p
            if mdp.is_terminal(s):
                continue
            for prob, ns, _, _ in mdp.transitions(s, policy[s]):
                nxt[ns] = nxt.get(ns, 0.0) + p * prob
        dist = nxt
    return {s: v for s, v in visits.items() if not mdp.is_terminal(s)}


def run_q_learning(world, cfg, seed, episodes):
    env = StochasticWarehouseEnv(world, seed=seed, max_steps=world.max_steps)
    agent = QLearningAgent(world, env.actions, cfg["alpha"], cfg["gamma"], 1.0, seed)
    rewards, successes = train(env, agent, episodes, 0.05, cfg["decay"])
    return agent, np.array(rewards), np.array(successes, dtype=float)


def plot_reward_curves(results, episodes, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, runs in results.items():
        mean_curve = np.mean([moving_average(r["rewards"]) for r in runs], axis=0)
        std_curve = np.std([moving_average(r["rewards"]) for r in runs], axis=0)
        x = np.arange(SMOOTH, episodes + 1)
        axes[0].plot(x, mean_curve, label=name)
        axes[0].fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
        axes[1].plot(x, np.mean([moving_average(r["successes"]) for r in runs], axis=0), label=name)
    axes[0].set(title=f"Cumulative reward per episode ({SMOOTH}-episode moving avg, mean ± std over seeds)",
                xlabel="episode", ylabel="cumulative reward")
    axes[1].set(title="Task success rate (moving avg, mean over seeds)", xlabel="episode",
                ylabel="success rate", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_seed_curves(runs, name, episodes, v_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(runs):
        ax.plot(r["rewards"], alpha=0.12, color=f"C{i}")
        ax.plot(np.arange(SMOOTH, episodes + 1), moving_average(r["rewards"]), color=f"C{i}", label=f"seed {r['seed']}")
    ax.axhline(v_star, color="k", ls="--", lw=1, label="V*(start) from value iteration")
    ax.set(title=f"Raw (faint) and smoothed reward per episode, {name}", xlabel="episode", ylabel="cumulative reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_policy(ax, world, policy, title, other=None):
    """Arrow plot of one phase; cells where `other` disagrees are shaded."""
    for r in range(world.size):
        for c in range(world.size):
            cell = int(world.grid[r, c])
            color = {SHELF: "#141414", CONGESTION: "#e13c3c", PICKUP: "#4682ff",
                     PACKING: "#fadc28", DOCK: "#3cc878"}.get(cell, "#f5f5f5")
            ax.add_patch(plt.Rectangle((c, world.size - 1 - r), 1, 1, facecolor=color, edgecolor="#c8c8c8"))
    for (r, c), a in policy.items():
        dx, dy = ARROW_VEC[a]
        x, y = c + 0.5 - dx / 2, world.size - 1 - r + 0.5 - dy / 2
        ax.arrow(x, y, dx, dy, head_width=0.18, length_includes_head=True, color="k" if other is None else other[(r, c)])
    ax.set(xlim=(0, world.size), ylim=(0, world.size), aspect="equal", title=title)
    ax.axis("off")


def plot_policy_comparison(mdp, vi_policy, q_policy, q_equiv, path):
    world = mdp.world
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, ((hp, dl), title) in enumerate(PHASES):
        sel = lambda p: {(s[0], s[1]): a for s, a in p.items() if (s[2], s[3]) == (hp, dl)}
        vi_cells, q_cells = sel(vi_policy), sel(q_policy)
        draw_policy(axes[0, col], world, vi_cells, f"Value iteration\n{title}")
        colors = {}
        for s, a in q_policy.items():
            if (s[2], s[3]) == (hp, dl):
                colors[(s[0], s[1])] = "k" if a == vi_policy[s] else ("tab:green" if q_equiv[s] else "tab:red")
        draw_policy(axes[1, col], world, q_cells, f"Q-learning (black = same as VI, green = equally optimal,\nred = suboptimal)\n{title}", colors)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()
    os.makedirs(PLOT_DIR, exist_ok=True)

    world = WarehouseGridWorld(seed=args.layout_seed)
    mdp = WarehouseMDP(world, gamma=EVAL_GAMMA)
    V_star, vi_policy, vi_iters = value_iteration(mdp, 1e-6)
    v_start_star = V_star[mdp.start_state]
    q_star = {s: {a: q_value(mdp, V_star, s, a) for a in mdp.actions} for s in vi_policy}
    weights = visit_weights(mdp, vi_policy, world.max_steps)
    on_path = [s for s, w in weights.items() if w >= 0.01]
    print(f"layout seed {args.layout_seed}: VI converged in {vi_iters} iterations, V*(start)={v_start_star:.2f}, "
          f"{len(vi_policy)} non-terminal states, {len(on_path)} frequently visited under the VI policy")

    results: Dict[str, List[dict]] = {}
    policies = {}
    for name, cfg in CONFIGS.items():
        results[name] = []
        for seed in args.seeds:
            agent, rewards, successes = run_q_learning(world, cfg, seed, args.episodes)
            policy = {s: agent.actions[int(np.argmax(agent._row(s)))] for s in vi_policy}
            V_pi = evaluate_policy(mdp, policy)
            equiv = {s: q_star[s][policy[s]] >= max(q_star[s].values()) - 1e-6 for s in vi_policy}
            run = dict(
                seed=seed, rewards=rewards, successes=successes, policy=policy, equiv=equiv,
                conv=convergence_episode(rewards),
                final_reward=rewards[-500:].mean(), final_success=successes[-500:].mean(),
                agree_all=np.mean([policy[s] == vi_policy[s] for s in vi_policy]),
                agree_path=np.mean([policy[s] == vi_policy[s] for s in on_path]),
                equiv_path=np.mean([equiv[s] for s in on_path]),
                v_start=V_pi[mdp.start_state],
            )
            results[name].append(run)
            policies[(name, seed)] = (policy, equiv)
            print(f"  {name} seed {seed}: final reward {run['final_reward']:.1f}, success {run['final_success']:.1%}, "
                  f"conv ep {run['conv']}, V^pi(start) {run['v_start']:.2f}, "
                  f"agree(all) {run['agree_all']:.1%}, agree(path) {run['agree_path']:.1%}, "
                  f"equiv(path) {run['equiv_path']:.1%}")

    print("\n| config | final reward (last 500) | success | convergence ep. | V^pi(start) | agree path | optimal-equiv path |")
    for name, runs in results.items():
        conv = [r["conv"] for r in runs if r["conv"] is not None]
        f = lambda key, pct=False: (f"{np.mean([r[key] for r in runs]) * (100 if pct else 1):.1f} ± "
                                    f"{np.std([r[key] for r in runs]) * (100 if pct else 1):.1f}")
        print(f"| {name} | {f('final_reward')} | {f('final_success', True)}% | "
              f"{np.mean(conv):.0f} ± {np.std(conv):.0f} (range {min(conv)}-{max(conv)}) | {f('v_start')} | "
              f"{f('agree_path', True)}% | {f('equiv_path', True)}% |")

    plot_reward_curves(results, args.episodes, os.path.join(PLOT_DIR, "reward_vs_episode_configs.png"))
    base = next(iter(results))
    plot_seed_curves(results[base], base, args.episodes, v_start_star, os.path.join(PLOT_DIR, "reward_vs_episode_seeds.png"))
    best_seed = max(results[base], key=lambda r: r["v_start"])["seed"]
    q_policy, q_equiv = policies[(base, best_seed)]
    plot_policy_comparison(mdp, vi_policy, q_policy, q_equiv, os.path.join(PLOT_DIR, "policy_comparison.png"))
    print(f"\npolicy figure uses the baseline run with seed {best_seed}; V*(start)={v_start_star:.2f}")


if __name__ == "__main__":
    main()
