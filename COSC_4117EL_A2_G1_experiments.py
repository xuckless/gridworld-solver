"""Experiments for Parts 2 and 3; all figures are saved to plots/.

AI use: originally implemented by our group (G1); refactored and bug-fixed with the help of
Claude (Anthropic, Claude Code). Claude-assisted changes in this file: `part2` (the former main
body, with a fix for a crash when no run reaches the convergence criterion), and the whole of
Part 3: `simulate`, `changed_states`, `plot_p_policies`, `plot_p_curves`, `part3`.

Part 2: Q-learning hyperparameter / seed study and comparison with the value-iteration policy.
Part 3: value iteration vs Q-learning for intended-action probabilities p = 1.00, 0.90, 0.80,
        and the states whose preferred action changes as p decreases.

Usage: python COSC_4117EL_A2_G1_experiments.py [--part 2|3|all] [--seeds 0 1 2 3 4]
"""

import argparse
import os
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import CONGESTION, DOCK, PACKING, PICKUP, SHELF, WarehouseGridWorld
from COSC_4117EL_A2_G1_mdp import P_VALUES, PHASES, State, WarehouseMDP
from COSC_4117EL_A2_G1_q_learning import QLearningAgent, StochasticWarehouseEnv, run_episode, train
from COSC_4117EL_A2_G1_value_iteration import optimal_actions, q_value, value_iteration

PLOT_DIR = "plots"
GAMMA = 0.99  # every policy is judged under the task discount, whatever gamma it was trained with
THETA = 1e-6
SMOOTH = 100
EVAL_EPISODES = 300  # Monte-Carlo episodes per policy evaluation in Part 3
ARROW_VEC = {"up": (0, 0.35), "down": (0, -0.35), "left": (-0.35, 0), "right": (0.35, 0)}
CELL_COLORS = {SHELF: "#141414", CONGESTION: "#e13c3c", PICKUP: "#4682ff", PACKING: "#fadc28", DOCK: "#3cc878"}

CONFIGS = {
    "baseline (a=0.1, g=0.99, decay=0.999)": dict(alpha=0.1, gamma=0.99, decay=0.999),
    "high alpha (a=0.5)": dict(alpha=0.5, gamma=0.99, decay=0.999),
    "low gamma (g=0.9)": dict(alpha=0.1, gamma=0.9, decay=0.999),
    "fast decay (0.995)": dict(alpha=0.1, gamma=0.99, decay=0.995),
    "slow decay (0.9995)": dict(alpha=0.1, gamma=0.99, decay=0.9995),
}
BASELINE = next(iter(CONFIGS))


# ----------------------------------------------------------------------------- shared helpers
def moving_average(x, window=SMOOTH):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def convergence_episode(rewards, tolerance=0.05, window=SMOOTH) -> Optional[int]:
    """First episode after which the moving average stays within `tolerance` of the final level."""
    smooth = moving_average(rewards, window)
    final = np.mean(rewards[-500:])
    close = np.abs(smooth - final) <= tolerance * abs(final)
    stays = np.flip(np.cumprod(np.flip(close)))  # True only if every later point is close too
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


def run_q_learning(world, cfg, seed, episodes, p=0.8):
    env = StochasticWarehouseEnv(world, seed=seed, max_steps=world.max_steps, p=p)
    agent = QLearningAgent(world, env.actions, cfg["alpha"], cfg["gamma"], 1.0, seed)
    rewards, successes = train(env, agent, episodes, 0.05, cfg["decay"])
    return agent, np.array(rewards), np.array(successes, dtype=float)


def draw_policy(ax, world, policy, title, colors=None, highlight=()):
    """Arrow plot of one phase. `colors` maps cell -> arrow colour; `highlight` cells get an orange frame."""
    for r in range(world.size):
        for c in range(world.size):
            face = CELL_COLORS.get(int(world.grid[r, c]), "#f5f5f5")
            ax.add_patch(plt.Rectangle((c, world.size - 1 - r), 1, 1, facecolor=face, edgecolor="#c8c8c8"))
    for r, c in highlight:
        ax.add_patch(plt.Rectangle((c + 0.04, world.size - 1 - r + 0.04), 0.92, 0.92,
                                   fill=False, edgecolor="#ff8c00", lw=2.5))
    for (r, c), a in policy.items():
        dx, dy = ARROW_VEC[a]
        x, y = c + 0.5 - dx / 2, world.size - 1 - r + 0.5 - dy / 2
        ax.arrow(x, y, dx, dy, head_width=0.18, length_includes_head=True,
                 color="k" if colors is None else colors[(r, c)])
    ax.set(xlim=(0, world.size), ylim=(0, world.size), aspect="equal", title=title)
    ax.axis("off")


def phase_cells(policy: Dict[State, str], phase) -> Dict[tuple, str]:
    return {(s[0], s[1]): a for s, a in policy.items() if (s[2], s[3]) == phase}


# ----------------------------------------------------------------------------- Part 2
def plot_reward_curves(results, episodes, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(SMOOTH, episodes + 1)
    for name, runs in results.items():
        curves = np.array([moving_average(r["rewards"]) for r in runs])
        mean_curve, std_curve = curves.mean(axis=0), curves.std(axis=0)
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


def plot_policy_comparison(mdp, vi_policy, q_policy, q_equiv, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, (phase, title) in enumerate(PHASES):
        draw_policy(axes[0, col], mdp.world, phase_cells(vi_policy, phase), f"Value iteration\n{title}")
        colors = {(s[0], s[1]): "k" if a == vi_policy[s] else ("tab:green" if q_equiv[s] else "tab:red")
                  for s, a in q_policy.items() if (s[2], s[3]) == phase}
        draw_policy(axes[1, col], mdp.world, phase_cells(q_policy, phase),
                    f"Q-learning (black = same as VI, green = equally optimal,\nred = suboptimal)\n{title}", colors)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt_mean_std(values, scale=1.0) -> str:
    return f"{np.mean(values) * scale:.1f} ± {np.std(values) * scale:.1f}"


def part2(world: WarehouseGridWorld, seeds: List[int], episodes: int) -> None:
    print("=" * 30, "PART 2: Q-learning experiments (p = 0.8)", "=" * 30)
    mdp = WarehouseMDP(world, gamma=GAMMA)
    V_star, vi_policy, vi_iters = value_iteration(mdp, THETA)
    v_start_star = V_star[mdp.start_state]
    opt = {s: optimal_actions(mdp, V_star, s) for s in vi_policy}
    weights = visit_weights(mdp, vi_policy, world.max_steps)
    on_path = [s for s, w in weights.items() if w >= 0.01]
    print(f"VI converged in {vi_iters} iterations, V*(start)={v_start_star:.2f}, "
          f"{len(vi_policy)} non-terminal states, {len(on_path)} frequently visited under the VI policy")

    results: Dict[str, List[dict]] = {}
    for name, cfg in CONFIGS.items():
        results[name] = []
        for seed in seeds:
            agent, rewards, successes = run_q_learning(world, cfg, seed, episodes)
            policy = agent.policy(vi_policy)
            equiv = {s: policy[s] in opt[s] for s in vi_policy}
            run = dict(
                seed=seed, rewards=rewards, successes=successes, policy=policy, equiv=equiv,
                conv=convergence_episode(rewards),
                final_reward=rewards[-500:].mean(), final_success=successes[-500:].mean(),
                agree_path=np.mean([policy[s] == vi_policy[s] for s in on_path]),
                equiv_path=np.mean([equiv[s] for s in on_path]),
                v_start=evaluate_policy(mdp, policy)[mdp.start_state],
            )
            results[name].append(run)
            print(f"  {name} seed {seed}: final reward {run['final_reward']:.1f}, success {run['final_success']:.1%}, "
                  f"conv ep {run['conv']}, V^pi(start) {run['v_start']:.2f}, "
                  f"agree(path) {run['agree_path']:.1%}, equiv(path) {run['equiv_path']:.1%}")

    print("\n| config | final reward (last 500) | success | convergence ep. | V^pi(start) | agree path | optimal-equiv path |")
    for name, runs in results.items():
        col = lambda key, scale=1.0: fmt_mean_std([r[key] for r in runs], scale)
        conv = [r["conv"] for r in runs if r["conv"] is not None]
        conv_txt = (f"{np.mean(conv):.0f} ± {np.std(conv):.0f} (range {min(conv)}-{max(conv)}, "
                    f"{len(conv)}/{len(runs)} runs)" if conv else "not reached")
        print(f"| {name} | {col('final_reward')} | {col('final_success', 100)}% | {conv_txt} | {col('v_start')} | "
              f"{col('agree_path', 100)}% | {col('equiv_path', 100)}% |")

    plot_reward_curves(results, episodes, os.path.join(PLOT_DIR, "reward_vs_episode_configs.png"))
    plot_seed_curves(results[BASELINE], BASELINE, episodes, v_start_star,
                     os.path.join(PLOT_DIR, "reward_vs_episode_seeds.png"))
    best = max(results[BASELINE], key=lambda r: r["v_start"])
    plot_policy_comparison(mdp, vi_policy, best["policy"], best["equiv"], os.path.join(PLOT_DIR, "policy_comparison.png"))
    print(f"\npolicy figure uses the baseline run with seed {best['seed']}; V*(start)={v_start_star:.2f}")


# ----------------------------------------------------------------------------- Part 3
def simulate(world, choose: Callable[[State], str], p: float, seed: int, n: int = EVAL_EPISODES):
    """Monte-Carlo (mean cumulative reward, success rate) of a fixed policy under slip probability p."""
    env = StochasticWarehouseEnv(world, seed=seed, max_steps=world.max_steps, p=p)
    runs = [run_episode(env, choose)[:2] for _ in range(n)]
    return float(np.mean([r for r, _ in runs])), float(np.mean([s for _, s in runs]))


def near_congestion(world, r, c) -> bool:
    return any(world.in_bounds(r + dr, c + dc) and world.grid[r + dr, c + dc] == CONGESTION
               for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))


def changed_states(ref_policy, mdp, V, policy, weights) -> List[dict]:
    """States where the action VI prefers at p = 1.00 is no longer optimal at this p.

    Ties are common when p = 1 (many equally short routes), so a state only counts as changed when
    the old action is strictly worse under the new model, not merely tie-broken differently.
    """
    out = []
    for s, a in policy.items():
        ref_a = ref_policy[s]
        if ref_a in optimal_actions(mdp, V, s):
            continue
        out.append(dict(state=s, action=a, ref_action=ref_a, visits=weights.get(s, 0.0),
                        congestion=near_congestion(mdp.world, s[0], s[1]),
                        gain=q_value(mdp, V, s, a) - q_value(mdp, V, s, ref_a)))
    return sorted(out, key=lambda d: (-d["visits"], -d["gain"]))


def plot_p_policies(world, vi, path):
    fig, axes = plt.subplots(len(P_VALUES), 3, figsize=(15, 5 * len(P_VALUES)))
    for row, p in enumerate(P_VALUES):
        for col, (phase, title) in enumerate(PHASES):
            changed = [(d["state"][0], d["state"][1]) for d in vi[p]["changed"] if (d["state"][2], d["state"][3]) == phase]
            draw_policy(axes[row, col], world, phase_cells(vi[p]["policy"], phase),
                        f"VI policy, p = {p:.2f}\n{title}", highlight=changed)
    fig.suptitle("Value-iteration policy vs intended-action probability p (orange frame = action changed from p = 1.00)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_p_curves(curves, episodes, vi, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(SMOOTH, episodes + 1)
    for i, p in enumerate(P_VALUES):
        smooth = np.array([moving_average(r) for r in curves[p]])
        ax.plot(x, smooth.mean(axis=0), color=f"C{i}", label=f"Q-learning, p = {p:.2f}")
        ax.fill_between(x, smooth.mean(axis=0) - smooth.std(axis=0), smooth.mean(axis=0) + smooth.std(axis=0),
                        color=f"C{i}", alpha=0.15)
        ax.axhline(vi[p]["mc_reward"], color=f"C{i}", ls="--", lw=1, label=f"VI policy mean reward, p = {p:.2f}")
    ax.set(title="Cumulative reward per episode vs p (Q-learning baseline, mean ± std over seeds)",
           xlabel="episode", ylabel="cumulative reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def part3(world: WarehouseGridWorld, seeds: List[int], episodes: int) -> None:
    print("\n" + "=" * 30, "PART 3: stochasticity p = 1.00 / 0.90 / 0.80", "=" * 30)
    cfg = CONFIGS[BASELINE]
    vi, curves, rows = {}, {}, []
    for p in P_VALUES:
        mdp = WarehouseMDP(world, gamma=GAMMA, p=p)
        V, policy, iters = value_iteration(mdp, THETA)
        mc_reward, mc_success = simulate(world, policy.__getitem__, p, seed=1000)
        vi[p] = dict(mdp=mdp, V=V, policy=policy, iters=iters, mc_reward=mc_reward,
                     weights=visit_weights(mdp, policy, world.max_steps))
        rows.append((p, "Value iteration", mc_reward, mc_success, f"{iters} sweeps", V[mdp.start_state], 100.0))

        q_stats, curves[p] = [], []
        for seed in seeds:
            agent, rewards, _ = run_q_learning(world, cfg, seed, episodes, p=p)
            q_policy = agent.policy(policy)
            reward, success = simulate(world, q_policy.__getitem__, p, seed=1000 + seed)
            on_path = [s for s, w in vi[p]["weights"].items() if w >= 0.01]
            q_stats.append(dict(reward=reward, success=success, conv=convergence_episode(rewards),
                                v=evaluate_policy(mdp, q_policy)[mdp.start_state],
                                agree=np.mean([q_policy[s] == policy[s] for s in on_path])))
            curves[p].append(rewards)
        conv = [q["conv"] for q in q_stats if q["conv"] is not None]
        rows.append((p, "Q-learning", np.mean([q["reward"] for q in q_stats]),
                     np.mean([q["success"] for q in q_stats]),
                     f"ep. {np.mean(conv):.0f} ± {np.std(conv):.0f}" if conv else "not reached",
                     np.mean([q["v"] for q in q_stats]), 100 * np.mean([q["agree"] for q in q_stats])))

    ref = vi[1.0]
    for p in P_VALUES:
        vi[p]["changed"] = [] if p == 1.0 else changed_states(
            ref["policy"], vi[p]["mdp"], vi[p]["V"], vi[p]["policy"], vi[p]["weights"])

    print(f"\nGreedy evaluation: {EVAL_EPISODES} Monte-Carlo episodes per policy (exploration off); "
          f"Q-learning = baseline config averaged over seeds {seeds}")
    print("| p | method | avg cumulative reward | success rate | convergence | V(start) | agrees with VI on path |")
    for p, method, reward, success, conv, v, agree in rows:
        print(f"| {p:.2f} | {method} | {reward:.1f} | {success:.1%} | {conv} | {v:.2f} | {agree:.0f}% |")

    for p in P_VALUES[1:]:
        changed = vi[p]["changed"]
        print(f"\np = {p:.2f}: {len(changed)} states whose preferred VI action changed from p = 1.00 "
              f"(the p = 1.00 action is now strictly worse). Most visited first:")
        for d in changed[:8]:
            (r, c, hp, dl) = d["state"]
            print(f"  (row {r}, col {c}, has_package={hp}, delivered={dl}): {d['ref_action']} -> {d['action']}, "
                  f"expected visits {d['visits']:.2f}, Q gain {d['gain']:.2f}"
                  f"{', next to a congestion zone' if d['congestion'] else ''}")

    plot_p_policies(world, vi, os.path.join(PLOT_DIR, "p_policies.png"))
    plot_p_curves(curves, episodes, vi, os.path.join(PLOT_DIR, "p_reward_vs_episode.png"))
    print("\nsaved plots/p_policies.png and plots/p_reward_vs_episode.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiments for Parts 2 and 3")
    parser.add_argument("--part", choices=["2", "3", "all"], default="all")
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Q-learning seeds")
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()
    os.makedirs(PLOT_DIR, exist_ok=True)

    world = WarehouseGridWorld(seed=args.layout_seed)
    print(f"layout seed {args.layout_seed}: Pickup={world.pickup_pos} Packing={world.packing_pos} Dock={world.dock_pos}")
    if args.part in ("2", "all"):
        part2(world, args.seeds, args.episodes)
    if args.part in ("3", "all"):
        part3(world, args.seeds, args.episodes)


if __name__ == "__main__":
    main()
