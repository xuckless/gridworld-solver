# COSC 4117EL Assignment 2 (Group 5): Warehouse Robot with DP and RL

Value iteration (Part 1), Q-learning (Part 2), and a stochasticity study over the intended-action
probability p = 1.00 / 0.90 / 0.80 (Part 3) on the provided `gridworld.py` warehouse environment,
plus a live pygame visualiser.

## Files

| File | Purpose |
|---|---|
| `gridworld.py` | Provided environment framework, **unchanged** |
| `COSC_4117EL_A2_G5_mdp.py` | MDP formulation: states, slip transitions with parameter `p`, rewards, terminal test |
| `COSC_4117EL_A2_G5_value_iteration.py` | Part 1: value iteration, policy extraction, text policy grids |
| `COSC_4117EL_A2_G5_q_learning.py` | Part 2: black-box environment, epsilon-greedy Q-learning agent, training, greedy evaluation run |
| `COSC_4117EL_A2_G5_experiments.py` | Parts 2 and 3: hyperparameter/seed study, VI vs Q comparison, p sweep, plots in `plots/` |
| `COSC_4117EL_A2_G5_solver_ui.py` | Step-by-step visualiser for both solvers and every p |
| `COSC_4117EL_A2_G5_main.py` | Launches the visualiser |

### Changes to the provided environment

`gridworld.py` is not modified. The stochastic transitions (intended direction with probability p,
each perpendicular direction with (1 - p) / 2, blocked moves leave the robot in place) are implemented
in `COSC_4117EL_A2_G5_mdp.py` on top of the environment's grid, positions, reward constants and
`state_to_index`. The step limit (`max_steps = 120`) of the environment is used as the episode cutoff.

## Setup

**macOS / Linux:**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The project uses `pygame-ce` (imported as `pygame`); the original `pygame` 2.6.1 fails to load its font
module on Python 3.14. If it is installed, run `.venv/bin/python -m pip uninstall pygame` first.

## Run

**macOS / Linux:**

```sh
.venv/bin/python COSC_4117EL_A2_G5_value_iteration.py [--p 0.8] [--gamma 0.99] [--theta 1e-6] [--seed 42]
.venv/bin/python COSC_4117EL_A2_G5_q_learning.py      [--p 0.8] [--alpha 0.1] [--episodes 5000] [--seed 42]
.venv/bin/python COSC_4117EL_A2_G5_experiments.py     [--part 2|3|all] [--seeds 0 1 2 3 4]
.venv/bin/python COSC_4117EL_A2_G5_main.py            # visualiser
```

**Windows:**

```
.\.venv\Scripts\python COSC_4117EL_A2_G5_value_iteration.py [--p 0.8] [--gamma 0.99] [--theta 1e-6] [--seed 42]
.\.venv\Scripts\python COSC_4117EL_A2_G5_q_learning.py      [--p 0.8] [--alpha 0.1] [--episodes 5000] [--seed 42]
.\.venv\Scripts\python COSC_4117EL_A2_G5_experiments.py      [--part 2|3|all] [--seeds 0 1 2 3 4]
.\.venv\Scripts\python COSC_4117EL_A2_G5_main.py             # visualiser`
```

Defaults: layout seed 42, gamma = 0.99, theta = 1e-6, Q-learning seeds 0-4, 5000 episodes,
alpha = 0.1, epsilon 1.0 -> 0.05 with decay 0.999 per episode.

Plots written by the experiments: `reward_vs_episode_configs.png`, `reward_vs_episode_seeds.png`,
`policy_comparison.png` (Part 2) and `p_policies.png`, `p_reward_vs_episode.png` (Part 3).

## Visualiser

| Key | Action |
|---|---|
| `1` / `2` or buttons | Value iteration / Q-learning |
| `P` or buttons | Cycle p = 1.00 / 0.90 / 0.80 |
| `F` | Phase whose values/arrows are shown (auto, pickup, carry, return) |
| `SPACE` / `S` | Pause / advance a single step |
| `R` / `N` | Rerun / next seeded layout |
| `UP` / `DOWN` | Faster / slower |

What it shows, step by step:
- Value iteration runs one Bellman sweep per tick. A gold heat-map shows V(s), and cells that changed
  most in the latest sweep glow cyan, fading over later sweeps.
- Q-learning trains in chunks of episodes. The latest exploration episode is drawn as a magenta trail,
  with earlier steps fainter.
- The final greedy run leaves a fading purple trail behind the robot, so slips are visible.
- Orange frames mark states where the p = 1.00 value-iteration action is no longer optimal at the
  selected p (Part 3).

## Generative AI use

The code was originally implemented by our group. It was then refactored and bug-fixed with the help
of Claude (Anthropic, via Claude Code). Each source file starts with a comment that lists the functions
Claude helped add or change.
