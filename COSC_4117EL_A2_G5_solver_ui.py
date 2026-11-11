"""Live pygame visualiser for the warehouse solvers (value iteration and Q-learning, Parts 1-3).

AI use: originally implemented by our group (G5); refactored and bug-fixed with the help of
Claude (Anthropic, Claude Code). Claude-assisted changes in this file: the step-by-step "glow"
visualisation (`SolverApp.tick`, `_vi_sweep`, `_ql_chunk`, `_robot_step`, `fade`, `draw_overlay`,
`draw_arrow`), the p selector for Part 3 (`set_p`, `_reference_policy`, changed-action frames),
the phase selector (`display_phase`), and the reworked control bar in `draw`.

What you see:
  * Value iteration is animated one Bellman sweep per tick. A faint gold heat-map shows V(s) for the
    displayed task phase; cells whose value changed most in the latest sweep glow cyan, and older
    changes fade out. Arrows are the greedy policy at that sweep.
  * Q-learning trains in chunks of episodes. The latest exploration episode is drawn as a magenta
    trail (older steps fainter); the gold heat-map is max_a Q(s, a) and arrows the greedy action.
  * After solving, the robot follows the greedy policy under the slip dynamics and leaves a fading
    purple trail, so slips are visible. With p < 1, orange frames mark cells where the action value
    iteration chose at p = 1.00 is no longer optimal (Part 3).

Controls: 1/2 or buttons = solver | P or buttons = p (1.00/0.90/0.80) | F = phase shown |
          SPACE pause | S single step | R rerun | N new layout | UP/DOWN speed | ESC quit
"""

from typing import Dict, Iterator, List, Optional

import numpy as np
import pygame

from gridworld import (
    CELL_SIZE,
    EMPTY,
    GRID_SIZE,
    ROBOT_COLOR,
    SCREEN_WIDTH,
    SHELF,
    TEXT_COLOR,
    WarehouseGridWorld,
    draw_grid,
)
from COSC_4117EL_A2_G5_mdp import P_VALUES, PHASES, State, WarehouseMDP
from COSC_4117EL_A2_G5_q_learning import QLearningAgent, StochasticWarehouseEnv, run_episode
from COSC_4117EL_A2_G5_value_iteration import (
    greedy_policy,
    optimal_actions,
    value_iteration,
    value_iteration_sweeps,
)

GAMMA = 0.99
VI_THETA = 1e-6
QL_EPISODES = 5000
QL_CHUNK = 25  # training episodes per tick
QL_PARAMS = dict(alpha=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999)
FADE = 0.82  # glow kept per tick, so older steps get fainter

# Layout
GRID_BOTTOM = GRID_SIZE * CELL_SIZE
PANEL_HEIGHT = 56  # status panel drawn by gridworld.draw_grid
MESSAGE_HEIGHT = 22
BAR_HEIGHT = 66
BAR_TOP = GRID_BOTTOM + PANEL_HEIGHT + MESSAGE_HEIGHT
WINDOW_SIZE = (SCREEN_WIDTH, BAR_TOP + BAR_HEIGHT)

APPROACHES = ["Value Iteration", "Q-Learning"]
SOLVER_BUTTONS = [pygame.Rect(10 + i * 128, BAR_TOP + 7, 122, 28) for i in range(len(APPROACHES))]
P_BUTTONS = [pygame.Rect(276 + i * 92, BAR_TOP + 7, 86, 28) for i in range(len(P_VALUES))]

# Colours
PANEL_COLOR = (245, 247, 250)
BAR_COLOR = (235, 238, 244)
ACTIVE_COLOR = (70, 130, 255)
IDLE_COLOR = (222, 226, 234)
HEAT_COLOR = (255, 185, 0)
VI_GLOW = (0, 210, 255)
QL_GLOW = (230, 40, 230)
RUN_GLOW = (140, 60, 255)
CHANGED_COLOR = (255, 140, 0)
ARROW_COLOR = (45, 45, 60)
ARROW_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class SolverApp:
    def __init__(self, seed: int = 42):
        self.world = WarehouseGridWorld(seed=seed)
        self.p_index = P_VALUES.index(0.8)
        self.approach: Optional[int] = None
        self.mode = "idle"  # idle | vi_sweeps | ql_training | running | done
        self.paused = False
        self.step_ms = 120
        self.since_tick = 0
        self.phase_choice = 0  # 0 = auto, otherwise index into PHASES + 1
        self.ref_policy: Optional[Dict[State, str]] = None  # VI policy at p = 1.00 for this layout
        self.message = "Pick a solver (1: Value Iteration, 2: Q-Learning)"
        self._clear_solver()

    # ------------------------------------------------------------------ solver lifecycle
    @property
    def p(self) -> float:
        return P_VALUES[self.p_index]

    def _clear_solver(self) -> None:
        self.mdp: Optional[WarehouseMDP] = None
        self.env: Optional[StochasticWarehouseEnv] = None
        self.agent: Optional[QLearningAgent] = None
        self.sweeps: Optional[Iterator] = None
        self.V: Optional[Dict[State, float]] = None
        self.policy: Optional[Dict[State, str]] = None
        self.iteration = 0
        self.hot_phase = PHASES[0][0]
        self.successes: List[bool] = []
        self.glow = np.zeros((GRID_SIZE, GRID_SIZE))
        self.glow_color = VI_GLOW

    def select(self, approach: int) -> None:
        """(Re)build the chosen solver for the current layout and p, then animate it."""
        self._clear_solver()
        self.approach = approach
        seed = self.world.seed_value * 1000 + self.p_index
        self.mdp = WarehouseMDP(self.world, gamma=GAMMA, p=self.p)
        self.env = StochasticWarehouseEnv(self.world, seed=seed, max_steps=self.world.max_steps, p=self.p)
        if approach == 0:
            self.sweeps = value_iteration_sweeps(self.mdp, VI_THETA)
            self.mode, self.glow_color = "vi_sweeps", VI_GLOW
            self.message = f"Value iteration started (p={self.p:.2f}, gamma={GAMMA}, theta={VI_THETA})"
        else:
            self.agent = QLearningAgent(self.world, self.env.actions, QL_PARAMS["alpha"], GAMMA,
                                        QL_PARAMS["epsilon"], seed=seed)
            self.mode, self.glow_color = "ql_training", QL_GLOW
            self.message = f"Q-learning started (p={self.p:.2f})"
        self.world.reset()
        self.env.reset()

    def restart_run(self) -> None:
        """Put the robot back on the dock and start a fresh greedy run."""
        self.world.reset()
        self.env.reset()
        self.glow[:] = 0
        self.glow_color = RUN_GLOW
        self.mode = "running"
        self.since_tick = 0

    def set_p(self, index: int) -> None:
        self.p_index = index
        if self.approach is not None:
            self.select(self.approach)
        else:
            self.message = f"p = {self.p:.2f} (intended-action probability)"

    def new_layout(self) -> None:
        self.world.regenerate_layout(self.world.seed_value + 1)
        self.ref_policy = None
        if self.approach is None:
            self.message = f"New layout (seed={self.world.seed_value})"
        else:
            self.select(self.approach)

    def _reference_policy(self) -> Dict[State, str]:
        if self.ref_policy is None:
            self.ref_policy = value_iteration(WarehouseMDP(self.world, gamma=GAMMA, p=1.0), VI_THETA)[1]
        return self.ref_policy

    # ------------------------------------------------------------------ one algorithmic step
    def fade(self) -> None:
        self.glow *= FADE
        self.glow[self.glow < 0.03] = 0

    def tick(self) -> None:
        self.fade()
        if self.mode == "vi_sweeps":
            self._vi_sweep()
        elif self.mode == "ql_training":
            self._ql_chunk()
        elif self.mode == "running":
            self._robot_step()

    def _vi_sweep(self) -> None:
        try:
            V, old_V, delta = next(self.sweeps)
        except StopIteration:
            self.message = (f"Value iteration converged in {self.iteration} sweeps (p={self.p:.2f}); "
                            f"following its greedy policy")
            self.restart_run()
            return
        self.iteration += 1
        self.V = V
        self.policy = greedy_policy(self.mdp, V)
        change = np.zeros((GRID_SIZE, GRID_SIZE))
        phase_change = {phase: 0.0 for phase, _ in PHASES}
        for s, v in V.items():
            d = abs(v - old_V[s])
            change[s[0], s[1]] = max(change[s[0], s[1]], d)
            phase_change[(s[2], s[3])] += d
        self.hot_phase = max(phase_change, key=phase_change.get)  # where the update wave currently is
        if change.max() > 0:
            self.glow = np.maximum(self.glow, change / change.max())
        self.message = f"Sweep {self.iteration}: max |dV| = {delta:.5f} (stops below theta = {VI_THETA})"

    def _ql_chunk(self) -> None:
        path = []
        for _ in range(QL_CHUNK):
            _, success, path = run_episode(self.env, self.agent.act, self.agent)
            self.successes.append(success)
            self.agent.epsilon = max(QL_PARAMS["epsilon_min"], self.agent.epsilon * QL_PARAMS["epsilon_decay"])
        # Draw the most recent episode as a trail: later steps brighter than earlier ones.
        for i, (s, _) in enumerate(path):
            self.glow[s[0], s[1]] = max(self.glow[s[0], s[1]], 0.25 + 0.75 * (i + 1) / len(path))
        done = len(self.successes)
        recent = self.successes[-200:]
        self.message = (f"Episode {done}/{QL_EPISODES}  epsilon={self.agent.epsilon:.2f}  "
                        f"recent success {sum(recent) / len(recent):.0%}  last episode {len(path) - 1} steps")
        if done >= QL_EPISODES:
            self.agent.epsilon = 0.0  # exploration off for the live demo
            self.message = f"Q-learning trained ({done} episodes, p={self.p:.2f}); following the greedy policy"
            self.restart_run()

    def _robot_step(self) -> None:
        choose = self.policy.__getitem__ if self.approach == 0 else self.agent.greedy_action
        state, reward, terminated, truncated = self.env.step(choose(self.env.state))
        world = self.world
        world.robot_pos = (state[0], state[1])
        world.has_package, world.delivered = bool(state[2]), bool(state[3])
        world.steps = self.env.steps
        world.score += reward
        self.glow[state[0], state[1]] = 1.0
        if terminated or truncated:
            world.done = True
            self.mode = "done"
            outcome = "Task complete" if terminated else "Step limit reached"
            self.message = f"{outcome} in {world.steps} steps, score {world.score:.0f}  (R to rerun)"

    def update(self, dt_ms: int) -> None:
        if self.paused or self.mode in ("idle", "done"):
            return
        self.since_tick += dt_ms
        while self.since_tick >= self.step_ms and self.mode not in ("idle", "done"):
            self.since_tick -= self.step_ms
            self.tick()

    # ------------------------------------------------------------------ input
    def handle_key(self, key: int) -> bool:
        """Return False when the app should quit."""
        if key == pygame.K_ESCAPE:
            return False
        if key in (pygame.K_1, pygame.K_2):
            self.select(key - pygame.K_1)
        elif key == pygame.K_p:
            self.set_p((self.p_index + 1) % len(P_VALUES))
        elif key == pygame.K_f:
            self.phase_choice = (self.phase_choice + 1) % (len(PHASES) + 1)
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_s and self.mode not in ("idle", "done"):
            self.paused = True
            self.tick()
        elif key == pygame.K_r and self.approach is not None:
            if self.mode in ("running", "done"):
                self.restart_run()
                self.message = f"{APPROACHES[self.approach]}: new run"
            else:
                self.select(self.approach)
        elif key == pygame.K_n:
            self.new_layout()
        elif key == pygame.K_UP:
            self.step_ms = max(10, int(self.step_ms * 0.7))
        elif key == pygame.K_DOWN:
            self.step_ms = min(1000, int(self.step_ms / 0.7) + 1)
        return True

    def handle_click(self, pos) -> None:
        for i, rect in enumerate(SOLVER_BUTTONS):
            if rect.collidepoint(pos):
                self.select(i)
        for i, rect in enumerate(P_BUTTONS):
            if rect.collidepoint(pos):
                self.set_p(i)

    # ------------------------------------------------------------------ drawing
    def display_phase(self):
        """(has_package, delivered) whose values/arrows are drawn."""
        if self.phase_choice:
            return PHASES[self.phase_choice - 1][0]
        if self.mode == "vi_sweeps":
            return self.hot_phase
        if self.mode == "ql_training":
            return PHASES[0][0]
        return (int(self.world.has_package), int(self.world.delivered))

    def _cell_info(self, phase):
        """{(r, c): (value, action)} for the displayed phase, or {} before anything is known."""
        info = {}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                s = (r, c, *phase)
                if self.V is not None and s in self.V:
                    info[(r, c)] = (self.V[s], self.policy.get(s))
                elif self.agent is not None and self.world.grid[r, c] != SHELF and self.agent.row(s).any():
                    q = self.agent.row(s)
                    info[(r, c)] = (q.max(), self.agent.actions[int(np.argmax(q))])
        return info

    def draw_overlay(self, screen, phase) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, GRID_BOTTOM), pygame.SRCALPHA)
        info = self._cell_info(phase)
        if info:
            values = [v for v, _ in info.values()]
            lo, span = min(values), (max(values) - min(values)) or 1.0
            for (r, c), (v, _) in info.items():
                overlay.fill((*HEAT_COLOR, int(15 + 85 * (v - lo) / span)),
                             pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for r, c in zip(*np.nonzero(self.glow)):
            g = float(self.glow[r, c])
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            overlay.fill((*self.glow_color, int(150 * g)), rect)
            pygame.draw.rect(overlay, (*self.glow_color, int(230 * g)), rect.inflate(-4, -4), 2, border_radius=6)
        screen.blit(overlay, (0, 0))

        # Part 3: frame cells where the p = 1.00 VI action is no longer optimal at this p.
        if self.approach == 0 and self.mode in ("running", "done") and self.p < 1.0:
            ref = self._reference_policy()
            for (r, c) in info:
                s = (r, c, *phase)
                if s in ref and ref[s] not in optimal_actions(self.mdp, self.V, s):
                    rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, CHANGED_COLOR, rect.inflate(-6, -6), 3, border_radius=4)

        for (r, c), (_, action) in info.items():
            if action is not None and (r, c) != self.world.robot_pos:
                draw_arrow(screen, r, c, action, small=self.world.grid[r, c] != EMPTY)

    def draw(self, screen, font, small_font) -> None:
        draw_grid(self.world, screen, font, small_font)
        phase = self.display_phase()
        self.draw_overlay(screen, phase)
        center = (int((self.world.robot_pos[1] + 0.5) * CELL_SIZE), int((self.world.robot_pos[0] + 0.5) * CELL_SIZE))
        pygame.draw.circle(screen, ROBOT_COLOR, center, CELL_SIZE // 3)

        # Replace draw_grid's help line with this app's status and message lines.
        panel_y = GRID_BOTTOM
        pygame.draw.rect(screen, PANEL_COLOR, pygame.Rect(0, panel_y + 28, SCREEN_WIDTH, PANEL_HEIGHT - 28 + MESSAGE_HEIGHT))
        name = APPROACHES[self.approach] if self.approach is not None else "none"
        speed = "paused (S = step)" if self.paused else f"{self.step_ms} ms/tick"
        phase_name = dict(PHASES)[phase].split(": ")[1]
        shown = ("auto: " if not self.phase_choice else "") + phase_name
        screen.blit(small_font.render(f"{name} | p={self.p:.2f} | {speed} | F shows: {shown}", True, TEXT_COLOR),
                    (10, panel_y + 32))
        screen.blit(small_font.render(self.message, True, TEXT_COLOR), (10, panel_y + 54))

        pygame.draw.rect(screen, BAR_COLOR, pygame.Rect(0, BAR_TOP, SCREEN_WIDTH, BAR_HEIGHT))
        buttons = [(r, f"{i + 1}  {n}", i == self.approach) for i, (r, n) in enumerate(zip(SOLVER_BUTTONS, APPROACHES))]
        buttons += [(r, f"p = {p:.2f}", i == self.p_index) for i, (r, p) in enumerate(zip(P_BUTTONS, P_VALUES))]
        for rect, label, active in buttons:
            pygame.draw.rect(screen, ACTIVE_COLOR if active else IDLE_COLOR, rect, border_radius=6)
            text = small_font.render(label, True, (255, 255, 255) if active else TEXT_COLOR)
            screen.blit(text, text.get_rect(center=rect.center))
        hint = "SPACE pause | S step | R rerun | N layout | P cycle p | F phase | UP/DOWN speed | ESC quit"
        screen.blit(small_font.render(hint, True, TEXT_COLOR), (10, BAR_TOP + 43))


def draw_arrow(screen, r: int, c: int, action: str, small: bool = False) -> None:
    """Policy arrow in a cell; `small` puts a shorter arrow in the corner so cell labels stay readable."""
    cx, cy = (c + 0.5) * CELL_SIZE, (r + 0.5) * CELL_SIZE
    if small:
        cx, cy = cx + CELL_SIZE * 0.27, cy + CELL_SIZE * 0.27
    dx, dy = ARROW_DELTA[action]
    length = CELL_SIZE * (0.14 if small else 0.28)
    tip = (cx + dx * length, cy + dy * length)
    tail = (cx - dx * length * 0.6, cy - dy * length * 0.6)
    pygame.draw.line(screen, ARROW_COLOR, tail, tip, 2)
    side = (-dy * 5, dx * 5)
    back = (tip[0] - dx * 8, tip[1] - dy * 8)
    pygame.draw.polygon(screen, ARROW_COLOR, [tip, (back[0] + side[0], back[1] + side[1]),
                                              (back[0] - side[0], back[1] - side[1])])


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Warehouse Solver Visualiser")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    small_font = pygame.font.SysFont("arial", 13)
    app = SolverApp()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = app.handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                app.handle_click(event.pos)
        app.update(clock.get_time())
        app.draw(screen, font, small_font)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
