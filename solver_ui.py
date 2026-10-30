"""Live pygame visualiser for the warehouse solvers (value iteration and Q-learning).

The robot moves under the 0.8/0.1/0.1 slip dynamics while following the chosen solver's
greedy policy. Reuses the grid drawing from gridworld.py.

Controls: 1/2 or click a button = choose solver | SPACE pause | R rerun | N new layout |
          UP/DOWN speed | ESC quit
"""

import random
from typing import Callable, List, Optional

import pygame

from gridworld import CELL_SIZE, GRID_SIZE, SCREEN_WIDTH, TEXT_COLOR, WarehouseGridWorld, draw_grid
from mdp import State, WarehouseMDP
from q_learning import QLearningAgent, StochasticWarehouseEnv, train
from value_iteration import value_iteration

GAMMA = 0.99
VI_THETA = 1e-6
QL_EPISODES = 5000
QL_CHUNK = 50  # training episodes per frame, so progress stays visible
QL_PARAMS = dict(alpha=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999)

PANEL_HEIGHT = 56  # status panel drawn by gridworld.draw_grid
BAR_HEIGHT = 44
GRID_BOTTOM = GRID_SIZE * CELL_SIZE
WINDOW_SIZE = (SCREEN_WIDTH, GRID_BOTTOM + PANEL_HEIGHT + BAR_HEIGHT)

APPROACHES = ["Value Iteration", "Q-Learning"]
BUTTON_RECTS = [
    pygame.Rect(10 + i * 170, GRID_BOTTOM + PANEL_HEIGHT + 7, 160, 30) for i in range(len(APPROACHES))
]
ACTIVE_COLOR = (70, 130, 255)
IDLE_COLOR = (225, 228, 235)


class SolverApp:
    def __init__(self, seed: int = 42):
        self.world = WarehouseGridWorld(seed=seed)
        self.approach: Optional[int] = None
        self.policy_fn: Optional[Callable[[State], str]] = None
        self.env: Optional[StochasticWarehouseEnv] = None
        self.agent: Optional[QLearningAgent] = None
        self.trained_episodes = 0
        self.successes: List[bool] = []
        self.paused = False
        self.step_ms = 150
        self.since_step = 0
        self.message = "Pick a solver (1: Value Iteration, 2: Q-Learning)"

    def select(self, approach: int) -> None:
        """(Re)build the solver for the current layout and start from the dock."""
        self.approach = approach
        self.env = StochasticWarehouseEnv(self.world, seed=random.randrange(2**31), max_steps=self.world.max_steps)
        self.policy_fn = None
        self.agent = None
        self.trained_episodes = 0
        self.successes = []
        if approach == 0:
            mdp = WarehouseMDP(self.world, gamma=GAMMA)
            _, policy, iters = value_iteration(mdp, VI_THETA)
            self.policy_fn = policy.__getitem__
            self.message = f"Value iteration converged in {iters} iterations (gamma={GAMMA}, theta={VI_THETA})"
        else:
            self.agent = QLearningAgent(
                self.world, self.env.actions, QL_PARAMS["alpha"], GAMMA, QL_PARAMS["epsilon"], seed=random.randrange(2**31)
            )
            self.message = "Training Q-learning..."
        self.restart_run()

    def restart_run(self) -> None:
        """Put the robot back on the dock and reset the displayed episode."""
        self.world.reset()
        if self.env is not None:
            self.env.reset()
        self.since_step = 0

    def new_layout(self) -> None:
        self.world.regenerate_layout(self.world.seed_value + 1)
        if self.approach is None:
            self.message = f"New layout (seed={self.world.seed_value})"
        else:
            self.select(self.approach)

    def training(self) -> bool:
        return self.agent is not None and self.policy_fn is None

    def train_chunk(self) -> None:
        _, successes = train(
            self.env, self.agent, QL_CHUNK, QL_PARAMS["epsilon_min"], QL_PARAMS["epsilon_decay"]
        )
        self.successes += successes
        self.trained_episodes += QL_CHUNK
        recent = self.successes[-200:]
        self.message = (
            f"Training Q-learning: episode {self.trained_episodes}/{QL_EPISODES}  "
            f"epsilon={self.agent.epsilon:.2f}  recent success {sum(recent) / len(recent):.0%}"
        )
        if self.trained_episodes >= QL_EPISODES:
            self.agent.epsilon = 0.0  # exploration off for the live demo
            self.policy_fn = self.agent.greedy_action
            self.restart_run()
            self.message = "Q-learning trained; following the greedy policy"

    def step_robot(self) -> None:
        if self.world.done:
            return
        state, reward, terminated, truncated = self.env.step(self.policy_fn(self.env.state))
        world = self.world
        world.robot_pos = (state[0], state[1])
        world.has_package = bool(state[2])
        world.delivered = bool(state[3])
        world.steps = self.env.steps
        world.score += reward
        world.done = terminated or truncated
        if terminated:
            self.message = f"Task complete in {world.steps} steps, score {world.score:.0f}  (R to rerun)"
        elif truncated:
            self.message = f"Step limit reached, score {world.score:.0f}  (R to rerun)"

    def update(self, dt_ms: int) -> None:
        if self.approach is None:
            return
        if self.training():
            self.train_chunk()
            return
        if self.paused or self.world.done:
            return
        self.since_step += dt_ms
        while self.since_step >= self.step_ms and not self.world.done:
            self.since_step -= self.step_ms
            self.step_robot()

    def handle_key(self, key: int) -> bool:
        """Return False when the app should quit."""
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_1:
            self.select(0)
        elif key == pygame.K_2:
            self.select(1)
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r and self.approach is not None and not self.training():
            self.restart_run()
            self.message = f"{APPROACHES[self.approach]}: new run"
        elif key == pygame.K_n:
            self.new_layout()
        elif key == pygame.K_UP:
            self.step_ms = max(20, self.step_ms - 30)
        elif key == pygame.K_DOWN:
            self.step_ms = min(600, self.step_ms + 30)
        return True

    def handle_click(self, pos) -> None:
        for i, rect in enumerate(BUTTON_RECTS):
            if rect.collidepoint(pos):
                self.select(i)

    def draw(self, screen, font, small_font) -> None:
        draw_grid(self.world, screen, font, small_font)
        panel_y = GRID_BOTTOM
        # Replace draw_grid's help line with this app's own status line.
        pygame.draw.rect(screen, (245, 247, 250), pygame.Rect(0, panel_y + 28, SCREEN_WIDTH, PANEL_HEIGHT - 28))
        name = APPROACHES[self.approach] if self.approach is not None else "none"
        speed = "paused" if self.paused else f"{self.step_ms} ms/step"
        screen.blit(small_font.render(f"Solver: {name} | {speed} | {self.message}", True, TEXT_COLOR), (10, panel_y + 32))

        pygame.draw.rect(screen, (235, 238, 244), pygame.Rect(0, panel_y + PANEL_HEIGHT, SCREEN_WIDTH, BAR_HEIGHT))
        for i, (rect, label) in enumerate(zip(BUTTON_RECTS, APPROACHES)):
            active = i == self.approach
            pygame.draw.rect(screen, ACTIVE_COLOR if active else IDLE_COLOR, rect, border_radius=6)
            text = small_font.render(f"{i + 1}  {label}", True, (255, 255, 255) if active else TEXT_COLOR)
            screen.blit(text, text.get_rect(center=rect.center))
        hint = small_font.render("SPACE pause | R rerun | N layout", True, TEXT_COLOR)
        screen.blit(hint, (BUTTON_RECTS[-1].right + 12, BUTTON_RECTS[0].y + 3))
        speed_hint = small_font.render("UP/DOWN speed | ESC quit", True, TEXT_COLOR)
        screen.blit(speed_hint, (BUTTON_RECTS[-1].right + 12, BUTTON_RECTS[0].y + 17))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Warehouse Solver Visualiser")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    small_font = pygame.font.SysFont("arial", 14)
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
