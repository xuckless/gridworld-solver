# Warehouse Grid World Solver

Value iteration and Q-learning for the warehouse robot grid world, with a live pygame visualiser.

## Setup

Use a virtual environment (Homebrew Python blocks global `pip install`, and a bare `pip`
may point at a different Python than `python3`):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```sh
.venv/bin/python main.py          # solver visualiser
.venv/bin/python experiments.py   # experiments and plots
```

The project uses `pygame-ce` (imported as `pygame`). The original `pygame` 2.6.1 fails
to load its font module on Python 3.14. If you have it installed, run
`.venv/bin/python -m pip uninstall pygame` before installing the requirements.
