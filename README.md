# David_Cyubahiro_rl_summative

Reinforcement learning project for traffic-light optimization at a single Kigali intersection.

## Implemented Components

- Custom Gymnasium environment in `environment/custom_env.py`
- Pygame visualization engine in `environment/rendering.py`
- DQN training pipeline in `training/dqn_training.py`
- PPO, A2C, and REINFORCE training pipeline in `training/pg_training.py`
- Random-action and trained-agent simulation entry point in `main.py`

## Project Structure

```text
project_root/
├── environment/
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── dqn_training.py
│   └── pg_training.py
├── models/
│   ├── dqn/
│   └── pg/
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Static Simulation (Random Actions)

```bash
python main.py --mode random --steps 1200 --fps 30
```

## Train DQN (10 Hyperparameter Runs)

```bash
python -m training.dqn_training --timesteps 120000 --runs 10 --output models/dqn
```

Outputs:

- Saved checkpoints in `models/dqn/`
- Experiment table in `models/dqn/results.csv`

## Train Policy Methods (PPO, A2C, REINFORCE)

```bash
python -m training.pg_training --timesteps 120000 --runs 10 --output models/pg --algos ppo,a2c,reinforce
```

Outputs:

- PPO table: `models/pg/ppo_results.csv`
- A2C table: `models/pg/a2c_results.csv`
- REINFORCE table: `models/pg/reinforce_results.csv`

## Run Best Trained Agent

If `--model-path` is omitted, `main.py` loads the best model from the corresponding CSV file.

```bash
# DQN
python main.py --mode model --algo dqn --episodes 3

# PPO
python main.py --mode model --algo ppo --episodes 3

# A2C
python main.py --mode model --algo a2c --episodes 3

# REINFORCE
python main.py --mode model --algo reinforce --episodes 3
```

## Evaluation Metrics Captured

- Mean evaluation reward
- Training run hyperparameters
- Saved model path for replay

You can extend evaluation with queue-length variance, convergence speed, and generalization tests in your report notebook/script.
