from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import DQN

from environment.custom_env import EnvConfig, KigaliTrafficEnv


def make_env(seed: int | None = None) -> KigaliTrafficEnv:
    return KigaliTrafficEnv(config=EnvConfig(seed=seed))


def evaluate_model(model: DQN, episodes: int = 5) -> float:
    scores = []
    for _ in range(episodes):
        env = make_env()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        env.close()
        scores.append(total_reward)
    return float(np.mean(scores))


def hyperparameter_grid() -> List[Dict]:
    return [
        {"learning_rate": 1e-3, "gamma": 0.95, "batch_size": 32, "buffer_size": 30000},
        {"learning_rate": 5e-4, "gamma": 0.95, "batch_size": 64, "buffer_size": 30000},
        {"learning_rate": 3e-4, "gamma": 0.97, "batch_size": 64, "buffer_size": 50000},
        {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 64, "buffer_size": 50000},
        {"learning_rate": 8e-4, "gamma": 0.98, "batch_size": 128, "buffer_size": 80000},
        {"learning_rate": 2e-4, "gamma": 0.99, "batch_size": 128, "buffer_size": 80000},
        {"learning_rate": 7e-4, "gamma": 0.96, "batch_size": 32, "buffer_size": 50000},
        {"learning_rate": 4e-4, "gamma": 0.98, "batch_size": 64, "buffer_size": 100000},
        {"learning_rate": 3e-4, "gamma": 0.995, "batch_size": 128, "buffer_size": 100000},
        {"learning_rate": 1e-4, "gamma": 0.995, "batch_size": 256, "buffer_size": 120000},
    ]


def run_dqn_experiments(timesteps: int, runs: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = hyperparameter_grid()[:runs]
    results = []

    for run_id, hp in enumerate(grid, start=1):
        env = make_env(seed=run_id)
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            batch_size=hp["batch_size"],
            buffer_size=hp["buffer_size"],
            train_freq=4,
            target_update_interval=250,
            exploration_fraction=0.25,
            verbose=1,
            tensorboard_log=str(output_dir / "tb"),
        )
        model.learn(total_timesteps=timesteps, progress_bar=True)

        mean_reward = evaluate_model(model, episodes=5)
        save_path = output_dir / f"dqn_run_{run_id}"
        model.save(str(save_path))
        env.close()

        row = {"run_id": run_id, **hp, "mean_eval_reward": mean_reward, "model_path": f"{save_path}.zip"}
        results.append(row)
        print(f"[DQN] run={run_id} mean_eval_reward={mean_reward:.2f}")

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = max(results, key=lambda x: x["mean_eval_reward"])
    print("\\nBest DQN run:")
    print(best)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Kigali traffic environment")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=str, default="models/dqn")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dqn_experiments(args.timesteps, args.runs, Path(args.output))
