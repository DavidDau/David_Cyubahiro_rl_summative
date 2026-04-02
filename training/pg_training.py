from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3 import A2C, PPO

from environment.custom_env import EnvConfig, KigaliTrafficEnv


class ReinforcePolicy(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_env(seed: int | None = None) -> KigaliTrafficEnv:
    return KigaliTrafficEnv(config=EnvConfig(seed=seed))


def evaluate_sb3(model, episodes: int = 5) -> float:
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


def evaluate_reinforce(policy: ReinforcePolicy, episodes: int = 5) -> float:
    policy.eval()
    scores = []
    for _ in range(episodes):
        env = make_env()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        scores.append(total_reward)
    return float(np.mean(scores))


def discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    ret = 0.0
    returns = []
    for r in reversed(rewards):
        ret = r + gamma * ret
        returns.append(ret)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    return returns_t


def train_reinforce(
    timesteps: int,
    learning_rate: float,
    gamma: float,
    entropy_coef: float,
    hidden_dim: int = 64,
) -> Tuple[ReinforcePolicy, int]:
    env = make_env()
    policy = ReinforcePolicy(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    steps_done = 0
    while steps_done < timesteps:
        obs, _ = env.reset()
        done = False

        log_probs = []
        entropies = []
        rewards = []

        while not done and steps_done < timesteps:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            rewards.append(float(reward))

            obs = next_obs
            done = terminated or truncated
            steps_done += 1

        returns_t = discounted_returns(rewards, gamma)
        log_probs_t = torch.cat(log_probs)
        entropy_t = torch.cat(entropies)

        loss = -(log_probs_t * returns_t).sum() - entropy_coef * entropy_t.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()
    return policy, steps_done


def ppo_grid() -> List[Dict]:
    return [
        {"learning_rate": 3e-4, "gamma": 0.95, "batch_size": 64, "ent_coef": 0.00},
        {"learning_rate": 3e-4, "gamma": 0.97, "batch_size": 64, "ent_coef": 0.01},
        {"learning_rate": 2e-4, "gamma": 0.98, "batch_size": 64, "ent_coef": 0.01},
        {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 64, "ent_coef": 0.02},
        {"learning_rate": 5e-4, "gamma": 0.98, "batch_size": 128, "ent_coef": 0.005},
        {"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 128, "ent_coef": 0.01},
        {"learning_rate": 2e-4, "gamma": 0.995, "batch_size": 128, "ent_coef": 0.02},
        {"learning_rate": 4e-4, "gamma": 0.96, "batch_size": 256, "ent_coef": 0.00},
        {"learning_rate": 1e-4, "gamma": 0.995, "batch_size": 256, "ent_coef": 0.01},
        {"learning_rate": 2e-4, "gamma": 0.99, "batch_size": 64, "ent_coef": 0.03},
    ]


def a2c_grid() -> List[Dict]:
    return [
        {"learning_rate": 7e-4, "gamma": 0.95, "ent_coef": 0.00},
        {"learning_rate": 5e-4, "gamma": 0.97, "ent_coef": 0.01},
        {"learning_rate": 3e-4, "gamma": 0.98, "ent_coef": 0.01},
        {"learning_rate": 2e-4, "gamma": 0.99, "ent_coef": 0.02},
        {"learning_rate": 1e-4, "gamma": 0.99, "ent_coef": 0.00},
        {"learning_rate": 8e-4, "gamma": 0.96, "ent_coef": 0.005},
        {"learning_rate": 4e-4, "gamma": 0.995, "ent_coef": 0.01},
        {"learning_rate": 3e-4, "gamma": 0.995, "ent_coef": 0.02},
        {"learning_rate": 2e-4, "gamma": 0.98, "ent_coef": 0.03},
        {"learning_rate": 1e-4, "gamma": 0.995, "ent_coef": 0.04},
    ]


def reinforce_grid() -> List[Dict]:
    return [
        {"learning_rate": 1e-3, "gamma": 0.95, "ent_coef": 0.000},
        {"learning_rate": 8e-4, "gamma": 0.97, "ent_coef": 0.000},
        {"learning_rate": 7e-4, "gamma": 0.98, "ent_coef": 0.001},
        {"learning_rate": 5e-4, "gamma": 0.99, "ent_coef": 0.001},
        {"learning_rate": 3e-4, "gamma": 0.99, "ent_coef": 0.002},
        {"learning_rate": 2e-4, "gamma": 0.995, "ent_coef": 0.002},
        {"learning_rate": 1e-4, "gamma": 0.995, "ent_coef": 0.005},
        {"learning_rate": 4e-4, "gamma": 0.96, "ent_coef": 0.000},
        {"learning_rate": 6e-4, "gamma": 0.98, "ent_coef": 0.001},
        {"learning_rate": 2e-4, "gamma": 0.99, "ent_coef": 0.004},
    ]


def write_results(rows: List[Dict], output_file: Path):
    if not rows:
        return
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_ppo(timesteps: int, runs: int, output_dir: Path):
    rows = []
    for run_id, hp in enumerate(ppo_grid()[:runs], start=1):
        env = make_env(seed=run_id)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            batch_size=hp["batch_size"],
            ent_coef=hp["ent_coef"],
            verbose=1,
            tensorboard_log=str(output_dir / "tb_ppo"),
        )
        model.learn(total_timesteps=timesteps, progress_bar=True)
        mean_reward = evaluate_sb3(model, episodes=5)
        save_path = output_dir / f"ppo_run_{run_id}"
        model.save(str(save_path))
        env.close()

        row = {"algo": "ppo", "run_id": run_id, **hp, "mean_eval_reward": mean_reward, "model_path": f"{save_path}.zip"}
        rows.append(row)
        print(f"[PPO] run={run_id} mean_eval_reward={mean_reward:.2f}")

    write_results(rows, output_dir / "ppo_results.csv")


def train_a2c(timesteps: int, runs: int, output_dir: Path):
    algo_dir = output_dir / "a2c"
    algo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, hp in enumerate(a2c_grid()[:runs], start=1):
        env = make_env(seed=100 + run_id)
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            ent_coef=hp["ent_coef"],
            verbose=1,
            tensorboard_log=str(output_dir / "tb_a2c"),
        )
        model.learn(total_timesteps=timesteps, progress_bar=True)
        mean_reward = evaluate_sb3(model, episodes=5)
        save_path = algo_dir / f"a2c_run_{run_id}"
        model.save(str(save_path))
        env.close()

        row = {"algo": "a2c", "run_id": run_id, **hp, "batch_size": None, "mean_eval_reward": mean_reward, "model_path": f"{save_path}.zip"}
        rows.append(row)
        print(f"[A2C] run={run_id} mean_eval_reward={mean_reward:.2f}")

    write_results(rows, algo_dir / "a2c_results.csv")


def train_reinforce_runs(timesteps: int, runs: int, output_dir: Path):
    algo_dir = output_dir / "reinforce"
    algo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, hp in enumerate(reinforce_grid()[:runs], start=1):
        policy, steps = train_reinforce(
            timesteps=timesteps,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            entropy_coef=hp["ent_coef"],
        )
        mean_reward = evaluate_reinforce(policy, episodes=5)
        save_path = algo_dir / f"reinforce_run_{run_id}.pt"
        torch.save({"state_dict": policy.state_dict(), "config": hp}, save_path)

        row = {
            "algo": "reinforce",
            "run_id": run_id,
            **hp,
            "batch_size": None,
            "steps_trained": steps,
            "mean_eval_reward": mean_reward,
            "model_path": str(save_path),
        }
        rows.append(row)
        print(f"[REINFORCE] run={run_id} mean_eval_reward={mean_reward:.2f}")

    write_results(rows, algo_dir / "reinforce_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO/A2C/REINFORCE on Kigali traffic environment")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=str, default="models/pg")
    parser.add_argument("--algos", type=str, default="ppo,a2c,reinforce")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    algos = {a.strip().lower() for a in args.algos.split(",") if a.strip()}
    if "ppo" in algos:
        train_ppo(args.timesteps, args.runs, output)
    if "a2c" in algos:
        train_a2c(args.timesteps, args.runs, output)
    if "reinforce" in algos:
        train_reinforce_runs(args.timesteps, args.runs, output)
