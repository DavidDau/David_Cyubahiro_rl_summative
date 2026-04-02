from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import EnvConfig, KigaliTrafficEnv
from environment.rendering import IntersectionRenderer
from training.pg_training import ReinforcePolicy


def run_random(steps: int, fps: int):
    env = KigaliTrafficEnv(config=EnvConfig())
    renderer = IntersectionRenderer(fps=fps)
    obs, _ = env.reset()

    for step in range(1, steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        keep_running = renderer.draw(obs, action=action, reward=reward, step=step, mode_label="RANDOM")
        if not keep_running or terminated or truncated:
            obs, _ = env.reset()

    renderer.close()
    env.close()


def load_best_from_csv(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["mean_eval_reward"]))
    return best["model_path"]


def run_sb3_model(algo: str, model_path: Path, episodes: int, fps: int):
    env = KigaliTrafficEnv(config=EnvConfig())
    renderer = IntersectionRenderer(fps=fps)

    if algo == "dqn":
        model = DQN.load(str(model_path))
    elif algo == "ppo":
        model = PPO.load(str(model_path))
    elif algo == "a2c":
        model = A2C.load(str(model_path))
    else:
        raise ValueError(f"Unsupported SB3 algorithm: {algo}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            keep_running = renderer.draw(obs, action=int(action), reward=reward, step=step, mode_label=algo.upper())
            if not keep_running:
                done = True
                break
            done = terminated or truncated
        print(f"Episode {ep}: reward={ep_reward:.2f}")

    renderer.close()
    env.close()


def run_reinforce(model_path: Path, episodes: int, fps: int):
    env = KigaliTrafficEnv(config=EnvConfig())
    renderer = IntersectionRenderer(fps=fps)

    policy = ReinforcePolicy()
    payload = torch.load(str(model_path), map_location="cpu")
    policy.load_state_dict(payload["state_dict"])
    policy.eval()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        while not done:
            step += 1
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            keep_running = renderer.draw(obs, action=action, reward=reward, step=step, mode_label="REINFORCE")
            if not keep_running:
                done = True
                break
            done = terminated or truncated
        print(f"Episode {ep}: reward={ep_reward:.2f}")

    renderer.close()
    env.close()


def resolve_model_path(algo: str, model_path: str | None) -> Path:
    if model_path:
        return Path(model_path)

    if algo == "dqn":
        return Path(load_best_from_csv(Path("models/dqn/results.csv")))
    if algo == "ppo":
        return Path(load_best_from_csv(Path("models/pg/ppo_results.csv")))
    if algo == "a2c":
        return Path(load_best_from_csv(Path("models/pg/a2c/a2c_results.csv")))
    if algo == "reinforce":
        return Path(load_best_from_csv(Path("models/pg/reinforce/reinforce_results.csv")))
    raise ValueError(f"Unknown algorithm: {algo}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run random or trained simulation with visualization")
    parser.add_argument("--mode", choices=["random", "model"], default="random")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default="dqn")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "random":
        run_random(steps=args.steps, fps=args.fps)
    else:
        path = resolve_model_path(args.algo, args.model_path)
        print(f"Loading model: {path}")
        time.sleep(0.2)
        if args.algo in {"dqn", "ppo", "a2c"}:
            run_sb3_model(args.algo, path, episodes=args.episodes, fps=args.fps)
        else:
            run_reinforce(path, episodes=args.episodes, fps=args.fps)
