from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"


def load_data() -> dict[str, pd.DataFrame]:
    dqn = pd.read_csv(MODELS_DIR / "dqn" / "results.csv")
    dqn["algo"] = "dqn"

    ppo = pd.read_csv(MODELS_DIR / "pg" / "ppo_results.csv")
    a2c = pd.read_csv(MODELS_DIR / "pg" / "a2c" / "a2c_results.csv")
    reinforce = pd.read_csv(MODELS_DIR / "pg" / "reinforce" / "reinforce_results.csv")

    for df in (ppo, a2c, reinforce):
        df["algo"] = df["algo"].str.lower()

    return {
        "dqn": dqn,
        "ppo": ppo,
        "a2c": a2c,
        "reinforce": reinforce,
    }


def save_mean_reward_comparison(frames: dict[str, pd.DataFrame]) -> None:
    order = ["dqn", "ppo", "a2c", "reinforce"]
    means = [frames[a]["mean_eval_reward"].mean() for a in order]
    stds = [frames[a]["mean_eval_reward"].std() for a in order]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(order, means, yerr=stds, capsize=6)
    plt.title("Average Evaluation Reward by Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Evaluation Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_mean_reward_comparison.png", dpi=200)
    plt.close()


def save_runwise_rewards(frames: dict[str, pd.DataFrame]) -> None:
    order = ["dqn", "ppo", "a2c", "reinforce"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for ax, algo in zip(axes.flatten(), order):
        df = frames[algo].sort_values("run_id")
        ax.plot(df["run_id"], df["mean_eval_reward"], marker="o")
        ax.set_title(algo.upper())
        ax.set_xlabel("Run ID")
        ax.set_ylabel("Mean Eval Reward")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Run-by-Run Reward Curves")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_runwise_reward_curves.png", dpi=200)
    plt.close(fig)


def save_reward_distribution(frames: dict[str, pd.DataFrame]) -> None:
    order = ["dqn", "ppo", "a2c", "reinforce"]
    data = [frames[a]["mean_eval_reward"].values for a in order]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[a.upper() for a in order], showmeans=True)
    plt.title("Reward Distribution Across Hyperparameter Runs")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Evaluation Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_reward_distribution_boxplot.png", dpi=200)
    plt.close()


def save_best_runs(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for algo, df in frames.items():
        best_idx = df["mean_eval_reward"].idxmax()
        best_row = df.loc[best_idx].to_dict()
        best_row["algo"] = algo
        rows.append(best_row)

    best = pd.DataFrame(rows).sort_values("mean_eval_reward", ascending=False)
    best.to_csv(FIGURES_DIR / "04_best_runs_summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(best["algo"].str.upper(), best["mean_eval_reward"])
    plt.title("Best Run per Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Best Mean Evaluation Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, best["mean_eval_reward"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_best_run_comparison.png", dpi=200)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_data()
    save_mean_reward_comparison(frames)
    save_runwise_rewards(frames)
    save_reward_distribution(frames)
    save_best_runs(frames)
    print(f"Saved report figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
