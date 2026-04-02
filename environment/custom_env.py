from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    max_steps: int = 500
    max_queue: int = 60
    max_init_queue: int = 12
    base_arrival_rate: float = 1.3
    peak_arrival_rate: float = 2.4
    discharge_green: int = 3
    discharge_all_red: int = 0
    alpha_passed: float = 1.2
    beta_switch: float = 0.35
    seed: Optional[int] = None


class KigaliTrafficEnv(gym.Env):
    """Single-intersection traffic light control environment."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=float(self.config.max_queue), shape=(4,), dtype=np.float32)

        self.queues = np.zeros(4, dtype=np.int32)
        self.last_action = -1
        self.step_count = 0
        self.total_wait_time = 0.0
        self.total_passed = 0
        self.total_switches = 0
        self.rng = np.random.default_rng(self.config.seed)

    def _traffic_rate(self) -> float:
        # Periodic peak pattern to emulate off-peak vs peak arrivals.
        phase = (self.step_count % 120) / 120.0
        curve = 0.5 * (1.0 + np.sin(2.0 * np.pi * phase - np.pi / 2.0))
        return self.config.base_arrival_rate + curve * (self.config.peak_arrival_rate - self.config.base_arrival_rate)

    def _arrivals(self) -> np.ndarray:
        rate = self._traffic_rate()
        arrivals = self.rng.poisson(lam=rate, size=4)
        return arrivals.astype(np.int32)

    def _departures(self, action: int) -> Tuple[np.ndarray, int]:
        departures = np.zeros(4, dtype=np.int32)
        capacity = self.config.discharge_green if action in (0, 1) else self.config.discharge_all_red

        if action == 0:
            # North-South green
            departures[0] = min(self.queues[0], capacity)
            departures[1] = min(self.queues[1], capacity)
        elif action == 1:
            # East-West green
            departures[2] = min(self.queues[2], capacity)
            departures[3] = min(self.queues[3], capacity)

        cars_passed = int(departures.sum())
        return departures, cars_passed

    def _get_obs(self) -> np.ndarray:
        return self.queues.astype(np.float32)

    def _get_info(self, reward: float, cars_passed: int, switched: int) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "queues": self.queues.copy(),
            "reward": reward,
            "cars_passed": cars_passed,
            "switched": switched,
            "total_wait_time": self.total_wait_time,
            "total_passed": self.total_passed,
            "total_switches": self.total_switches,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.queues = self.rng.integers(0, self.config.max_init_queue + 1, size=4, dtype=np.int32)
        self.last_action = -1
        self.step_count = 0
        self.total_wait_time = 0.0
        self.total_passed = 0
        self.total_switches = 0

        obs = self._get_obs()
        info = self._get_info(reward=0.0, cars_passed=0, switched=0)
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        arrivals = self._arrivals()
        departures, cars_passed = self._departures(action)
        switched = int(self.last_action != -1 and self.last_action != action)

        self.queues = self.queues + arrivals - departures
        self.queues = np.clip(self.queues, 0, self.config.max_queue).astype(np.int32)

        wait_time = float(self.queues.sum())
        switch_penalty = self.config.beta_switch * switched
        reward = -wait_time + (self.config.alpha_passed * cars_passed) - switch_penalty

        self.step_count += 1
        self.last_action = action
        self.total_wait_time += wait_time
        self.total_passed += cars_passed
        self.total_switches += switched

        overflow = bool((self.queues >= self.config.max_queue).any())
        terminated = overflow
        truncated = self.step_count >= self.config.max_steps

        info = self._get_info(reward=reward, cars_passed=cars_passed, switched=switched)
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return (
                f"step={self.step_count} queues={self.queues.tolist()} "
                f"last_action={self.last_action} total_passed={self.total_passed}"
            )
        return None

    def close(self):
        return None
