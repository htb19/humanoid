from __future__ import annotations

import csv
from collections import deque
import json
from pathlib import Path
import subprocess
from statistics import mean
import sys
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


class TrainingMonitorCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        print_freq_updates: int = 5,
        rolling_window: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.print_freq_updates = print_freq_updates
        self.rolling_window = rolling_window
        self.completed_episodes = 0
        self.update_idx = 0
        self._episode_history: deque[dict[str, float]] = deque(maxlen=rolling_window)
        self._csv_path = self.run_dir / "update_metrics.csv"
        self._csv_handle = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_handle = self._csv_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "update",
            "timesteps",
            "completed_episodes",
            "mean_episode_reward",
            "mean_episode_length",
            "success_rate",
            "reached_rate",
            "grasped_rate",
            "lifted_rate",
            "stable_hold_rate",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "learning_rate",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
        ]
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.completed_episodes += 1
            self._episode_history.append(
                {
                    "episode_reward": float(info.get("episode_reward", info.get("episode", {}).get("r", 0.0))),
                    "episode_length": float(info.get("episode_length", info.get("episode", {}).get("l", 0.0))),
                    "is_success": float(info.get("is_success", 0.0)),
                    "reached_object": float(info.get("reached_object", 0.0)),
                    "grasped_object": float(info.get("grasped_object", 0.0)),
                    "lifted_object": float(info.get("lifted_object", 0.0)),
                    "stable_hold": float(info.get("stable_hold", 0.0)),
                }
            )
        return True

    def _on_rollout_end(self) -> None:
        self.update_idx += 1
        rewards = [item["episode_reward"] for item in self._episode_history]
        lengths = [item["episode_length"] for item in self._episode_history]
        success = [item["is_success"] for item in self._episode_history]
        reached = [item["reached_object"] for item in self._episode_history]
        grasped = [item["grasped_object"] for item in self._episode_history]
        lifted = [item["lifted_object"] for item in self._episode_history]
        stable_hold = [item["stable_hold"] for item in self._episode_history]

        metrics = {
            "mean_episode_reward": _safe_mean(rewards),
            "mean_episode_length": _safe_mean(lengths),
            "success_rate": _safe_mean(success),
            "reached_rate": _safe_mean(reached),
            "grasped_rate": _safe_mean(grasped),
            "lifted_rate": _safe_mean(lifted),
            "stable_hold_rate": _safe_mean(stable_hold),
            "completed_episodes": float(self.completed_episodes),
            "update": float(self.update_idx),
            "timesteps": float(self.num_timesteps),
        }
        for key, value in metrics.items():
            self.logger.record(f"robotics/{key}", value)

        logger_values = self.logger.name_to_value
        row = {
            "update": self.update_idx,
            "timesteps": self.num_timesteps,
            "completed_episodes": self.completed_episodes,
            "mean_episode_reward": metrics["mean_episode_reward"],
            "mean_episode_length": metrics["mean_episode_length"],
            "success_rate": metrics["success_rate"],
            "reached_rate": metrics["reached_rate"],
            "grasped_rate": metrics["grasped_rate"],
            "lifted_rate": metrics["lifted_rate"],
            "stable_hold_rate": metrics["stable_hold_rate"],
            "policy_loss": logger_values.get("train/policy_gradient_loss", float("nan")),
            "value_loss": logger_values.get("train/value_loss", float("nan")),
            "entropy_loss": logger_values.get("train/entropy_loss", float("nan")),
            "learning_rate": logger_values.get("train/learning_rate", float("nan")),
            "approx_kl": logger_values.get("train/approx_kl", float("nan")),
            "clip_fraction": logger_values.get("train/clip_fraction", float("nan")),
            "explained_variance": logger_values.get("train/explained_variance", float("nan")),
        }
        self._csv_writer.writerow(row)
        self._csv_handle.flush()

        if self.update_idx % self.print_freq_updates == 0:
            print(
                "[train] "
                f"update={self.update_idx} "
                f"timesteps={self.num_timesteps} "
                f"episodes={self.completed_episodes} "
                f"mean_reward={row['mean_episode_reward']:.2f} "
                f"mean_len={row['mean_episode_length']:.1f} "
                f"success={row['success_rate']:.2%} "
                f"grasped={row['grasped_rate']:.2%} "
                f"lifted={row['lifted_rate']:.2%} "
                f"policy_loss={row['policy_loss']:.4f} "
                f"value_loss={row['value_loss']:.4f}"
            )

    def _on_training_end(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.close()


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        eval_script: Path,
        model_dir: Path,
        eval_freq_steps: int = 20_000,
        n_eval_episodes: int = 5,
        robot_description_path: Path | None = None,
        headless: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.eval_script = eval_script
        self.model_dir = model_dir
        self.eval_freq_steps = eval_freq_steps
        self.n_eval_episodes = n_eval_episodes
        self.robot_description_path = robot_description_path
        self.headless = headless
        self.last_eval_step = 0
        self.eval_idx = 0
        self.best_success_rate = -np.inf
        self.best_mean_reward = -np.inf
        self.eval_dir = self.run_dir / "evaluations"
        self.best_model_dir = self.run_dir / "best_model"

    def _on_training_start(self) -> None:
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "recommended_eval_model.txt").open("w", encoding="utf-8") as handle:
            handle.write(str(self.model_dir / "ppo_humanoid_brick_latest.zip"))

    def _should_eval(self) -> bool:
        return self.eval_freq_steps > 0 and (self.num_timesteps - self.last_eval_step) >= self.eval_freq_steps

    def _load_summary_file(self, summary_path: Path) -> dict[str, Any] | None:
        if not summary_path.exists():
            print(f"[eval][warn] summary file was not created: {summary_path}")
            return None
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[eval][warn] failed to parse summary JSON at {summary_path}: {exc}")
            return None
        except OSError as exc:
            print(f"[eval][warn] failed to read summary file at {summary_path}: {exc}")
            return None

    def _run_eval(self) -> dict[str, Any] | None:
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        candidate_model = self.model_dir / "eval_candidate"
        self.model.save(str(candidate_model))
        summary_path = self.eval_dir / f"eval_{self.eval_idx:04d}_{self.num_timesteps:09d}.json"
        command = [
            sys.executable,
            str(self.eval_script),
            "--model",
            str(candidate_model) + ".zip",
            "--episodes",
            str(self.n_eval_episodes),
            "--headless",
            str(self.headless).lower(),
            "--summary-out",
            str(summary_path),
        ]
        if self.robot_description_path is not None:
            command.extend(["--robot-description-path", str(self.robot_description_path)])
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[eval][warn] evaluation subprocess failed with exit code {exc.returncode}")
            return None
        except OSError as exc:
            print(f"[eval][warn] failed to launch evaluation subprocess: {exc}")
            return None
        return self._load_summary_file(summary_path)

    def _on_step(self) -> bool:
        if not self._should_eval():
            return True
        self.last_eval_step = self.num_timesteps
        self.eval_idx += 1
        summary = self._run_eval()
        if summary is None:
            print(
                "[eval][warn] "
                f"skipping eval logging for eval={self.eval_idx} timesteps={self.num_timesteps}"
            )
            return True

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                self.logger.record(f"eval/{key}", value)

        is_better = (
            summary["success_rate"] > self.best_success_rate
            or (
                np.isclose(summary["success_rate"], self.best_success_rate)
                and summary["mean_reward"] > self.best_mean_reward
            )
        )
        if is_better:
            self.best_success_rate = summary["success_rate"]
            self.best_mean_reward = summary["mean_reward"]
            best_model_path = self.best_model_dir / "best_model"
            self.model.save(str(best_model_path))
            with (self.best_model_dir / "best_model_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            with (self.run_dir / "recommended_eval_model.txt").open("w", encoding="utf-8") as handle:
                handle.write(str(best_model_path) + ".zip")

        print(
            "[eval] "
            f"eval={self.eval_idx} "
            f"timesteps={self.num_timesteps} "
            f"mean_reward={summary['mean_reward']:.2f} "
            f"success={summary['success_rate']:.2%} "
            f"grasped={summary['grasped_rate']:.2%} "
            f"lifted={summary['lifted_rate']:.2%}"
        )
        return True
