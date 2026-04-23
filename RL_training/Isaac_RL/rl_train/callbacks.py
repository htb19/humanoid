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
            "mean_final_distance",
            "mean_min_distance",
            "mean_final_brick_height",
            "mean_action_magnitude",
            "mean_last_action_magnitude",
            "reward_action_penalty",
            "reward_distance",
            "reward_approach",
            "reward_reach_bonus",
            "reward_grasp_bonus",
            "reward_success_bonus",
            "reward_lift_bonus",
            "reward_height_bonus",
            "reward_velocity_penalty",
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
                    # --- Per-term reward diagnostics ---
                    "reward_action_penalty": float(info.get("reward_action_penalty", 0.0)),
                    "reward_distance": float(info.get("reward_distance", 0.0)),
                    "reward_approach": float(info.get("reward_approach", 0.0)),
                    "reward_reach_bonus": float(info.get("reward_reach_bonus", 0.0)),
                    "reward_grasp_bonus": float(info.get("reward_grasp_bonus", 0.0)),
                    "reward_success_bonus": float(info.get("reward_success_bonus", 0.0)),
                    "reward_lift_bonus": float(info.get("reward_lift_bonus", 0.0)),
                    "reward_height_bonus": float(info.get("reward_height_bonus", 0.0)),
                    "reward_velocity_penalty": float(info.get("reward_velocity_penalty", 0.0)),
                    "distance_to_brick": float(
                        info.get("final_distance_to_brick", info.get("distance_to_brick", 0.0))
                    ),
                    "min_episode_distance": float(info.get("min_episode_distance", 0.0)),
                    "brick_height": float(info.get("final_brick_height", info.get("brick_height", 0.0))),
                    "action_magnitude": float(info.get("action_magnitude", 0.0)),
                    "last_action_magnitude": float(info.get("last_action_magnitude", 0.0)),
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
        distances = [item.get("distance_to_brick", 0.0) for item in self._episode_history]
        min_distances = [item.get("min_episode_distance", 0.0) for item in self._episode_history]
        heights = [item.get("brick_height", 0.0) for item in self._episode_history]
        action_magnitudes = [item.get("action_magnitude", 0.0) for item in self._episode_history]
        last_action_magnitudes = [item.get("last_action_magnitude", 0.0) for item in self._episode_history]

        metrics = {
            "mean_episode_reward": _safe_mean(rewards),
            "mean_episode_length": _safe_mean(lengths),
            "success_rate": _safe_mean(success),
            "reached_rate": _safe_mean(reached),
            "grasped_rate": _safe_mean(grasped),
            "lifted_rate": _safe_mean(lifted),
            "stable_hold_rate": _safe_mean(stable_hold),
            "mean_final_distance": _safe_mean(distances),
            "mean_min_distance": _safe_mean(min_distances),
            "mean_final_brick_height": _safe_mean(heights),
            "mean_action_magnitude": _safe_mean(action_magnitudes),
            "mean_last_action_magnitude": _safe_mean(last_action_magnitudes),
            "completed_episodes": float(self.completed_episodes),
            "update": float(self.update_idx),
            "timesteps": float(self.num_timesteps),
        }
        for key, value in metrics.items():
            self.logger.record(f"robotics/{key}", value)

        # --- Per-term reward diagnostics for tensorboard ---
        reward_term_keys = [
            "reward_action_penalty", "reward_distance", "reward_approach",
            "reward_reach_bonus", "reward_grasp_bonus", "reward_success_bonus",
            "reward_lift_bonus", "reward_height_bonus", "reward_velocity_penalty",
        ]
        for term_key in reward_term_keys:
            values = [item.get(term_key, 0.0) for item in self._episode_history]
            self.logger.record(f"reward_terms/{term_key}", _safe_mean(values))

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
            "mean_final_distance": metrics["mean_final_distance"],
            "mean_min_distance": metrics["mean_min_distance"],
            "mean_final_brick_height": metrics["mean_final_brick_height"],
            "mean_action_magnitude": metrics["mean_action_magnitude"],
            "mean_last_action_magnitude": metrics["mean_last_action_magnitude"],
            "reward_action_penalty": logger_values.get("reward_terms/reward_action_penalty", float("nan")),
            "reward_distance": logger_values.get("reward_terms/reward_distance", float("nan")),
            "reward_approach": logger_values.get("reward_terms/reward_approach", float("nan")),
            "reward_reach_bonus": logger_values.get("reward_terms/reward_reach_bonus", float("nan")),
            "reward_grasp_bonus": logger_values.get("reward_terms/reward_grasp_bonus", float("nan")),
            "reward_success_bonus": logger_values.get("reward_terms/reward_success_bonus", float("nan")),
            "reward_lift_bonus": logger_values.get("reward_terms/reward_lift_bonus", float("nan")),
            "reward_height_bonus": logger_values.get("reward_terms/reward_height_bonus", float("nan")),
            "reward_velocity_penalty": logger_values.get("reward_terms/reward_velocity_penalty", float("nan")),
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
                f"reached={row['reached_rate']:.2%} "
                f"grasped={row['grasped_rate']:.2%} "
                f"lifted={row['lifted_rate']:.2%} "
                f"dist={row['mean_final_distance']:.3f} "
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
        reaching_only: bool = False,
        reach_distance_threshold: float | None = None,
        reach_threshold_phase: int | None = None,
        use_grasp_tcp: str | bool | None = None,
        arm_action_scale: float | None = None,
        reaching_brick_range: dict[str, float] | None = None,
        reaching_home_overrides: dict[str, float] | None = None,
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
        self.reaching_only = reaching_only
        self.reach_distance_threshold = reach_distance_threshold
        self.reach_threshold_phase = reach_threshold_phase
        self.use_grasp_tcp = use_grasp_tcp
        self.arm_action_scale = arm_action_scale
        self.reaching_brick_range = reaching_brick_range
        self.reaching_home_overrides = reaching_home_overrides
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
        if self.reaching_only:
            command.extend(["--reaching-only", "true"])
        if self.reach_distance_threshold is not None:
            command.extend(["--reach-threshold", str(self.reach_distance_threshold)])
        if self.reach_threshold_phase is not None:
            command.extend(["--reach-threshold-phase", str(self.reach_threshold_phase)])
        if self.use_grasp_tcp is not None:
            command.extend(["--use-grasp-tcp", str(self.use_grasp_tcp).lower()])
        if self.arm_action_scale is not None:
            command.extend(["--arm-action-scale", str(self.arm_action_scale)])
        if self.reaching_brick_range is not None:
            command.extend([
                "--reaching-brick-x-range",
                f"{self.reaching_brick_range['x_min']},{self.reaching_brick_range['x_max']}",
                "--reaching-brick-y-range",
                f"{self.reaching_brick_range['y_min']},{self.reaching_brick_range['y_max']}",
            ])
        if self.reaching_home_overrides:
            command.extend([
                "--reaching-home-overrides",
                ",".join(f"{joint}={value}" for joint, value in self.reaching_home_overrides.items()),
            ])
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
