#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from robot_imitation_pipeline.io_utils import list_episode_dirs, load_episode_arrays, read_json


def _rate(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    duration = float(times[-1] - times[0])
    return 0.0 if duration <= 0 else float((len(times) - 1) / duration)


def _max_gap(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    return float(np.max(np.diff(times)))


def validate_episode(episode_dir: Path, max_gap_sec: float) -> List[str]:
    issues: List[str] = []
    meta_path = episode_dir / "meta.json"
    success_path = episode_dir / "success.json"
    if not meta_path.exists():
        issues.append("missing meta.json")
        return issues
    if not success_path.exists():
        issues.append("missing success.json")
    meta = read_json(meta_path)
    arrays = load_episode_arrays(episode_dir)
    required = ["timestamps", "joint_pos", "joint_vel", "actions", "action_valid", "gripper"]
    for name in required:
        if name not in arrays:
            issues.append(f"missing {name}.npy")
    if issues:
        return issues
    n = len(arrays["timestamps"])
    for name in ["joint_pos", "joint_vel", "actions", "action_valid", "gripper"]:
        if arrays[name].shape[0] != n:
            issues.append(f"{name}.npy length {arrays[name].shape[0]} != timestamps length {n}")
    if n == 0:
        issues.append("episode has zero state/action samples")
        return issues
    if arrays["joint_pos"].ndim != 2 or arrays["joint_pos"].shape[1] != 16:
        issues.append(f"joint_pos shape should be Nx16, got {arrays['joint_pos'].shape}")
    if arrays["actions"].ndim != 2 or arrays["actions"].shape[1] != 16:
        issues.append(f"actions shape should be Nx16, got {arrays['actions'].shape}")
    timestamps = arrays["timestamps"]
    if np.any(np.diff(timestamps) <= 0):
        issues.append("timestamps are not strictly increasing")
    gap = _max_gap(timestamps)
    if gap > max_gap_sec:
        issues.append(f"state/action max timestamp gap {gap:.3f}s > {max_gap_sec:.3f}s")
    if not np.isfinite(arrays["joint_pos"]).all():
        issues.append("joint_pos contains NaN/Inf")
    valid = arrays["action_valid"]
    if valid.ndim == 2 and not valid.all(axis=1).all():
        invalid_count = int((~valid.all(axis=1)).sum())
        issues.append(f"{invalid_count} samples have incomplete action components")
    for camera_name, expected_count in meta.get("camera_frame_counts", {}).items():
        camera_dir = episode_dir / camera_name
        frames = sorted(camera_dir.glob("frame_*.*"))
        if len(frames) != int(expected_count):
            issues.append(f"{camera_name} image count {len(frames)} != meta count {expected_count}")
        ts_path = episode_dir / f"{camera_name}_timestamps.npy"
        if ts_path.exists():
            image_times = np.load(ts_path, allow_pickle=False)
            if len(image_times) != len(frames):
                issues.append(f"{camera_name} timestamps length {len(image_times)} != image count {len(frames)}")
        elif expected_count:
            issues.append(f"missing {camera_name}_timestamps.npy")
    return issues


def print_stats(episode_dir: Path) -> None:
    meta = read_json(episode_dir / "meta.json")
    arrays = load_episode_arrays(episode_dir)
    success = read_json(episode_dir / "success.json").get("success") if (episode_dir / "success.json").exists() else None
    timestamps = arrays.get("timestamps", np.zeros((0,), dtype=np.float64))
    print(f"\n{episode_dir}")
    print(f"  success: {success}")
    print(f"  samples: {len(timestamps)}")
    print(f"  duration_sec: {meta.get('duration_sec', 0.0):.3f}")
    print(f"  sample_rate_hz: {_rate(timestamps):.2f}")
    print(f"  max_sample_gap_sec: {_max_gap(timestamps):.3f}")
    for camera_name, count in meta.get("camera_frame_counts", {}).items():
        ts_path = episode_dir / f"{camera_name}_timestamps.npy"
        image_rate = 0.0
        if ts_path.exists():
            image_rate = _rate(np.load(ts_path, allow_pickle=False))
        print(f"  {camera_name}: {count} frames, {image_rate:.2f} Hz")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Validate recorded imitation episodes.")
    parser.add_argument("path", type=Path, help="Episode directory or raw dataset root.")
    parser.add_argument("--max-gap-sec", type=float, default=0.35)
    args = parser.parse_args(argv)

    episodes = list_episode_dirs(args.path)
    if not episodes:
        raise SystemExit(f"No episode directories found under {args.path}")
    total_issues: Dict[str, List[str]] = {}
    for episode in episodes:
        issues = validate_episode(episode, args.max_gap_sec)
        print_stats(episode)
        if issues:
            total_issues[str(episode)] = issues
            print("  issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  issues: none")
    if total_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
