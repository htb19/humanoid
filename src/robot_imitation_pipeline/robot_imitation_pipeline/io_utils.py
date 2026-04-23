import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


ACTION_COMPONENTS = {
    "left_joint_target": [0, 6],
    "right_joint_target": [6, 12],
    "neck_joint_target": [12, 14],
    "gripper_open_command": [14, 16],
}


def stamp_to_float(stamp: Any) -> float:
    if stamp is None:
        return float("nan")
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def now_to_float(clock: Any) -> float:
    msg = clock.now().to_msg()
    return stamp_to_float(msg)


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def next_episode_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("episode_"):
            try:
                existing.append(int(child.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    next_idx = max(existing, default=0) + 1
    path = root / f"episode_{next_idx:06d}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def load_episode_arrays(episode_dir: Path) -> Dict[str, np.ndarray]:
    episode_dir = Path(episode_dir)
    arrays: Dict[str, np.ndarray] = {}
    for name in [
        "timestamps",
        "joint_pos",
        "joint_vel",
        "actions",
        "action_valid",
        "gripper",
    ]:
        path = episode_dir / f"{name}.npy"
        if path.exists():
            arrays[name] = np.load(path, allow_pickle=False)
    return arrays


def list_episode_dirs(root: Path) -> List[Path]:
    root = Path(root)
    if root.name.startswith("episode_") and root.is_dir():
        return [root]
    return sorted(p for p in root.glob("episode_*") if p.is_dir())


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if len(source_times) == 0:
        return np.full(len(target_times), -1, dtype=np.int64)
    indices = np.searchsorted(source_times, target_times)
    indices = np.clip(indices, 0, len(source_times) - 1)
    prev_indices = np.clip(indices - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(source_times[prev_indices] - target_times) < np.abs(
        source_times[indices] - target_times
    )
    indices[choose_prev] = prev_indices[choose_prev]
    return indices.astype(np.int64)


def maybe_load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML config files") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def nested_get(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def split_state_action(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    state = arrays["joint_pos"].astype(np.float32)
    action = arrays["actions"].astype(np.float32)
    mask = np.isfinite(state).all(axis=1) & np.isfinite(action).all(axis=1)
    if "action_valid" in arrays:
        valid = arrays["action_valid"]
        if valid.ndim == 2:
            mask &= valid.all(axis=1)
        else:
            mask &= valid.astype(bool)
    return state[mask], action[mask]
