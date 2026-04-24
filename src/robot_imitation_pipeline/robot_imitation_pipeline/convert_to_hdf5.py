#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from robot_imitation_pipeline.io_utils import list_episode_dirs, load_episode_arrays, read_json, split_state_action, write_json


def convert(raw_root: Path, output_dir: Path, val_fraction: float, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = list_episode_dirs(raw_root)
    if not episodes:
        raise SystemExit(f"No episodes found under {raw_root}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(episodes) * val_fraction))) if len(episodes) > 1 else 0
    val_set = set(indices[:val_count].tolist())

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    episode_index: List[np.ndarray] = []
    splits = {"train": [], "val": []}
    manifest = []
    for ep_idx, episode in enumerate(episodes):
        arrays = load_episode_arrays(episode)
        if "joint_pos" not in arrays or "actions" not in arrays:
            print(f"Skipping {episode}: missing joint_pos/actions")
            continue
        state, action = split_state_action(arrays)
        if len(state) == 0:
            print(f"Skipping {episode}: no valid samples")
            continue
        states.append(state)
        actions.append(action)
        episode_index.append(np.full(len(state), ep_idx, dtype=np.int64))
        split_name = "val" if ep_idx in val_set else "train"
        splits[split_name].append(ep_idx)
        meta = read_json(episode / "meta.json")
        manifest.append(
            {
                "episode_index": ep_idx,
                "episode_dir": str(episode),
                "split": split_name,
                "num_raw_samples": int(meta.get("num_samples", len(arrays.get("timestamps", [])))),
                "num_valid_samples": int(len(state)),
                "success": read_json(episode / "success.json").get("success")
                if (episode / "success.json").exists()
                else None,
            }
        )

    if not states:
        raise SystemExit("No valid samples found.")
    state_all = np.concatenate(states, axis=0).astype(np.float32)
    action_all = np.concatenate(actions, axis=0).astype(np.float32)
    episode_index_all = np.concatenate(episode_index, axis=0)
    np.savez_compressed(
        output_dir / "dataset.npz",
        obs_state=state_all,
        actions=action_all,
        episode_index=episode_index_all,
    )
    write_json(
        output_dir / "splits.json",
        {"train_episode_indices": splits["train"], "val_episode_indices": splits["val"]},
    )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    try:
        import h5py
    except ImportError:
        print("h5py is not installed; wrote dataset.npz only.")
        return
    with h5py.File(output_dir / "dataset.hdf5", "w") as h5:
        h5.create_dataset("obs/state", data=state_all, compression="gzip")
        h5.create_dataset("actions", data=action_all, compression="gzip")
        h5.create_dataset("episode_index", data=episode_index_all, compression="gzip")
        h5.attrs["format"] = "robot_imitation_pipeline_state_action_v0"
        h5.attrs["action_mode"] = "joint_position_targets"
    print(f"Wrote {len(state_all)} samples to {output_dir}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Convert raw demo episodes to a training dataset.")
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/imitation_converted"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    convert(args.raw_root, args.output_dir, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
