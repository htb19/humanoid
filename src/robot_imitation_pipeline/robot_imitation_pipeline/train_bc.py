#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from robot_imitation_pipeline.io_utils import maybe_load_yaml, read_json


def _load_config(path: Path) -> Dict:
    data = maybe_load_yaml(path)
    return data or {}


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_model(input_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
    import torch.nn as nn

    layers = []
    last = input_dim
    for _ in range(num_layers):
        layers += [nn.Linear(last, hidden_dim), nn.ReLU()]
        last = hidden_dim
    layers.append(nn.Linear(last, action_dim))
    return nn.Sequential(*layers)


def _load_dataset(dataset_path: Path, splits_path: Path | None, val_fraction: float, seed: int) -> Tuple:
    data = np.load(dataset_path)
    obs = data["obs_state"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    episode_index = data["episode_index"].astype(np.int64)
    if splits_path and splits_path.exists():
        splits = read_json(splits_path)
        val_eps = set(splits.get("val_episode_indices", []))
        val_mask = np.asarray([int(ep) in val_eps for ep in episode_index], dtype=bool)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(obs))
        val_count = int(round(len(obs) * val_fraction))
        val_mask = np.zeros(len(obs), dtype=bool)
        val_mask[perm[:val_count]] = True
    train_mask = ~val_mask
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise SystemExit("Train/val split is empty; collect more episodes or change val_fraction.")
    return obs[train_mask], actions[train_mask], obs[val_mask], actions[val_mask]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a minimal state-to-action BC baseline.")
    parser.add_argument("--config", type=Path, default=Path("src/robot_imitation_pipeline/config/training.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch is required for train_bc.py") from exc

    cfg = _load_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    dataset_path = args.dataset or Path(dataset_cfg.get("path", "data/imitation_converted/dataset.npz"))
    output_dir = args.output_dir or Path(train_cfg.get("output_dir", "data/imitation_runs/bc_state"))
    splits_path = dataset_path.parent / "splits.json"

    seed = int(train_cfg.get("seed", 7))
    torch.manual_seed(seed)
    train_obs, train_actions, val_obs, val_actions = _load_dataset(
        dataset_path,
        splits_path,
        float(train_cfg.get("val_fraction", 0.15)),
        seed,
    )

    input_dim = int(model_cfg.get("input_dim") or train_obs.shape[1])
    action_dim = int(model_cfg.get("action_dim") or train_actions.shape[1])
    model = _make_model(
        input_dim,
        action_dim,
        int(model_cfg.get("hidden_dim", 256)),
        int(model_cfg.get("num_layers", 3)),
    )
    device = _device(str(train_cfg.get("device", "auto")))
    model.to(device)

    batch_size = int(train_cfg.get("batch_size", 128))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_obs), torch.from_numpy(train_actions)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_obs_t = torch.from_numpy(val_obs).to(device)
    val_actions_t = torch.from_numpy(val_actions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs = int(train_cfg.get("epochs", 50))
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for obs, action in train_loader:
            obs = obs.to(device)
            action = action.to(device)
            pred = model(obs)
            loss = F.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(val_obs_t), val_actions_t).detach().cpu())
        train_loss = float(np.mean(train_losses))
        print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        checkpoint = {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "action_dim": action_dim,
            "config": cfg,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Saved checkpoints to {output_dir}")
    print(f"Input shape: state vector ({input_dim},), action shape: ({action_dim},)")


if __name__ == "__main__":
    main()
