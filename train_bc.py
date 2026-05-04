"""Behaviour-cloning trainer for the CPU-distillation dataset.

Reads the NPZ produced by collect_distill_data.py and trains a small MLP
π(action | per-kart obs + race info) with cross-entropy loss. Saves a
checkpoint that downstream code can either:
  - sample from directly (greedy / temperature) as a CPU-AI imitator, or
  - use as a warm-start for PPO fine-tuning (state encoder + action head
    align with the existing 20-action discrete space).

Default architecture: 83 → 256 → 256 → 20 (ReLU). Adam, CE loss, 80/20
train/val split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from cpu_action_inference import N_ACTIONS

PER_KART_OBS_DIMS = 78
RACE_INFO_DIMS = 5
INPUT_DIMS = PER_KART_OBS_DIMS + RACE_INFO_DIMS  # 83


class BCPolicy(nn.Module):
    def __init__(self, hidden: int = 256, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIMS, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="instance_info/distill_dataset.npz")
    p.add_argument("--out",  type=str, default="checkpoints/bc_policy.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--standardize", action="store_true",
                   help="Standardize inputs (zero-mean unit-variance) using train-split stats.")
    return p.parse_args()


def load_dataset(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    obs = data["obs"].astype(np.float32)
    race = data["race_info"].astype(np.float32)
    actions = data["action"].astype(np.int64)
    if obs.ndim != 2 or obs.shape[1] != PER_KART_OBS_DIMS:
        raise ValueError(f"unexpected obs shape {obs.shape}; expected (N, {PER_KART_OBS_DIMS})")
    if race.ndim != 2 or race.shape[1] != RACE_INFO_DIMS:
        raise ValueError(f"unexpected race_info shape {race.shape}; expected (N, {RACE_INFO_DIMS})")
    if actions.ndim != 1 or actions.shape[0] != obs.shape[0]:
        raise ValueError(f"action shape {actions.shape} doesn't match obs[0]={obs.shape[0]}")
    inputs = np.concatenate([race, obs], axis=1)  # (N, 83) — race first to match collect_distill ordering
    return inputs, actions


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_path = (project_root / args.data).resolve()
    out_path  = (project_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[bc] loading {data_path}")
    inputs, actions = load_dataset(data_path)
    n = len(actions)
    print(f"[bc] dataset: {n} samples, action distribution:")
    counts = np.bincount(actions, minlength=N_ACTIONS)
    for a, c in enumerate(counts):
        if c:
            print(f"      action {a:3d}: {c:6d}  ({100*c/n:5.2f}%)")

    if n < 100:
        raise SystemExit(f"[bc] too few samples ({n}); collect more before training.")

    # Optional standardisation. Compute stats over train split only to avoid leakage.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(round(n * args.val_split))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    if args.standardize:
        mean = inputs[train_idx].mean(axis=0, keepdims=True)
        std  = inputs[train_idx].std(axis=0,  keepdims=True) + 1e-6
        inputs = (inputs - mean) / std
    else:
        mean = np.zeros((1, INPUT_DIMS), dtype=np.float32)
        std  = np.ones((1, INPUT_DIMS),  dtype=np.float32)

    inputs_t = torch.from_numpy(inputs).float()
    actions_t = torch.from_numpy(actions).long()
    full = TensorDataset(inputs_t, actions_t)
    train_set = torch.utils.data.Subset(full, train_idx.tolist())
    val_set   = torch.utils.data.Subset(full, val_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    model = BCPolicy(hidden=args.hidden).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * xb.size(0)
            train_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            train_n += xb.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                val_correct += int((logits.argmax(dim=-1) == yb).sum().item())
                val_n += xb.size(0)

        train_acc = train_correct / max(train_n, 1)
        val_acc   = val_correct   / max(val_n,   1)
        print(f"[bc] epoch {epoch:3d} | "
              f"train loss={train_loss/max(train_n,1):.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss/max(val_n,1):.4f} acc={val_acc:.3f} | "
              f"{time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt = {
        "model_state_dict": best_state,
        "config": {
            "input_dims":   INPUT_DIMS,
            "hidden":       args.hidden,
            "n_actions":    N_ACTIONS,
            "standardize":  bool(args.standardize),
            "input_mean":   mean.tolist(),
            "input_std":    std.tolist(),
        },
        "best_val_acc": float(best_val_acc),
        "n_train":      int(len(train_set)),
        "n_val":        int(len(val_set)),
    }
    torch.save(ckpt, out_path)
    print(f"[bc] saved checkpoint → {out_path}  (best val acc={best_val_acc:.3f})")
    # Also write a tiny JSON sidecar for quick inspection.
    sidecar = out_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({k: ckpt[k] for k in ("config", "best_val_acc", "n_train", "n_val")}, f, indent=2)


if __name__ == "__main__":
    main()
