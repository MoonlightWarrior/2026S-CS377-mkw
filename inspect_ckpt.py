"""Inspect a PPO checkpoint to find expected obs_dim, action_dim, and obs_rms shape."""
import sys
import torch
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/checkpoint_update_2875.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("Top-level keys:", list(ckpt.keys()))
sd = ckpt["model_state_dict"]
for k, v in sd.items():
    print(f"  {k}: {tuple(v.shape)}")

if "obs_rms_mean" in ckpt:
    m = np.asarray(ckpt["obs_rms_mean"])
    var = np.asarray(ckpt["obs_rms_var"])
    print(f"obs_rms_mean shape={m.shape}  min={m.min():.4f} max={m.max():.4f}")
    print(f"obs_rms_var  shape={var.shape}  min={var.min():.4f} max={var.max():.4f}")
    print(f"obs_rms_count: {ckpt.get('obs_rms_count')}")

# infer obs_dim and action_dim from actor weights
actor_first = sd["actor.0.weight"]   # (hidden, obs_dim)
actor_last = None
for k in sd:
    if k.startswith("actor.") and k.endswith(".weight"):
        actor_last = k
print(f"\nactor.0.weight: in_features={actor_first.shape[1]} → obs_dim={actor_first.shape[1]}")
last = sd[actor_last]
print(f"{actor_last}: out_features={last.shape[0]} → action_dim={last.shape[0]}")
