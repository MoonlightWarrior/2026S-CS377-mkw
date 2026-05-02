"""Run a trained 1p PPO checkpoint as the policy for ALL 12 karts in the 12p env.

The checkpoint was trained on the original Vlab-WiiRL 1-player environment
(obs_dim=332 = framestack(4) × single-frame(83); action_dim=20 = Discrete(20)
with stickX(5) × R(2) × Up(1, always True) × L(2)).

Our 12p env emits a 941-wide framestack obs per env. For each kart i ∈ [0..11],
we slice out:
    race_info (5)        from obs[:, :5]
    that-kart's 78 dims  from obs[:, 5 + 78*i : 5 + 78*(i+1)]
concat across the framestack dimension to get a (4, 83) per-kart frame, flatten
to 332, batch all 12 karts together, run the PPO agent, take argmax (or sample
with --stochastic), and dispatch as a (num_envs, 12) action array.

All 12 karts share the same policy weights and the same per-kart obs normalizer.

Usage (inside the headless docker container):
    HEADLESS=1 NUM_ENVS=1 docker compose run --rm --no-deps -T \\
        -e MKW_PLAY_NUM=4 -e MKW_DISABLE_RESET=1 -e PYTHONUNBUFFERED=1 \\
        wii-rl bash -c '
          cd /src/Vlab-WiiRL
          uv sync --quiet
          uv run python -u run_ppo_12p.py \\
              --load_checkpoint checkpoints/checkpoint_update_2875.pt \\
              --steps 1000 --num_envs 1 --reset_mode race_start
        '
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from DolphinEnv import DolphinEnv
from ppo_agent import PPOAgent


NUM_KARTS = 12
RACE_INFO_DIMS = 5
PER_KART_DIMS = 78
SINGLE_FRAME_DIMS = RACE_INFO_DIMS + PER_KART_DIMS   # 83


@dataclass
class ObsNormalizer:
    """Same normalizer as kinoko-env/Vlab-WiiRL/eval_ppo.py."""
    mean: np.ndarray
    var: np.ndarray

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, obs_dim: int) -> "ObsNormalizer":
        mean = np.asarray(
            checkpoint.get("obs_rms_mean", np.zeros(obs_dim, dtype=np.float64)),
            dtype=np.float64,
        )
        var = np.asarray(
            checkpoint.get("obs_rms_var", np.ones(obs_dim, dtype=np.float64)),
            dtype=np.float64,
        )
        return cls(mean=mean, var=np.maximum(var, 1e-6))

    def normalize(self, obs: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a trained 1p PPO checkpoint as the policy for all 12 karts."
    )
    p.add_argument("--load_checkpoint", type=str, required=True)
    p.add_argument("--steps", type=int, default=1000,
                   help="Total master-step rollout length.")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reset_mode", type=str, default="race_start")
    p.add_argument("--reset_savestate", type=str, default=None)
    p.add_argument("--episode_timeout_steps", type=int, default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from the policy instead of argmax.")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def per_kart_obs(obs: np.ndarray) -> np.ndarray:
    """Slice the 941-wide framestack obs into per-kart 83-wide framestack obs.

    Input obs:    (num_envs, framestack=4, 941)
    Output:       (num_envs, NUM_KARTS, framestack, SINGLE_FRAME_DIMS=83)

    Each per-kart frame = race_info(5) + that-kart's 78 dims.
    """
    num_envs, framestack, total = obs.shape
    assert total == RACE_INFO_DIMS + PER_KART_DIMS * NUM_KARTS, \
        f"obs total dim {total} != {RACE_INFO_DIMS + PER_KART_DIMS * NUM_KARTS}"
    out = np.empty((num_envs, NUM_KARTS, framestack, SINGLE_FRAME_DIMS), dtype=np.float32)
    race_info = obs[:, :, :RACE_INFO_DIMS]                      # (E, F, 5)
    for k in range(NUM_KARTS):
        s = RACE_INFO_DIMS + PER_KART_DIMS * k
        e = s + PER_KART_DIMS
        out[:, k, :, :RACE_INFO_DIMS] = race_info
        out[:, k, :, RACE_INFO_DIMS:] = obs[:, :, s:e]
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[ppo12p] device: {device}")

    if args.num_envs > 4:
        raise SystemExit("num_envs > 4 forbidden on this host.")

    ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    obs_dim = sd["actor.0.weight"].shape[1]
    action_dim = sd["actor.4.weight"].shape[0]
    print(f"[ppo12p] checkpoint obs_dim={obs_dim} action_dim={action_dim} update={ckpt.get('update')}")

    expected_obs_dim = 4 * SINGLE_FRAME_DIMS  # framestack 4 × 83 = 332
    if obs_dim != expected_obs_dim:
        raise SystemExit(
            f"checkpoint obs_dim={obs_dim} != expected {expected_obs_dim}; "
            f"this runner only supports the framestack-4 × 83-dim per-kart layout."
        )

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    agent.load_state_dict(sd)
    agent.eval()
    obs_norm = ObsNormalizer.from_checkpoint(ckpt, obs_dim)
    print(f"[ppo12p] obs_norm mean range [{obs_norm.mean.min():.3f}, {obs_norm.mean.max():.3f}]")

    env = DolphinEnv(
        num_envs=args.num_envs,
        reset_mode=args.reset_mode,
        reset_savestate=args.reset_savestate,
        episode_timeout_steps=args.episode_timeout_steps,
    )
    obs, _ = env.reset()
    print(f"[ppo12p] env up. obs.shape={obs.shape}")

    out_csv = Path(__file__).resolve().parent / "instance_info" / "run_ppo_12p.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "env", "kart", "RaceCompletion", "kart_x", "kart_z",
                    "race_pos", "action", "reward_env", "done_env", "trun_env"])

        t0 = time.time()
        for step in range(args.steps):
            # obs: (E, 4, 941) → per-kart: (E, 12, 4, 83) → flatten: (E*12, 332)
            pk = per_kart_obs(obs)                                      # (E, 12, 4, 83)
            flat = pk.reshape(args.num_envs * NUM_KARTS, expected_obs_dim)
            normed = obs_norm.normalize(flat)                            # (E*12, 332)

            inp = torch.as_tensor(normed, dtype=torch.float32, device=device)
            with torch.no_grad():
                if args.stochastic:
                    actions, _, _, _ = agent.get_action_and_value(inp)
                else:
                    logits = agent.get_logits(inp)
                    actions = torch.argmax(logits, dim=-1)
            actions_np = actions.cpu().numpy().reshape(args.num_envs, NUM_KARTS).astype(np.int64)

            env.step_async(actions_np)
            obs, rewards, dones, truns, infos = env.step_wait()

            rc_all = infos.get("RaceCompletion_all", np.zeros((args.num_envs, NUM_KARTS), dtype=np.float32))
            kx_all = infos.get("kart_x_all",        np.zeros((args.num_envs, NUM_KARTS), dtype=np.float32))
            kz_all = infos.get("kart_z_all",        np.zeros((args.num_envs, NUM_KARTS), dtype=np.float32))
            rp_all = infos.get("race_pos_all",      np.zeros((args.num_envs, NUM_KARTS), dtype=np.int32))

            for i in range(args.num_envs):
                for k in range(NUM_KARTS):
                    w.writerow([
                        step, i, k,
                        float(rc_all[i, k]),
                        float(kx_all[i, k]),
                        float(kz_all[i, k]),
                        int(rp_all[i, k]),
                        int(actions_np[i, k]),
                        float(rewards[i]),
                        bool(dones[i]),
                        bool(truns[i]),
                    ])

            if step % max(1, args.log_every) == 0:
                elapsed = time.time() - t0
                sps = (step + 1) / max(elapsed, 1e-6)
                rc0 = rc_all[0].tolist() if args.num_envs >= 1 else []
                a0 = actions_np[0].tolist() if args.num_envs >= 1 else []
                print(
                    f"[ppo12p] step={step:5d} sps={sps:5.1f} "
                    f"rc[env0]={['%.3f'%v for v in rc0]} "
                    f"act[env0]={a0} "
                    f"r={rewards.tolist()} d={dones.astype(int).tolist()}"
                )
                f.flush()
                os.fsync(f.fileno())

    print(f"[ppo12p] done. wrote {out_csv}")


if __name__ == "__main__":
    main()
