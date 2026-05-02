"""12-player random-policy demo for the MKW input-override hack.

Phases:
  probe    — read-only. The slave is launched with MKW_PROBE_INPUT=1 (set by the
             caller via -e), so it dumps per-frame InputState reads to
             instance_info/input_probe_env*.csv. We send dummy zero actions to
             keep the loop pumping. Used to verify the documented InputData layout
             against this ROM (Phase 1 of the plan).
  sentinel — write a constant input only to one slot (--slot, default 4) and
             leave the other slots' actions at 0. Use to confirm a single CPU
             kart's behavior changes when our writes happen.
  constant — every kart written stickX=0, accel-only (action 16 in the
             5*2*2*2 decode: stick_idx=2 (=0.0), R=False, Up=False, L=False).
  random   — every kart samples Discrete(40) independently per step. Default.

Trajectory CSVs (random12_run.csv etc.) are dumped to instance_info/ for
offline inspection. No wandb, no PPO, no learning.

Run via the project's headless docker stack:

    HEADLESS=1 NUM_ENVS=1 docker compose run --rm \\
        -e MKW_PROBE_INPUT=1 wii-rl python run_random_12p.py --phase probe

    HEADLESS=1 NUM_ENVS=1 docker compose run --rm \\
        wii-rl python run_random_12p.py --phase sentinel --slot 4

    HEADLESS=1 NUM_ENVS=1 docker compose run --rm \\
        wii-rl python run_random_12p.py --phase random --steps 600
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from DolphinEnv import DolphinEnv


NUM_KARTS = 12

# Constant-action mapping: stickX=0.0, R=False, Up=False, L=False
# decode: stride = 2*2*2 = 8; action = stick_idx*8 + r_idx*4 + up_idx*2 + l_idx
#   stick_idx=2 (stickX_values[2] = 0)  → 16
ACTION_FORWARD_STRAIGHT = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase",
                   choices=["probe", "sentinel", "constant", "random", "all_left", "all_right"],
                   default="random")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--slot", type=int, default=4, help="Sentinel-write target slot (sentinel phase only)")
    p.add_argument("--sentinel_action", type=int, default=36,
                   help="Action to write to --slot. Default 36 = stickX=1.0, R=False, Up=False, L=False")
    p.add_argument("--reset_mode", type=str, default="savestate")
    p.add_argument("--reset_savestate", type=str, default=None)
    p.add_argument("--episode_timeout_steps", type=int, default=700)
    p.add_argument("--log_every", type=int, default=30)
    return p.parse_args()


def actions_for_phase(phase: str, num_envs: int, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    if phase == "probe":
        # Write nothing meaningful; slave will ignore writes since MKW_PROBE_INPUT=1.
        return np.zeros((num_envs, NUM_KARTS), dtype=np.int64)
    if phase == "constant":
        return np.full((num_envs, NUM_KARTS), ACTION_FORWARD_STRAIGHT, dtype=np.int64)
    if phase == "sentinel":
        a = np.zeros((num_envs, NUM_KARTS), dtype=np.int64)
        slot = max(0, min(NUM_KARTS - 1, int(args.slot)))
        a[:, slot] = int(args.sentinel_action)
        return a
    if phase == "all_left":
        # action=0: stickX=-1.0, R=False, Up=False, L=False (accel-only, full left)
        return np.zeros((num_envs, NUM_KARTS), dtype=np.int64)
    if phase == "all_right":
        # action=32: stickX=+1.0, R=False, Up=False, L=False (accel-only, full right)
        return np.full((num_envs, NUM_KARTS), 32, dtype=np.int64)
    # random
    return rng.integers(0, 40, size=(num_envs, NUM_KARTS), dtype=np.int64)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(0)

    instance_info = Path(__file__).resolve().parent / "instance_info"
    instance_info.mkdir(exist_ok=True)
    out_csv = instance_info / f"run_random_12p_{args.phase}.csv"
    print(f"[runner] phase={args.phase} steps={args.steps} num_envs={args.num_envs} → {out_csv}")
    if args.num_envs > 4:
        raise SystemExit("[runner] num_envs > 4 is forbidden on this host (compose.yml/HEADLESS cap).")

    env = DolphinEnv(
        num_envs=args.num_envs,
        reset_mode=args.reset_mode,
        reset_savestate=args.reset_savestate,
        episode_timeout_steps=args.episode_timeout_steps,
    )
    obs, _ = env.reset()
    print(f"[runner] env up. obs.shape={obs.shape}")

    # CSV header: one row per (step, env, kart) so all 12 trajectories per env are captured.
    header = ["step", "env", "kart", "RaceCompletion", "kart_x", "kart_z", "race_pos",
              "reward_env", "done_env", "trun_env", "race_stage_env"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        t0 = time.time()
        for step in range(args.steps):
            actions = actions_for_phase(args.phase, args.num_envs, args, rng)
            env.step_async(actions)
            obs, rewards, dones, truns, infos = env.step_wait()

            rc_all = infos.get("RaceCompletion_all", np.zeros((args.num_envs, NUM_KARTS), dtype=np.float32))
            kx_all = infos.get("kart_x_all",        np.zeros((args.num_envs, NUM_KARTS), dtype=np.float32))
            kz_all = infos.get("kart_z_all",        np.zeros((args.num_envs, NUM_KARTS), dtype=np.float32))
            rp_all = infos.get("race_pos_all",      np.zeros((args.num_envs, NUM_KARTS), dtype=np.int32))
            stage  = infos.get("race_stage",        np.zeros(args.num_envs,             dtype=np.int32))

            for i in range(args.num_envs):
                for k in range(NUM_KARTS):
                    writer.writerow([
                        step, i, k,
                        float(rc_all[i, k]),
                        float(kx_all[i, k]),
                        float(kz_all[i, k]),
                        int(rp_all[i, k]),
                        float(rewards[i]),
                        bool(dones[i]),
                        bool(truns[i]),
                        int(stage[i]),
                    ])

            if (step % max(1, args.log_every)) == 0:
                elapsed = time.time() - t0
                sps = (step + 1) / max(elapsed, 1e-6)
                rc_env0 = rc_all[0].tolist() if args.num_envs >= 1 else []
                print(
                    f"[runner] step={step:4d} sps={sps:5.1f} "
                    f"rc[env0,kart0..11]={['%.3f'%v for v in rc_env0]} "
                    f"r={rewards.tolist()} d={dones.astype(int).tolist()} t={truns.astype(int).tolist()}"
                )
                f.flush()
                os.fsync(f.fileno())

    print(f"[runner] done. wrote {out_csv}")


if __name__ == "__main__":
    main()
