"""Inspect the per-kart observation slice produced by DolphinEnv for a 12p run.

Builds a short rollout, then prints — for each kart — the 78-dim slice in the obs
vector. Used to confirm whether all 12 slots actually carry live observation data
or whether some are stale/zero (the case when MKW_PLAY_NUM < 12).

Run via the headless docker stack:
    HEADLESS=1 NUM_ENVS=1 docker compose run --rm wii-rl bash -c '
        cd /src/Vlab-WiiRL
        uv sync --quiet
        timeout 90 uv run python -u inspect_obs.py
    '
"""

from __future__ import annotations

import numpy as np

from DolphinEnv import DolphinEnv

NUM_KARTS = 12
RACE_INFO_DIMS = 5
PER_KART_DIMS = 78


def main() -> None:
    env = DolphinEnv(num_envs=1, reset_mode="savestate", episode_timeout_steps=700)
    obs, _ = env.reset()
    # Run a few steps so the framestack settles and karts are post-countdown.
    actions = np.full((1, NUM_KARTS), 16, dtype=np.int64)  # accel, stickX=0
    for _ in range(40):
        env.step_async(actions)
        obs, _, _, _, _ = env.step_wait()

    # obs.shape = (num_envs, framestack, obs_shape) → grab the latest frame, env 0.
    last = np.asarray(obs[0, -1])
    print(f"obs vector shape: {last.shape}, sum: {last.sum():.4f}, "
          f"nonzero count: {int(np.count_nonzero(last))}, "
          f"min: {last.min():.4f}, max: {last.max():.4f}")

    print("race_info (first 5):", last[:RACE_INFO_DIMS].tolist())
    for k in range(NUM_KARTS):
        s = RACE_INFO_DIMS + k * PER_KART_DIMS
        e = s + PER_KART_DIMS
        slice_ = last[s:e]
        nonzero = int(np.count_nonzero(slice_))
        s_min, s_max = float(slice_.min()), float(slice_.max())
        # First 5 fields per kart are id/character/character/etc — show first 8 floats:
        head = [round(float(v), 3) for v in slice_[:8]]
        print(f"kart {k:2d}: nonzero={nonzero:2d}/78  range=[{s_min:.3f}, {s_max:.3f}]  head={head}")

    env.close()


if __name__ == "__main__":
    main()
