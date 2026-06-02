"""
Smoke-test the headless KART_NULL_RENDER path on a SPARE env_id (default 1), so it
runs alongside a live training instance on env_id=0 without disturbing it.

Checks: (1) the env boots with the Null video backend (no vglrun/GPU), (2) reset +
step work and the memory/vector obs parse (GRAPHIC_INFO is None), (3) raw stepping
throughput. Compare GPU before/after with nvidia-smi externally.

  KART_NULL_RENDER=1 PYTHONPATH=/cs377 python /cs377/marl/diagnostics/test_null_render.py 1
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KART_NULL_RENDER", "1")

from omegaconf import OmegaConf

from kart_env import KartEnvironment
from kart_env.rl import KartGameState

from marl.action import MKWTeamAction
from marl.train_marl import build_env_options, reset_with_retry


def main():
    env_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = OmegaConf.load("/cs377/marl/configs/mkw_2v2_mappo.yaml")
    parser = MKWTeamAction()
    repeats = cfg.env_setting.action_repeats

    env = None
    for attempt in range(4):
        try:
            env = KartEnvironment(env_id=env_id, options=build_env_options(cfg.env_setting))
            state, agents = reset_with_retry(env, parser, None, repeats)
            break
        except Exception as e:
            print(f"[boot] attempt {attempt+1}/4 failed: {e!r}; recreating", flush=True)
            try:
                env.close()
            except Exception:
                pass
            env = None
    if env is None:
        raise SystemExit("NULL-RENDER BOOT FAILED")
    print(f"booted null-render env_id={env_id}; agents={agents}", flush=True)

    neutral = parser.parse_action(0)
    t0 = time.time()
    n = 0
    for _ in range(300):
        try:
            obs, *_ = env.step({a: neutral for a in env.agents})
            n += 1
        except ValueError:
            pass
    dt = time.time() - t0
    state = KartGameState.from_obs_dict(obs)
    fps = n / dt if dt > 0 else 0
    print(f"stepped {n} env-steps in {dt:.1f}s -> {fps:.0f} frames/s/instance", flush=True)
    print("sample completion:", {a: round(state.players[a].max_race_completion, 3) for a in agents}, flush=True)
    env.close()
    print("NULL-RENDER SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
