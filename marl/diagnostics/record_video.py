"""
Record an MP4 of the current policy racing, from the start until all four karts
have finished a lap (each kart's lap counter advances by >=1), then stop.

Renders the game (must run WITHOUT KART_NULL_RENDER), drives all four karts with
the loaded policy, captures the splitscreen each policy step via PIL ImageGrab on
display :0, and encodes to MP4 incrementally with imageio (bundled ffmpeg).

  PYTHONPATH=/cs377 python /cs377/marl/diagnostics/record_video.py \
      --checkpoint <ckpt.pt> --out /cs377/marl/results/videos/race.mp4

Pauses training while it runs (uses the emulator + GPU). Resume training after.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import imageio
from PIL import ImageGrab
from omegaconf import OmegaConf

from kart_env import KartEnvironment
from kart_env.rl import KartGameState

from marl.obs import AsymmetricTeamObs
from marl.action import MKWTeamAction
from marl.algo.mappo import SharedActor
from marl.train_marl import build_env_options, reset_with_retry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/cs377/marl/configs/mkw_2v2_mappo.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=12000, help="game-frame safety cap")
    ap.add_argument("--lap-delta", type=float, default=1.0,
                    help="race-completion gain that counts as one finished lap (~1.0)")
    ap.add_argument("--display", default=":0")
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampled actions")
    args = ap.parse_args()

    assert os.environ.get("KART_NULL_RENDER") != "1", "must RENDER: do not set KART_NULL_RENDER=1"

    cfg = OmegaConf.load(args.config)
    obs_b = AsymmetricTeamObs(num_agents=cfg.env_setting.num_agents,
                              opponent_obs=cfg.model.get("opponent_obs", "coarse"))
    parser = MKWTeamAction()
    repeats = cfg.env_setting.action_repeats
    sc = cfg.get("start_states", None)
    start_states = (list(sc.files) + [None] * int(sc.get("n_start_line", 1))) if sc else None

    actor = SharedActor(obs_b.get_obs_size(), parser.get_action_space().n,
                        hidden=OmegaConf.to_container(cfg.model.policy_kwargs.layer_sizes))
    actor.load(args.checkpoint)

    env = None
    for attempt in range(5):
        try:
            env = KartEnvironment(env_id=cfg.env_setting.env_id,
                                  options=build_env_options(cfg.env_setting))
            state, agents = reset_with_retry(env, parser, None, repeats, start_states=start_states)
            break
        except RuntimeError as e:
            print(f"[rec] boot {attempt+1}/5 failed: {e}; recreating", flush=True)
            try:
                env.close()
            except Exception:
                pass
            env = None
    if env is None:
        raise SystemExit("env never stabilized")
    obs_b.reset(agents, state)

    def act(obs):
        if not args.greedy:
            return actor.get_actions(obs)
        x = torch.as_tensor(np.stack([obs[a] for a in agents]), dtype=torch.float32)
        with torch.no_grad():
            idx = actor.net(x).argmax(-1).tolist()
        return {a: int(idx[i]) for i, a in enumerate(agents)}

    # the savestate starts ON the start/finish line, so current_lap ticks up the
    # instant they cross it (= "started" lap 1, not finished one). Use a completion
    # GAIN of ~1 lap instead, which robustly means "drove a full lap".
    start_comp = {a: float(state.players[a].max_race_completion) for a in agents}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)

    gframes, captured, done_reason = 0, 0, "max-frames cap"
    while gframes < args.max_frames:
        obs = {a: obs_b.build_obs(a, state) for a in agents}
        kart = {a: parser.parse_action(act(obs)[a]) for a in agents}
        try:
            for _ in range(repeats):
                od, *_ = env.step(kart)
            state = KartGameState.from_obs_dict(od)
        except ValueError:
            continue
        gframes += repeats
        try:
            writer.append_data(np.asarray(ImageGrab.grab(xdisplay=args.display))[:, :, :3])
            captured += 1
        except Exception as e:
            if captured == 0:
                print(f"[rec] frame grab failed: {e!r}", flush=True)
        if all(float(state.players[a].max_race_completion) - start_comp[a] >= args.lap_delta
               for a in agents):
            done_reason = "all 4 karts finished a lap"
            break

    writer.close()
    env.close()
    print(f"[rec] DONE ({done_reason}): {captured} frames @ {args.fps}fps -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
