"""
Capture mid-track Dolphin save states for diverse-start (curriculum) training.

Drives the converged greedy policy from the start line and saves a full game
state each time the team's furthest progress crosses a milestone. Loading these
mid-track states as episode start points (see train_marl.py) gives the agents
concentrated practice at the part of the course where they currently stall,
instead of always restarting at lap 1.

Run (inside the container, with training stopped so the emulator is free):
  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/diagnostics/capture_startstates.py \
      --checkpoint /cs377/marl/results/checkpoints/cs377-2v2-mappo-06-revert/checkpoint_2003556.pt
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from omegaconf import OmegaConf

from kart_env import KartEnvironment
from kart_env.rl import KartGameState

from marl.obs import AsymmetricTeamObs
from marl.action import MKWTeamAction
from marl.algo.mappo import SharedActor
from marl.train_marl import build_env_options, reset_with_retry


def greedy(actor: SharedActor, obs) -> dict:
    agents = list(obs.keys())
    x = torch.as_tensor(np.stack([obs[a] for a in agents]), dtype=torch.float32)
    with torch.no_grad():
        idx = actor.net(x).argmax(-1).tolist()
    return {a: int(idx[i]) for i, a in enumerate(agents)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/cs377/marl/configs/mkw_2v2_mappo.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="/cs377/marl/results/startstates")
    ap.add_argument("--milestones", default="1.10,1.20,1.30",
                    help="completion values at which to save a state")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    obs_b = AsymmetricTeamObs(num_agents=cfg.env_setting.num_agents)
    parser = MKWTeamAction()
    repeats = cfg.env_setting.action_repeats

    actor = SharedActor(obs_b.get_obs_size(), parser.get_action_space().n,
                        hidden=OmegaConf.to_container(cfg.model.policy_kwargs.layer_sizes))
    actor.load(args.checkpoint)
    print(f"loaded {args.checkpoint}", flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = sorted(float(x) for x in args.milestones.split(","))

    # the emulator boot is intermittently flaky; recreate the env until one sticks
    env = None
    for attempt in range(5):
        try:
            env = KartEnvironment(env_id=cfg.env_setting.env_id,
                                  options=build_env_options(cfg.env_setting))
            state, agents = reset_with_retry(env, parser, None, repeats)
            break
        except RuntimeError as e:
            print(f"[capture] env didn't stabilize (attempt {attempt+1}/5): {e}; recreating...", flush=True)
            try:
                env.close()
            except Exception:
                pass
            env = None
    if env is None:
        raise RuntimeError("capture: env never stabilized after 5 attempts")
    obs_b.reset(agents, state)

    saved, best, stall = [], 0.0, 0
    for step in range(5000):
        obs = {a: obs_b.build_obs(a, state) for a in agents}
        acts = greedy(actor, obs)
        kart = {a: parser.parse_action(acts[a]) for a in agents}
        try:
            for _ in range(repeats):
                od, *_ = env.step(kart)
            state = KartGameState.from_obs_dict(od)
        except ValueError:
            continue

        cur = max(state.players[a].max_race_completion for a in agents)
        while todo and cur >= todo[0]:
            m = todo.pop(0)
            path = str(out / f"track_{int(round(m * 100)):03d}.sav")
            env.save_file(path)
            saved.append((m, path))
            print(f"saved {path} at completion {cur:.3f}", flush=True)

        if cur > best + 1e-4:
            best, stall = cur, 0
        else:
            stall += 1
        if stall > 250 or not todo:
            break

    print(f"DONE: captured {len(saved)} states, max_completion={best:.3f}", flush=True)
    for m, p in saved:
        print(f"  {m:.2f} -> {p}")
    env.close()


if __name__ == "__main__":
    main()
