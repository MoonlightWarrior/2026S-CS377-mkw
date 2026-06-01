"""
Phase 2 — MAPPO 2v2 training entrypoint (CTDE).

Wires our contributions together:
  - actor obs  : AsymmetricTeamObs   (restricted, 59-d)        [marl/obs.py]
  - critic obs : CentralizedTeamState (global, 102-d)          [marl/obs.py]
  - actions    : MKWTeamAction        (item on L)              [marl/action.py]
  - learner    : MAPPO centralized critic                      [marl/algo/mappo.py]
  - env        : Vlab KartEnvironment (4-player MKW)

Status: runnable plumbing. The reward here is an INTERIM progress+rank stub —
Phase 3 replaces it with the proposal's rank-based team reward. Self-play
(Phase 4) is not wired yet: all four karts use the current shared policy.

Clean shutdown: SIGTERM/SIGINT (e.g. `docker stop`) breaks the loop, saves a
final checkpoint, and calls wandb.finish() — so stopped runs show as finished,
not "crashed".

Run:
  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/train_marl.py \
      --config /cs377/marl/configs/mkw_2v2_mappo.yaml
"""
from __future__ import annotations

import argparse
import random
import signal
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from kart_env import KartEnvironment, OptionType
from kart_env.utils.macro_helper import (
    CCChoice, CharacterChoice, CupChoice, CourseChoice, DriftModeChoice,
    VehicleChoice, coerce_choice,
)
from kart_env.rl import KartGameState

from marl.obs import AsymmetricTeamObs, CentralizedTeamState
from marl.action import MKWTeamAction
from marl.algo.mappo import SharedActor, MAPPOLearner
from marl.rewards import build_reward
from marl.selfplay import SnapshotPool

EMPTY_ITEM = 20  # MKW "no item" sentinel (Phase 0 finding)

# ── clean-shutdown flag ──────────────────────────────────────────────────────
_STOP = {"flag": False}


def _request_stop(signum, frame):
    print(f"\n[shutdown] signal {signum} received — finishing current episode then saving.", flush=True)
    _STOP["flag"] = True


def build_env_options(env_cfg) -> OptionType:
    return OptionType(
        num_agents=env_cfg.num_agents,
        character=[coerce_choice(c, CharacterChoice) for c in env_cfg.character],
        vehicle=[coerce_choice(v, VehicleChoice) for v in env_cfg.vehicle],
        drift_modes=[coerce_choice(d, DriftModeChoice) for d in env_cfg.drift_modes],
        cup=coerce_choice(env_cfg.cup, CupChoice),
        course=coerce_choice(env_cfg.course, CourseChoice),
        cc=coerce_choice(env_cfg.cc, CCChoice),
        disable_cpu=bool(env_cfg.get("disable_cpu", False)),  # CPUs off -> clean 4-kart field
    )


def reset_with_retry(env, action_parser, agents_hint, repeats, tries=5, start_states=None):
    """
    The memory reader is flaky during race load; retry reset + neutral warmup.
    If `start_states` is given (list whose entries are None=start line, or a save
    file path), one is sampled per reset for diverse-start (curriculum) training.
    """
    neutral = action_parser.parse_action(0)
    choice = random.choice(start_states) if start_states else None
    options = {"file": choice} if choice else {}
    for attempt in range(tries):
        try:
            obs_dict, _ = env.reset(options=dict(options))
            agents = list(env.agents)
            for _ in range(10):
                for _ in range(repeats):
                    obs_dict, *_ = env.step({a: neutral for a in agents})
            return KartGameState.from_obs_dict(obs_dict), agents
        except ValueError:
            print(f"[warmup] memory not ready ({attempt+1}/{tries}), retrying...", flush=True)
            time.sleep(2.0)
    raise RuntimeError("env memory never stabilized after reset")


def collect_episode(env, learner, opponent, learner_team, actor_obs_b, critic_b,
                    action_parser, reward_fn, repeats, max_steps=20_000,
                    stall_patience=150, start_states=None):
    """
    Self-play rollout. `learner` (live shared policy) drives `learner_team`;
    `opponent` (frozen snapshot, or a frozen copy of the learner early on) drives
    the rest. Only the learner team's transitions are returned for training.

    Truncates the episode after `stall_patience` policy steps with no new track
    progress (max_race_completion not advancing). The agents drive off-road early
    and get stuck; without this they sit there for the full 20k frames, flooding
    the buffer with useless "stuck" data. Truncating ends the attempt fast so the
    env resets to the start line, giving many more attempts at the opening + the
    first turn per unit time.
    """
    state, agents = reset_with_retry(env, action_parser, None, repeats,
                                     start_states=start_states)
    learner_team = [a for a in learner_team if a in agents]
    opp_team = [a for a in agents if a not in learner_team]

    actor_obs_b.reset(agents, state)
    critic_b.reset(agents, state)
    reward_fn.reset(agents, state)

    traj = {a: [] for a in learner_team}
    cum_reward = {a: 0.0 for a in learner_team}
    speed_acc = {a: 0.0 for a in learner_team}
    steps = 0

    start_completion = float(np.mean([state.players[a].max_race_completion for a in learner_team]))
    best_progress = max(state.players[a].max_race_completion for a in learner_team)
    stall_steps = 0
    stalled = False

    flat = {a: actor_obs_b.build_obs(a, state) for a in agents}
    gstate = {a: critic_b.build_obs(a, state) for a in agents}

    while steps < max_steps:
        # decentralized execution: each side acts on its own restricted views
        actions = learner.get_actions({a: flat[a] for a in learner_team})
        if opp_team:
            actions.update(opponent.get_actions({a: flat[a] for a in opp_team}))
        kart_actions = {a: action_parser.parse_action(actions[a]) for a in agents}
        try:
            for _ in range(repeats):
                obs_dict, _, terminations, truncations, _ = env.step(kart_actions)
            state = KartGameState.from_obs_dict(obs_dict)
        except ValueError:
            continue  # transient read at a load boundary

        rewards = reward_fn.get_rewards(agents, state)   # needs all agents (team term)
        next_flat = {a: actor_obs_b.build_obs(a, state) for a in agents}
        next_g = {a: critic_b.build_obs(a, state) for a in agents}

        # stall detection: max_race_completion is monotone; if the team's furthest
        # progress hasn't advanced for `stall_patience` steps, they're stuck.
        cur_progress = max(state.players[a].max_race_completion for a in learner_team)
        if cur_progress > best_progress + 1e-4:
            best_progress = cur_progress
            stall_steps = 0
        else:
            stall_steps += 1
        stalled = stall_steps >= stall_patience

        done = any(terminations.values()) or any(truncations.values()) or stalled

        for a in learner_team:                            # train ONLY learner team
            traj[a].append({
                "actor_obs":         flat[a],
                "global_state":      gstate[a],
                "next_global_state": next_g[a],
                "action":            actions[a],
                "reward":            rewards[a],
                "done":              done,
            })
            cum_reward[a] += rewards[a]
            speed_acc[a] += state.players[a].speed

        flat, gstate = next_flat, next_g
        steps += repeats
        if done or _STOP["flag"]:   # responsive to SIGTERM mid-episode (docker stop)
            break

    denom = max(steps // repeats, 1)
    # team win = lower rank-sum than the opponent team (lower position = better)
    learner_ranks = sum(state.players[a].race_position for a in learner_team)
    opp_ranks = sum(state.players[a].race_position for a in opp_team) if opp_team else 1e9
    end_completion = float(np.mean([state.players[a].max_race_completion for a in learner_team]))
    stats = {
        "episode_steps": steps,
        "stalled": float(stalled),
        "start_completion": start_completion,                 # ~0 = start line; >1 = mid-track curriculum
        "progress": end_completion - start_completion,        # distance driven THIS episode (start-agnostic)
        "mean_episode_reward": float(np.mean(list(cum_reward.values()))),
        "mean_race_completion": end_completion,
        "finish_rate": float(np.mean([float(state.players[a].is_finished) for a in learner_team])),
        "mean_speed": float(np.mean([speed_acc[a] / denom for a in learner_team])),
        "best_race_position": float(min(state.players[a].race_position for a in learner_team)),
        "win_vs_snapshot": float(learner_ranks < opp_ranks),
    }
    return traj, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/cs377/marl/configs/mkw_2v2_mappo.yaml")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cfg = OmegaConf.load(args.config)
    hp = cfg.hyperparameter

    actor_obs_b = AsymmetricTeamObs(num_agents=cfg.env_setting.num_agents)
    critic_b = CentralizedTeamState(num_agents=cfg.env_setting.num_agents)
    action_parser = MKWTeamAction(
        disable_item_use=bool(cfg.env_setting.get("disable_item_use", False)))
    reward_fn = build_reward(cfg.reward, gamma=hp.gae_gamma)
    obs_size = actor_obs_b.get_obs_size()
    state_size = critic_b.get_obs_size()
    n_actions = action_parser.get_action_space().n
    print(f"obs_size={obs_size}  state_size={state_size}  n_actions={n_actions}", flush=True)

    actor = SharedActor(obs_size, n_actions,
                        hidden=OmegaConf.to_container(cfg.model.policy_kwargs.layer_sizes))
    learner = MAPPOLearner(
        actor, state_size,
        critic_hidden=OmegaConf.to_container(cfg.model.critic_kwargs.layer_sizes),
        actor_lr=hp.policy_lr, critic_lr=hp.critic_lr, gamma=hp.gae_gamma, lam=hp.gae_lambda,
        clip_eps=hp.clip_eps, ent_coef=hp.ppo_ent_coef, vf_coef=hp.vf_coef,
        ppo_epochs=hp.ppo_epochs, mini_batch_size=hp.ppo_minibatch_size,
        max_grad_norm=hp.max_grad_norm,
    )
    if args.checkpoint:
        learner.load_checkpoint(args.checkpoint)
        print(f"[checkpoint] resumed from {args.checkpoint}", flush=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb.wandb_project, name=cfg.wandb.wandb_run_name,
                   entity=cfg.wandb.wandb_entity, config=OmegaConf.to_container(cfg, resolve=True))

    def make_env():
        return KartEnvironment(env_id=cfg.env_setting.env_id,
                               options=build_env_options(cfg.env_setting))

    env = make_env()

    # ── self-play opponent pool (Phase 4) ────────────────────────────────────
    sp = cfg.selfplay
    pool = SnapshotPool(capacity=sp.pool_size, strategy=sp.opponent_sample)
    learner_team = list(sp.learner_team)
    pool.push(actor)                 # seed with the initial policy
    last_snapshot = 0

    save_dir = Path("/cs377/marl/results/checkpoints") / cfg.wandb.wandb_run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # diverse-start curriculum: a list whose entries are None (=start line) or a
    # save-state path. None entries keep some full-race starts so the agent doesn't
    # forget the opening. Empty/absent -> always start at the line (old behavior).
    sc = cfg.get("start_states", None)
    start_states = (list(sc.files) + [None] * int(sc.get("n_start_line", 1))) if sc else None
    if start_states:
        print(f"diverse starts: {len(start_states)} options "
              f"({sum(x is None for x in start_states)} at start line, "
              f"{sum(x is not None for x in start_states)} mid-track)", flush=True)

    total_ts = 0
    iteration = 0
    last_save = 0
    try:
        while total_ts < hp.timestep_limit and not _STOP["flag"]:
            opponent = pool.sample() or actor.clone_frozen()
            try:
                traj, stats = collect_episode(env, actor, opponent, learner_team,
                                              actor_obs_b, critic_b, action_parser, reward_fn,
                                              cfg.env_setting.action_repeats,
                                              start_states=start_states)
            except RuntimeError as e:
                # a bad emulator boot (~1/3) leaves the race unloaded; rebuild it
                print(f"[env] {e}; recreating Dolphin and retrying...", flush=True)
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env()
                continue
            metrics = learner.update(traj, learner_team)
            total_ts += stats["episode_steps"]
            iteration += 1

            print(f"[iter {iteration:5d}] ts={total_ts:10,d}  "
                  f"eplen={stats['episode_steps']:5d}{'T' if stats['stalled'] else ' '}  "
                  f"rew={stats['mean_episode_reward']:+.3f}  "
                  f"start={stats['start_completion']:.2f} completion={stats['mean_race_completion']:.3f} "
                  f"prog={stats['progress']:+.2f}  "
                  f"win={stats['win_vs_snapshot']:.0f}  pool={len(pool)}  "
                  f"speed={stats['mean_speed']:.1f}", flush=True)

            if use_wandb:
                import wandb
                wandb.log({**{f"rollout/{k}": v for k, v in stats.items()},
                           **metrics, "selfplay/pool_size": len(pool),
                           "train/timesteps": total_ts,
                           "train/iteration": iteration}, step=total_ts)

            # grow the opponent pool with the improving policy
            if total_ts - last_snapshot >= sp.snapshot_every_ts:
                pool.push(actor)
                last_snapshot = total_ts
                print(f"[selfplay] pushed snapshot (pool={len(pool)})", flush=True)

            if total_ts - last_save >= hp.save_every_ts:
                learner.save_checkpoint(save_dir / f"checkpoint_{total_ts}.pt")
                print(f"[checkpoint] saved checkpoint_{total_ts}.pt", flush=True)
                last_save = total_ts
    finally:
        # always save + close cleanly so stops are not flagged as crashes
        learner.save_checkpoint(save_dir / f"checkpoint_{total_ts}_final.pt")
        print(f"[shutdown] final checkpoint at ts={total_ts}", flush=True)
        if use_wandb:
            import wandb
            wandb.finish()
        env.close()


if __name__ == "__main__":
    main()
