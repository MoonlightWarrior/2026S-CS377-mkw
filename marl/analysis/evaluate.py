"""
Phase 5 — Evaluation harness: load a trained policy, run greedy eval episodes,
and emit the RQ metrics as a CSV plus plots for the report.

All four karts are driven by the loaded (greedy) shared policy so we observe its
behavior across the full range of ranks. Outputs go to
results/analysis/<run_name>/ : summary.csv, item_holding_by_rank.png,
item_use_rate_by_rank.png.

Run:
  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/analysis/evaluate.py \
      --config /cs377/marl/configs/mkw_2v2_mappo.yaml \
      --checkpoint /cs377/marl/results/checkpoints/<run>/checkpoint_XXXX.pt \
      --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:                      # CSV is the deliverable; plots are optional
    HAVE_MPL = False

from kart_env import KartEnvironment
from kart_env.rl import KartGameState

from marl.obs import AsymmetricTeamObs
from marl.action import MKWTeamAction, ITEM_USE_ACTIONS
from marl.algo.mappo import SharedActor
from marl.analysis import RolloutAnalyzer
# reuse env-option + reset-retry helpers from training
from marl.train_marl import build_env_options, reset_with_retry


def greedy_actions(actor: SharedActor, obs_dict) -> dict:
    agents = list(obs_dict.keys())
    x = torch.as_tensor(np.stack([obs_dict[a] for a in agents]), dtype=torch.float32)
    with torch.no_grad():
        idx = actor.net(x).argmax(dim=-1).tolist()
    return {a: int(idx[i]) for i, a in enumerate(agents)}


def clean_display() -> None:
    """OS-level display reset between env recreations.

    Dolphin periodically wedges after a couple of episodes; once wedged, a
    Python-level env.close()+recreate is NOT enough — the stale Xvfb/x11vnc/
    websockify keep their ports (":6080 Address already in use") so every fresh
    Dolphin boots into a broken display and wedges again. Killing those and
    removing the X locks (what the training monitor does externally) lets the
    next env boot cleanly. Kills by pattern only, so it never hits this process.
    """
    for cmd in (
        "for p in $(pgrep -f '[d]olphin-emu'); do kill -9 $p 2>/dev/null; done",
        "for p in $(pgrep -f '[X]vfb|[x]11vnc|[w]ebsockify|[n]ovnc'); do kill -9 $p 2>/dev/null; done",
        "rm -f /tmp/.X0-lock /tmp/.X11-unix/X0",
    ):
        subprocess.run(["bash", "-lc", cmd], check=False)
    time.sleep(2)


def write_outputs(az, ep_completion, cfg, args) -> dict:
    """Compute the RQ summary from whatever episodes have completed and write
    summary.csv + plots. Called after EVERY completed episode so partial
    results survive even if a later episode wedges the run out."""
    comps = [c for c, _ in ep_completion]
    fins = [f for _, f in ep_completion]
    finish_rate = float(np.mean(fins)) if fins else 0.0

    summary = az.summary()
    summary["completion_mean"] = float(np.mean(comps)) if comps else 0.0
    summary["completion_max"] = float(np.max(comps)) if comps else 0.0
    summary["finish_rate"] = finish_rate
    summary["n_episodes"] = len(ep_completion)
    out_dir = Path("/cs377/marl/results/analysis") / (
        cfg.wandb.wandb_run_name + ("-stochastic" if args.stochastic else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            if not isinstance(v, dict):
                w.writerow([k, v])
        w.writerow([])
        w.writerow(["rank", "mean_item_holding_frames", "n_uses", "item_use_rate"])
        hold = summary["item_holding_by_rank"]
        rate = summary["item_use_rate_by_rank"]
        for r in sorted(set(hold) | set(rate)):
            w.writerow([r, hold.get(r, {}).get("mean_hold", ""),
                        hold.get(r, {}).get("n", ""), rate.get(r, "")])

    if HAVE_MPL:
        hold = summary["item_holding_by_rank"]
        if hold:
            ranks = sorted(hold)
            plt.figure(figsize=(5, 3.2))
            plt.bar([str(r) for r in ranks], [hold[r]["mean_hold"] for r in ranks])
            plt.xlabel("rank at item use (1 = leader)")
            plt.ylabel("mean item-holding frames")
            plt.title("RQ1: rank-conditioned item holding")
            plt.tight_layout()
            plt.savefig(out_dir / "item_holding_by_rank.png", dpi=150)
            plt.close()
        rate = summary["item_use_rate_by_rank"]
        if rate:
            ranks = sorted(rate)
            plt.figure(figsize=(5, 3.2))
            plt.bar([str(r) for r in ranks], [rate[r] for r in ranks], color="darkorange")
            plt.xlabel("rank (1 = leader)")
            plt.ylabel("item-use rate while holding")
            plt.title("RQ1: rank-conditioned item-use rate")
            plt.tight_layout()
            plt.savefig(out_dir / "item_use_rate_by_rank.png", dpi=150)
            plt.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/cs377/marl/configs/mkw_2v2_mappo.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions from the policy (true variance) instead of greedy argmax")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    obs_b = AsymmetricTeamObs(num_agents=cfg.env_setting.num_agents,
                              opponent_obs=cfg.model.get("opponent_obs", "coarse"))
    action_parser = MKWTeamAction()   # items ALWAYS enabled for eval (observe item behavior)
    repeats = cfg.env_setting.action_repeats
    # Load the SAME start-state baseline as training (the verified 4xFunky savestate)
    # so the eval runs on identical karts; otherwise reset() falls back to the
    # menu-nav slot 0, whose characters are wrong and would invalidate RQ1.
    sc = cfg.get("start_states", None)
    start_states = (list(sc.files) + [None] * int(sc.get("n_start_line", 1))) if sc else None

    actor = SharedActor(obs_b.get_obs_size(), action_parser.get_action_space().n,
                        hidden=OmegaConf.to_container(cfg.model.policy_kwargs.layer_sizes))
    actor.load(args.checkpoint)   # accepts actor-only and full learner checkpoints

    def make_env():
        return KartEnvironment(env_id=cfg.env_setting.env_id,
                               options=build_env_options(cfg.env_setting))

    env = make_env()
    az = RolloutAnalyzer(item_use_actions=ITEM_USE_ACTIONS)
    ep_completion = []   # per-episode (max completion reached, finished?)

    # A periodic Dolphin wedge makes env.step raise ValueError forever; the old
    # `except ValueError: continue` then busy-loops. Cap consecutive failures so
    # a wedge RAISES, then discard the partial episode (truncate the analyzer's
    # step buffer back to a pre-episode mark so stats aren't contaminated),
    # recreate the env, and retry the same episode.
    ep = 0
    recreations = 0
    while ep < args.episodes:
        mark = len(az._steps)   # rollback point if this episode wedges
        try:
            state, agents = reset_with_retry(env, action_parser, None, repeats,
                                             start_states=start_states)
            obs_b.reset(agents, state)
            steps = 0
            step_fails = 0
            while steps < 20_000:
                obs = {a: obs_b.build_obs(a, state) for a in agents}
                actions = actor.get_actions(obs) if args.stochastic else greedy_actions(actor, obs)
                az.record(state, actions)
                kart_actions = {a: action_parser.parse_action(actions[a]) for a in agents}
                try:
                    for _ in range(repeats):
                        obs_dict, _, term, trunc, _ = env.step(kart_actions)
                    state = KartGameState.from_obs_dict(obs_dict)
                    step_fails = 0
                except ValueError:
                    step_fails += 1
                    if step_fails > 400:
                        raise RuntimeError("env.step failed >400x consecutively; Dolphin wedged")
                    continue
                steps += repeats
                if any(term.values()) or any(trunc.values()):
                    break
        except RuntimeError as e:
            del az._steps[mark:]   # drop this episode's partial records
            recreations += 1
            print(f"[eval] episode {ep+1} wedged ({e}); cleaning display + "
                  f"recreating env [recreation {recreations}]", flush=True)
            if recreations > 30:
                raise SystemExit("too many env recreations; aborting eval")
            try:
                env.close()
            except Exception:
                pass
            clean_display()         # OS-level reset so the wedge actually clears
            env = make_env()
            continue
        comp = max(float(state.players[a].max_race_completion) for a in agents)
        fin = any(bool(getattr(state.players[a], "is_finished", False)) for a in agents)
        ep_completion.append((comp, fin))
        print(f"[eval] episode {ep+1}/{args.episodes} done ({steps} steps, "
              f"max_completion={comp:.2f}, finished={fin})", flush=True)
        # write outputs after EVERY episode so partial results survive a later wedge
        write_outputs(az, ep_completion, cfg, args)
        ep += 1

    summary = write_outputs(az, ep_completion, cfg, args)
    comps = [c for c, _ in ep_completion]
    fins = [f for _, f in ep_completion]
    finish_rate = float(np.mean(fins)) if fins else 0.0
    out_dir = Path("/cs377/marl/results/analysis") / (
        cfg.wandb.wandb_run_name + ("-stochastic" if args.stochastic else ""))
    print(f"[eval] FINISH: {len(comps)} eps | completion mean="
          f"{np.mean(comps):.2f} max={np.max(comps):.2f} "
          f"| finish_rate={finish_rate:.2f} ({sum(fins)}/{len(fins)} eps)", flush=True)
    print(f"[eval] wrote {out_dir}/summary.csv + plots")
    print(f"[eval] RQ1 differentiation (best-worst hold) = "
          f"{summary['holding_differentiation_best_minus_worst']}")
    print(f"[eval] RQ2 opp:teamkill ratio = {summary['opp_to_teamkill_ratio']:.2f} "
          f"(opp={summary['opponent_hits']}, team={summary['teamkills']})")

    env.close()


if __name__ == "__main__":
    main()
