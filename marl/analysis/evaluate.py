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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/cs377/marl/configs/mkw_2v2_mappo.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    obs_b = AsymmetricTeamObs(num_agents=cfg.env_setting.num_agents)
    action_parser = MKWTeamAction()
    repeats = cfg.env_setting.action_repeats

    actor = SharedActor(obs_b.get_obs_size(), action_parser.get_action_space().n,
                        hidden=OmegaConf.to_container(cfg.model.policy_kwargs.layer_sizes))
    actor.load(args.checkpoint)   # accepts actor-only and full learner checkpoints

    env = KartEnvironment(env_id=cfg.env_setting.env_id,
                          options=build_env_options(cfg.env_setting))
    az = RolloutAnalyzer(item_use_actions=ITEM_USE_ACTIONS)

    for ep in range(args.episodes):
        state, agents = reset_with_retry(env, action_parser, None, repeats)
        obs_b.reset(agents, state)
        steps = 0
        while steps < 20_000:
            obs = {a: obs_b.build_obs(a, state) for a in agents}
            actions = greedy_actions(actor, obs)
            az.record(state, actions)
            kart_actions = {a: action_parser.parse_action(actions[a]) for a in agents}
            try:
                for _ in range(repeats):
                    obs_dict, _, term, trunc, _ = env.step(kart_actions)
                state = KartGameState.from_obs_dict(obs_dict)
            except ValueError:
                continue
            steps += repeats
            if any(term.values()) or any(trunc.values()):
                break
        print(f"[eval] episode {ep+1}/{args.episodes} done ({steps} steps)", flush=True)

    summary = az.summary()
    out_dir = Path("/cs377/marl/results/analysis") / cfg.wandb.wandb_run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
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

    # RQ1 plot: item-holding time vs rank
    hold = summary["item_holding_by_rank"]
    if hold and HAVE_MPL:
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
    if rate and HAVE_MPL:
        ranks = sorted(rate)
        plt.figure(figsize=(5, 3.2))
        plt.bar([str(r) for r in ranks], [rate[r] for r in ranks], color="darkorange")
        plt.xlabel("rank (1 = leader)")
        plt.ylabel("item-use rate while holding")
        plt.title("RQ1: rank-conditioned item-use rate")
        plt.tight_layout()
        plt.savefig(out_dir / "item_use_rate_by_rank.png", dpi=150)
        plt.close()

    print(f"[eval] wrote {out_dir}/summary.csv + plots")
    print(f"[eval] RQ1 differentiation (best-worst hold) = "
          f"{summary['holding_differentiation_best_minus_worst']}")
    print(f"[eval] RQ2 opp:teamkill ratio = {summary['opp_to_teamkill_ratio']:.2f} "
          f"(opp={summary['opponent_hits']}, team={summary['teamkills']})")

    env.close()


if __name__ == "__main__":
    main()
