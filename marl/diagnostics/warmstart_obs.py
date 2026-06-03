"""
Warm-start a new opponent-observability condition from the converged COARSE policy.

The opponent-observability ablation (none / coarse / full) changes the actor's
input dimension, so each condition needs its own network and cannot resume a
coarse checkpoint directly. But almost everything transfers: the centralized
critic is identical across conditions, and the actor's hidden+output layers and
its self/teammate/team-indicator input columns are laid out identically. Only the
opponent columns of the actor's first layer differ.

This tool loads the coarse checkpoint and writes a full MAPPO checkpoint for the
target condition that:
  * copies the critic verbatim (same 102-d global state in every condition),
  * copies the actor hidden + output layers verbatim,
  * copies the actor first-layer SELF (cols 0--49) and TEAM-indicator columns,
  * freshly initializes only the opponent columns (none: dropped; full: new),
  * re-initializes both optimizers (Adam momentum re-warms quickly).
So the new condition starts as a driving-competent, value-aware policy and only
has to adapt its opponent pathway, instead of re-running the race-first curriculum.

  PYTHONPATH=/cs377 python marl/diagnostics/warmstart_obs.py \
      --config marl/configs/mkw_2v2_mappo.yaml \
      --coarse-checkpoint <coarse_final.pt> --opponent-obs none --out <warm.pt>

Obs layout (build_obs order, see marl/obs.py): self[0:25], mate[25:50],
opponents[50:...], team-indicator[last]. So self+mate = cols 0..49 in EVERY mode.
"""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from marl.obs import AsymmetricTeamObs, CentralizedTeamState, FULL_DIM
from marl.action import MKWTeamAction
from marl.algo.mappo import SharedActor, MAPPOLearner

SELF_MATE_COLS = 2 * FULL_DIM   # 50: self + teammate blocks, identical in all modes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/cs377/marl/configs/mkw_2v2_mappo.yaml")
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--opponent-obs", required=True, choices=["none", "full"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    hidden = OmegaConf.to_container(cfg.model.policy_kwargs.layer_sizes)
    critic_hidden = OmegaConf.to_container(cfg.model.critic_kwargs.layer_sizes)
    n_actions = MKWTeamAction().get_action_space().n

    coarse_obs = AsymmetricTeamObs(opponent_obs="coarse")
    new_obs = AsymmetricTeamObs(opponent_obs=args.opponent_obs)
    state_size = CentralizedTeamState().get_obs_size()
    coarse_team_col = coarse_obs.get_obs_size() - 1   # last col = team indicator
    new_team_col = new_obs.get_obs_size() - 1

    ckpt = torch.load(args.coarse_checkpoint, weights_only=True, map_location="cpu")
    src_actor = ckpt["actor"]
    assert src_actor["0.weight"].shape[1] == coarse_obs.get_obs_size(), (
        "coarse checkpoint obs_size mismatch", src_actor["0.weight"].shape)

    # build the target-condition learner (fresh init) then overwrite from coarse
    actor = SharedActor(new_obs.get_obs_size(), n_actions, hidden=hidden)
    learner = MAPPOLearner(actor, state_size=state_size, critic_hidden=critic_hidden)

    new_actor = actor.net.state_dict()
    for k, v in src_actor.items():
        if k == "0.weight":
            continue                       # input layer: column-remapped below
        assert new_actor[k].shape == v.shape, (k, new_actor[k].shape, v.shape)
        new_actor[k] = v.clone()           # 0.bias + all hidden/output layers verbatim

    # input-layer first row of weights: keep fresh opponent columns, copy the rest
    W_src = src_actor["0.weight"]           # [H, coarse_obs]
    W_new = new_actor["0.weight"].clone()   # [H, new_obs] (fresh init)
    W_new[:, :SELF_MATE_COLS] = W_src[:, :SELF_MATE_COLS]      # self + teammate
    W_new[:, new_team_col] = W_src[:, coarse_team_col]         # team indicator
    new_actor["0.weight"] = W_new
    actor.net.load_state_dict(new_actor)

    # critic is identical across conditions -> copy verbatim
    learner.critic.load_state_dict(ckpt["critic"])

    learner.save_checkpoint(args.out)
    n_opp_cols = W_new.shape[1] - SELF_MATE_COLS - 1
    print(f"[warmstart] {args.opponent_obs}: actor in {coarse_obs.get_obs_size()}"
          f"->{new_obs.get_obs_size()}  copied self+mate(50)+team(1), "
          f"fresh opponent cols={n_opp_cols}; critic+hidden copied; optims reset.")
    print(f"[warmstart] wrote {args.out}")


if __name__ == "__main__":
    main()
