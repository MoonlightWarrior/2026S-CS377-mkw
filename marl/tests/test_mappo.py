"""
Offline correctness test for the Phase 2 MAPPO learner (no Dolphin needed).

Checks:
  - actor consumes the 59-d restricted obs, critic consumes the 102-d global state
  - get_actions returns valid discrete actions for all agents
  - a full learner.update() runs and returns finite metrics
  - the update actually optimizes BOTH networks (shared actor + centralized critic)
  - clone_frozen() yields a detached, non-trainable opponent copy (Phase 4 hook)

Run:  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/tests/test_mappo.py
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from marl.obs import AsymmetricTeamObs, CentralizedTeamState
from marl.algo.mappo import SharedActor, MAPPOLearner

N_ACTIONS = 21
T = 40
AGENTS = [0, 1, 2, 3]


def _synthetic_trajectory(obs_size, state_size):
    rng = np.random.default_rng(0)
    traj = {a: [] for a in AGENTS}
    for a in AGENTS:
        for t in range(T):
            traj[a].append({
                "actor_obs":         rng.standard_normal(obs_size).astype(np.float32),
                "global_state":      rng.standard_normal(state_size).astype(np.float32),
                "next_global_state": rng.standard_normal(state_size).astype(np.float32),
                "action":            int(rng.integers(0, N_ACTIONS)),
                "reward":            float(rng.standard_normal()),
                "done":              (t == T - 1),
            })
    return traj


def main() -> None:
    torch.manual_seed(0)
    obs_size = AsymmetricTeamObs(4).get_obs_size()       # 59
    state_size = CentralizedTeamState(4).get_obs_size()  # 102
    assert (obs_size, state_size) == (59, 102), (obs_size, state_size)

    actor = SharedActor(obs_size, N_ACTIONS, hidden=[64, 64])
    learner = MAPPOLearner(actor, state_size, critic_hidden=[64, 64],
                           mini_batch_size=32, ppo_epochs=4)

    # critic must reject a restricted obs (wrong width) — proves it is centralized
    try:
        learner.critic(torch.zeros(1, obs_size))
        raise AssertionError("centralized critic accepted a 59-d obs — not centralized")
    except RuntimeError:
        pass  # expected: critic expects 102-d global state

    # get_actions
    obs = {a: np.random.randn(obs_size).astype(np.float32) for a in AGENTS}
    acts = actor.get_actions(obs)
    assert set(acts) == set(AGENTS)
    assert all(0 <= v < N_ACTIONS for v in acts.values()), acts

    # snapshot params to confirm the update optimizes both networks
    a_before = copy.deepcopy(next(actor.net.parameters()).detach().clone())
    c_before = copy.deepcopy(next(learner.critic.parameters()).detach().clone())

    traj = _synthetic_trajectory(obs_size, state_size)
    metrics = learner.update(traj, AGENTS)

    assert all(np.isfinite(v) for v in metrics.values()), metrics
    a_after = next(actor.net.parameters()).detach().clone()
    c_after = next(learner.critic.parameters()).detach().clone()
    assert not torch.allclose(a_before, a_after), "shared actor was not updated"
    assert not torch.allclose(c_before, c_after), "centralized critic was not updated"

    # clone_frozen: detached opponent for self-play
    frozen = actor.clone_frozen()
    assert all(not p.requires_grad for p in frozen.net.parameters())
    x = torch.randn(3, obs_size)
    with torch.no_grad():
        assert torch.allclose(frozen.net(x), actor.net(x)), "frozen clone differs from source"

    print(f"PASS  actor_in={obs_size} critic_in={state_size}  "
          f"metrics={ {k: round(v,4) for k,v in metrics.items()} }  "
          f"both nets updated, critic rejects restricted obs, frozen-clone OK")


if __name__ == "__main__":
    main()
