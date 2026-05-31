"""
Offline self-test for the Phase 1 observation builders.

Builds a synthetic 4-agent KartGameState (no Dolphin required) and checks:
  - actor obs size = 25 + 25 + 2*4 + 1 = 59
  - critic state size = 4*25 + 2 = 102
  - asymmetry: opponent ITEM is NOT present in the actor obs but IS in the
    critic state (the defining CTDE property of the proposal).

Run:  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/tests/test_obs_dims.py
"""
from __future__ import annotations

import numpy as np

from kart_env.rl.game_state import KartGameState, PlayerState, RaceInfo
from marl.obs import AsymmetricTeamObs, CentralizedTeamState, ITEM_NORM


def _player(pid: int, *, item: int, rank: int, x: float) -> PlayerState:
    return PlayerState(
        player_id=pid, current_race_completion=1.0 + 0.1 * pid, max_race_completion=1.0,
        current_lap=1, max_lap=1, speed=80.0 + pid, position=(x, 0.0, 0.0),
        velocity=(50.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0),
        main_rotation=(0.0, 0.0, 0.0, 1.0), drift_state=0, miniturbo_charge=0,
        smt_charge=0, race_position=rank, state_bit=0, item=item,
        mt_boost_timer=0, mushroom_boost_timer=0, start_boost_charge=0.0,
    )


def main() -> None:
    # opponents (ids 2,3) carry a very distinctive item value so we can search for it
    OPP_ITEM = 17
    players = {
        0: _player(0, item=3, rank=1, x=100.0),   # learner team
        1: _player(1, item=5, rank=2, x=200.0),   # learner team (teammate of 0)
        2: _player(2, item=OPP_ITEM, rank=3, x=300.0),  # opponent
        3: _player(3, item=OPP_ITEM, rank=4, x=400.0),  # opponent
    }
    state = KartGameState(
        race_info=RaceInfo(frame_count=1234, course_id=0, player_count=12),
        players=players,
    )

    actor = AsymmetricTeamObs(num_agents=4)
    critic = CentralizedTeamState(num_agents=4)

    a0 = actor.build_obs(0, state)
    c0 = critic.build_obs(0, state)

    assert actor.get_obs_size() == 59, actor.get_obs_size()
    assert critic.get_obs_size() == 102, critic.get_obs_size()
    assert a0.shape == (59,), a0.shape
    assert c0.shape == (102,), c0.shape

    # Asymmetry check: opponent item (normalized) must NOT leak into the actor obs,
    # but MUST be visible to the centralized critic.
    opp_item_norm = min(OPP_ITEM / ITEM_NORM, 1.0)
    in_actor = np.any(np.isclose(a0, opp_item_norm, atol=1e-6))
    in_critic = np.any(np.isclose(c0, opp_item_norm, atol=1e-6))
    assert not in_actor, "LEAK: opponent item visible to actor — asymmetry broken"
    assert in_critic, "opponent item should be in the centralized critic state"

    # team indicator is the last actor feature
    assert a0[-1] == 0.0, "agent 0 should be team 0"
    a2 = actor.build_obs(2, state)
    assert a2[-1] == 1.0, "agent 2 should be team 1"

    print("PASS  actor=59  critic=102  asymmetry OK (opp item hidden from actor, "
          "present in critic)  team-indicator OK")


if __name__ == "__main__":
    main()
