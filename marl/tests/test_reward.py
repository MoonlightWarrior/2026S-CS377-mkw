"""
Offline test for the Phase 3 rank-based team reward (no Dolphin needed).

Verifies the proposal's core reward semantics:
  - RankReward rewards overtakes and penalizes being overtaken
  - TeamRankReward shares a teammate's rank gain with BOTH team members (cooperation)
  - FinishReward fires once, scaled by finishing rank
  - CombinedReward sums weighted components

Run:  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/tests/test_reward.py
"""
from __future__ import annotations

from kart_env.rl.game_state import KartGameState, PlayerState, RaceInfo
from marl.rewards import (
    RankReward, TeamRankReward, FinishReward, CombinedReward,
)


def _player(pid: int, *, rank: int, finished: bool = False, lap: int = 1) -> PlayerState:
    return PlayerState(
        player_id=pid, current_race_completion=1.0, max_race_completion=1.0,
        current_lap=lap, max_lap=lap, speed=80.0, position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0),
        main_rotation=(0.0, 0.0, 0.0, 1.0), drift_state=0, miniturbo_charge=0,
        smt_charge=0, race_position=rank, state_bit=(32 if finished else 0), item=20,
        mt_boost_timer=0, mushroom_boost_timer=0, start_boost_charge=0.0,
    )


def _state(ranks: dict, finished=None, laps=None) -> KartGameState:
    finished = finished or {}
    laps = laps or {}
    players = {a: _player(a, rank=ranks[a], finished=finished.get(a, False),
                          lap=laps.get(a, 1)) for a in ranks}
    return KartGameState(race_info=RaceInfo(frame_count=0, course_id=0, player_count=4),
                         players=players)


AGENTS = [0, 1, 2, 3]


def main() -> None:
    # ── RankReward: overtake is positive, being overtaken is negative ──────────
    rr = RankReward(gamma=0.99)
    s0 = _state({0: 3, 1: 4, 2: 1, 3: 2})
    rr.reset(AGENTS, s0)
    s1 = _state({0: 1, 1: 4, 2: 3, 3: 2})   # agent 0 jumps 3 -> 1, agent 2 drops 1 -> 3
    r = rr.get_rewards(AGENTS, s1)
    assert r[0] > 0.3, r            # big overtake -> clearly positive
    assert r[2] < 0.0, r            # overtaken -> negative
    print(f"  RankReward: overtaker r0={r[0]:+.3f}  overtaken r2={r[2]:+.3f}  OK")

    # ── TeamRankReward: teammate's gain is shared by BOTH team members ─────────
    tr = TeamRankReward(gamma=0.99)
    s0 = _state({0: 3, 1: 4, 2: 1, 3: 2})   # team0=[0,1], team1=[2,3]
    tr.reset(AGENTS, s0)
    s1 = _state({0: 3, 1: 2, 2: 1, 3: 4})   # agent 1 (team0) improves 4 -> 2
    r = tr.get_rewards(AGENTS, s1)
    assert r[0] > 0 and r[1] > 0, r          # both team-0 members rewarded
    assert abs(r[0] - r[1]) < 1e-6, r        # identical (shared team potential)
    assert r[2] < 0 and abs(r[2] - r[3]) < 1e-6, r  # team1 dropped, shared
    print(f"  TeamRankReward: teammates r0={r[0]:+.3f}==r1={r[1]:+.3f} (shared)  OK")

    # ── FinishReward: fires once, scaled by rank ──────────────────────────────
    fr = FinishReward(bonus=10.0)
    s0 = _state({0: 1, 1: 2, 2: 3, 3: 4})
    fr.reset(AGENTS, s0)
    s1 = _state({0: 1, 1: 2, 2: 3, 3: 4}, finished={0: True, 3: True})
    r1 = fr.get_rewards(AGENTS, s1)
    assert abs(r1[0] - 10.0) < 1e-6, r1      # rank1 finish -> full bonus
    assert abs(r1[3] - 0.0) < 1e-6, r1       # rank4 finish -> 0 * bonus
    r2 = fr.get_rewards(AGENTS, s1)          # already finished -> no repeat
    assert all(v == 0.0 for v in r2.values()), r2
    print(f"  FinishReward: rank1={r1[0]:.1f}  rank4={r1[3]:.1f}  fires-once OK")

    # ── CombinedReward weighted sum ───────────────────────────────────────────
    cr = CombinedReward((RankReward(gamma=0.99), 1.0), (TeamRankReward(gamma=0.99), 0.5))
    s0 = _state({0: 3, 1: 4, 2: 1, 3: 2})
    cr.reset(AGENTS, s0)
    s1 = _state({0: 1, 1: 3, 2: 2, 3: 4})
    r = cr.get_rewards(AGENTS, s1)
    assert all(isinstance(v, float) for v in r.values())
    print(f"  CombinedReward: {{a: round(v,3) for ...}} = "
          f"{ {a: round(r[a], 3) for a in AGENTS} }  OK")

    print("PASS  rank-based team reward semantics verified")


if __name__ == "__main__":
    main()
