"""
Offline test for the Phase 5 analysis harness (no Dolphin needed).

Builds a synthetic rollout in which a rank-1 kart holds an item far longer than a
rank-4 kart before using it, an item use is followed by an opponent speed drop,
plus an overtake and a mini-turbo activation — and checks the analyzer recovers
all of it.

Run:  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/tests/test_metrics.py
"""
from __future__ import annotations

from kart_env.rl.game_state import KartGameState, PlayerState, RaceInfo
from marl.analysis import RolloutAnalyzer
from marl.action import ITEM_USE_ACTIONS


def _player(pid, rank, item, speed, boost) -> PlayerState:
    return PlayerState(
        player_id=pid, current_race_completion=1.0, max_race_completion=1.0,
        current_lap=1, max_lap=1, speed=speed, position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0),
        main_rotation=(0.0, 0.0, 0.0, 1.0), drift_state=0, miniturbo_charge=0,
        smt_charge=0, race_position=rank, state_bit=0, item=item,
        mt_boost_timer=boost, mushroom_boost_timer=0, start_boost_charge=0.0,
    )


def _state(per_agent) -> KartGameState:
    players = {a: _player(a, **per_agent[a]) for a in per_agent}
    return KartGameState(race_info=RaceInfo(frame_count=0, course_id=0, player_count=4),
                         players=players)


def main() -> None:
    az = RolloutAnalyzer(item_use_actions=ITEM_USE_ACTIONS, speed_drop_thresh=15.0, hit_window=3)

    NO = 0           # forward, no item
    USE = 2          # item-use action
    rows = []        # (per_agent_state, actions)
    for t in range(13):
        # ranks: agent1 overtakes agent3 at t>=5
        r1, r3 = (3, 2) if t < 5 else (2, 3)
        st = {
            0: dict(rank=1, item=(5 if 2 <= t <= 10 else 20), speed=80, boost=0),
            1: dict(rank=r1, item=20, speed=80, boost=0),
            2: dict(rank=4, item=(5 if t == 8 or t == 9 else 20),
                    speed=(60 if t == 11 else 80), boost=0),       # speed drop at t=11
            3: dict(rank=r3, item=20, speed=80, boost=(30 if t == 6 else 0)),  # MT at t=6
        }
        act = {0: NO, 1: NO, 2: NO, 3: NO}
        if t == 10:
            act[0] = USE     # rank-1 kart uses after holding since t=2  -> hold 8
        if t == 9:
            act[2] = USE     # rank-4 kart uses after holding since t=8  -> hold 1
        rows.append((_state(st), act))

    for state, actions in rows:
        az.record(state, actions)

    s = az.summary()
    hold = s["item_holding_by_rank"]
    assert 1 in hold and 4 in hold, hold
    assert hold[1]["mean_hold"] > hold[4]["mean_hold"], hold      # rank1 holds longer
    assert s["holding_differentiation_best_minus_worst"] > 0, s   # RQ1 differentiation
    assert s["overtakes_total"] >= 1, s
    assert s["miniturbo_total"] >= 1, s
    assert s["opponent_hits"] >= 1, s
    assert s["teamkills"] == 0, s

    print(f"PASS  rank1 hold={hold[1]['mean_hold']:.0f} vs rank4 hold={hold[4]['mean_hold']:.0f}  "
          f"diff={s['holding_differentiation_best_minus_worst']:.0f}  "
          f"overtakes={s['overtakes_total']}  MT={s['miniturbo_total']}  "
          f"opp_hits={s['opponent_hits']} teamkills={s['teamkills']}")


if __name__ == "__main__":
    main()
