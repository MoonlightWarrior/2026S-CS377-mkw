"""
Phase 5 — Behavioral analysis for the two research questions.

RolloutAnalyzer consumes a rollout step-by-step (KartGameState + the discrete
action each agent took) and, at the end, produces the proposal's metrics:

RQ1 — rank-differentiated item strategy:
    * item-holding time (frames a kart holds an item before using it),
      conditioned on the holder's rank at the moment of use. The headline
      contrast is rank 1 (defensive / long holding) vs the last rank
      (aggressive / immediate use). Collapse <=> the distributions coincide.
    * item-use rate conditioned on rank.

RQ2 — team cooperation (approximate, no hit address in memory yet):
    * a teammate-vs-opponent "hit" proxy: when a kart fires an item, any other
      kart whose speed drops sharply within a short window is counted as a hit,
      classified by team. Reported as the opponent-hit : teamkill ratio.

Plus domain behaviors: overtakes (rank improvements) and mini-turbo activations.

Everything is computed from a stored per-step time series, so the analyzer is
fully unit-testable offline (see marl/tests/test_metrics.py).
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

EMPTY_ITEM = 20


def _team_of(agent_id: int) -> int:
    return int(agent_id) // 2


class RolloutAnalyzer:
    def __init__(self, item_use_actions: Set[int], speed_drop_thresh: float = 15.0,
                 hit_window: int = 3):
        """
        item_use_actions  : discrete action indices that press the item button.
        speed_drop_thresh : speed units/step fall that counts as "got hit".
        hit_window        : steps after an item use to look for a victim speed drop.
        """
        self.item_use_actions = set(item_use_actions)
        self.speed_drop_thresh = speed_drop_thresh
        self.hit_window = hit_window
        self.reset()

    def reset(self) -> None:
        self._steps: List[dict] = []   # each: {agent: {rank,item,speed,boost,action,used_item}}

    def record(self, state, actions: Dict[int, int]) -> None:
        snap = {}
        for a, p in state.players.items():
            snap[a] = {
                "rank":   int(p.race_position),
                "item":   int(p.item),
                "speed":  float(p.speed),
                "boost":  int(p.mt_boost_timer),
                "action": int(actions.get(a, -1)),
                "used_item": (int(actions.get(a, -1)) in self.item_use_actions
                              and p.item != EMPTY_ITEM),
            }
        self._steps.append(snap)

    # ── metrics ──────────────────────────────────────────────────────────────
    def _agents(self) -> List[int]:
        return list(self._steps[0].keys()) if self._steps else []

    def item_holding_by_rank(self) -> Dict[int, dict]:
        """Mean item-holding frames before use, grouped by rank at time of use."""
        durations: Dict[int, List[int]] = {}   # rank -> list of holding lengths
        for a in self._agents():
            holding_since = None
            for t, snap in enumerate(self._steps):
                s = snap[a]
                had = s["item"] != EMPTY_ITEM
                if had and holding_since is None:
                    holding_since = t
                if s["used_item"] and holding_since is not None:
                    durations.setdefault(s["rank"], []).append(t - holding_since)
                    holding_since = None
                if not had:
                    holding_since = None
        return {r: {"mean_hold": float(np.mean(v)), "n": len(v)}
                for r, v in sorted(durations.items())}

    def item_use_rate_by_rank(self) -> Dict[int, float]:
        """Fraction of item-holding steps on which the kart fires, per rank."""
        fired = {}
        held = {}
        for snap in self._steps:
            for a, s in snap.items():
                if s["item"] != EMPTY_ITEM:
                    held[s["rank"]] = held.get(s["rank"], 0) + 1
                    if s["used_item"]:
                        fired[s["rank"]] = fired.get(s["rank"], 0) + 1
        return {r: fired.get(r, 0) / held[r] for r in sorted(held)}

    def overtakes(self) -> Dict[int, int]:
        """Count of rank improvements (rank decreases) per agent."""
        out = {a: 0 for a in self._agents()}
        for t in range(1, len(self._steps)):
            for a in self._agents():
                if self._steps[t][a]["rank"] < self._steps[t - 1][a]["rank"]:
                    out[a] += 1
        return out

    def miniturbo_activations(self) -> Dict[int, int]:
        out = {a: 0 for a in self._agents()}
        for t in range(1, len(self._steps)):
            for a in self._agents():
                if self._steps[t][a]["boost"] > 0 and self._steps[t - 1][a]["boost"] == 0:
                    out[a] += 1
        return out

    def hit_proxy(self) -> Dict[str, int]:
        """
        Approximate hits: for each item-use, attribute a sharp speed drop in any
        OTHER kart within `hit_window` steps to that user, classified by team.
        Returns opponent_hits, teamkills (and their ratio in summary()).
        """
        opp, team = 0, 0
        T = len(self._steps)
        for t in range(T):
            for a, s in self._steps[t].items():
                if not s["used_item"]:
                    continue
                for b in self._agents():
                    if b == a:
                        continue
                    for dt in range(1, self.hit_window + 1):
                        if t + dt >= T:
                            break
                        drop = self._steps[t + dt - 1][b]["speed"] - self._steps[t + dt][b]["speed"]
                        if drop >= self.speed_drop_thresh:
                            if _team_of(b) == _team_of(a):
                                team += 1
                            else:
                                opp += 1
                            break
        return {"opponent_hits": opp, "teamkills": team}

    def summary(self) -> dict:
        hits = self.hit_proxy()
        ratio = hits["opponent_hits"] / max(hits["teamkills"], 1)
        hold = self.item_holding_by_rank()
        ranks_sorted = sorted(hold)
        # headline RQ1 contrast: best rank vs worst rank seen at item use
        differentiation = None
        if len(ranks_sorted) >= 2:
            differentiation = hold[ranks_sorted[0]]["mean_hold"] - hold[ranks_sorted[-1]]["mean_hold"]
        return {
            "item_holding_by_rank": hold,
            "item_use_rate_by_rank": self.item_use_rate_by_rank(),
            "holding_differentiation_best_minus_worst": differentiation,
            "overtakes_total": int(sum(self.overtakes().values())),
            "miniturbo_total": int(sum(self.miniturbo_activations().values())),
            "opponent_hits": hits["opponent_hits"],
            "teamkills": hits["teamkills"],
            "opp_to_teamkill_ratio": ratio,
            "n_steps": len(self._steps),
        }
