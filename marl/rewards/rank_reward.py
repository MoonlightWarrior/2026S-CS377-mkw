"""
Phase 3 — Rank-based team reward.

Implements the proposal's "shared rank-based reward [that] balances individual
rank with team-level success". Two primary signals, both **potential-based**
(r_t = gamma * Phi(s') - Phi(s)) so they give a dense, stable overtake/defence
signal without changing the optimal policy:

  RankReward      — individual: potential = the agent's own rank value.
  TeamRankReward  — team: potential = mean rank value of the agent's 2-kart team
                    (shared by both teammates → rewards cooperation).

Plus sparse milestones and light technique shaping for learnability:
  FinishReward    — one-off bonus on crossing the line, scaled by finishing rank.
  LapReward       — small bonus each new lap.
  SpeedReward     — tiny dense keep-moving term.
  MTBoostReward   — reward on each mini-turbo boost activation.

`rank value` = (N - race_position) / (N - 1) ∈ [0,1], 1 = first place, computed
against the LIVE field size N = race_info.player_count, so it is correct for both
a clean 4-kart 2v2 (ranks 1–4) and a 12-kart field (ranks 1–12).

Team layout: agents [0,1] vs [2,3] (team = agent_id // 2).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

EMPTY_ITEM = 20  # MKW "no item" sentinel (Phase 0)


def _rank_value(player, player_count: int) -> float:
    n = max(player_count, 2)
    return (n - player.race_position) / (n - 1)


def _team_of(agent_id: int) -> int:
    return int(agent_id) // 2


class RewardFunction(ABC):
    @abstractmethod
    def reset(self, agents: List[int], state) -> None: ...
    @abstractmethod
    def get_rewards(self, agents: List[int], state) -> Dict[int, float]: ...


class CombinedReward(RewardFunction):
    """Weighted sum of component rewards. Build with (fn, weight) pairs."""

    def __init__(self, *weighted):
        self.weighted = list(weighted)

    def reset(self, agents, state) -> None:
        for fn, _ in self.weighted:
            fn.reset(agents, state)

    def get_rewards(self, agents, state) -> Dict[int, float]:
        total = {a: 0.0 for a in agents}
        for fn, w in self.weighted:
            part = fn.get_rewards(agents, state)
            for a in agents:
                total[a] += w * part[a]
        return total


class RankReward(RewardFunction):
    """Individual potential-based rank shaping."""

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def reset(self, agents, state) -> None:
        n = state.race_info.player_count
        self.prev = {a: _rank_value(state.players[a], n) for a in agents}

    def get_rewards(self, agents, state) -> Dict[int, float]:
        n = state.race_info.player_count
        out = {}
        for a in agents:
            phi = _rank_value(state.players[a], n)
            out[a] = self.gamma * phi - self.prev[a]
            self.prev[a] = phi
        return out


class TeamRankReward(RewardFunction):
    """Team-success potential: mean rank value of the team, shared by both members."""

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def _team_phi(self, agents, state) -> Dict[int, float]:
        n = state.race_info.player_count
        teams: Dict[int, List[float]] = {}
        for a in agents:
            teams.setdefault(_team_of(a), []).append(_rank_value(state.players[a], n))
        return {t: sum(v) / len(v) for t, v in teams.items()}

    def reset(self, agents, state) -> None:
        self.prev = self._team_phi(agents, state)

    def get_rewards(self, agents, state) -> Dict[int, float]:
        phi = self._team_phi(agents, state)
        out = {}
        for a in agents:
            t = _team_of(a)
            out[a] = self.gamma * phi[t] - self.prev[t]
        self.prev = phi
        return out


class FinishReward(RewardFunction):
    """One-off bonus when a kart crosses the line, scaled by its finishing rank."""

    def __init__(self, bonus: float = 10.0):
        self.bonus = bonus

    def reset(self, agents, state) -> None:
        self.fired = {a: False for a in agents}

    def get_rewards(self, agents, state) -> Dict[int, float]:
        n = state.race_info.player_count
        out = {}
        for a in agents:
            p = state.players[a]
            if p.is_finished and not self.fired[a]:
                out[a] = self.bonus * _rank_value(p, n)
                self.fired[a] = True
            else:
                out[a] = 0.0
        return out


class LapReward(RewardFunction):
    """Small bonus on each new lap (sparse progress milestone)."""

    def __init__(self, bonus: float = 1.0):
        self.bonus = bonus

    def reset(self, agents, state) -> None:
        self.prev_lap = {a: state.players[a].current_lap for a in agents}

    def get_rewards(self, agents, state) -> Dict[int, float]:
        out = {}
        for a in agents:
            lap = state.players[a].current_lap
            out[a] = self.bonus if lap > self.prev_lap[a] else 0.0
            self.prev_lap[a] = lap
        return out


class RaceProgressReward(RewardFunction):
    """
    Dense forward-progress: reward the increase in max_race_completion (which goes
    0 -> 3 over a 3-lap race). This is a *directional* progress signal — unlike raw
    speed it only pays for actually advancing along the track — so it drives the
    agent toward finishing rather than just moving fast. Telescopes to
    `scale * final_completion` over an episode.
    """

    def __init__(self, scale: float = 1.0, use_current: bool = False):
        self.scale = scale
        # use_current: key on current_race_completion (SIGNED delta) instead of
        # max_race_completion (monotone). Signed delta yields a NEGATIVE reward for
        # going backward / looping off the racing line, giving a directional
        # gradient that pulls the agent back toward the track — targeting the
        # off-road-wandering failure mode without an explicit (timid) offroad penalty.
        self.use_current = use_current

    def _c(self, p):
        return p.current_race_completion if self.use_current else p.max_race_completion

    def reset(self, agents, state) -> None:
        self.prev = {a: self._c(state.players[a]) for a in agents}

    def get_rewards(self, agents, state) -> Dict[int, float]:
        out = {}
        for a in agents:
            c = self._c(state.players[a])
            delta = c - self.prev[a]
            out[a] = self.scale * (delta if self.use_current else max(0.0, delta))
            self.prev[a] = c
        return out


class SpeedReward(RewardFunction):
    """Tiny dense keep-moving signal (normalized speed). NOTE: prefer
    RaceProgressReward — raw speed is direction-agnostic and over-rewards moving
    fast without making track progress."""

    def __init__(self, speed_norm: float = 120.0):
        self.speed_norm = speed_norm

    def reset(self, agents, state) -> None:
        pass

    def get_rewards(self, agents, state) -> Dict[int, float]:
        return {a: state.players[a].speed / self.speed_norm for a in agents}


class ProgressGatedSpeedReward(RewardFunction):
    """Reward normalized speed ONLY while the kart is making forward track
    progress (current_race_completion increased this step).

    This pays for *carrying speed along the racing line* while avoiding both
    failure modes seen earlier: (i) raw SpeedReward is direction-agnostic and
    rewards fast-the-wrong-way motion — here, going the wrong way does not advance
    completion, so it earns nothing; (ii) an explicit OffroadPenalty made the
    policy timid — here off-road simply earns less, because grass caps speed and
    stalls progress, but is never punished below zero.
    """

    def __init__(self, speed_norm: float = 120.0, eps: float = 1e-4):
        self.speed_norm = speed_norm
        self.eps = eps

    def reset(self, agents, state) -> None:
        self.prev = {a: state.players[a].current_race_completion for a in agents}

    def get_rewards(self, agents, state) -> Dict[int, float]:
        out = {}
        for a in agents:
            p = state.players[a]
            c = p.current_race_completion
            progressing = (c - self.prev[a]) > self.eps
            out[a] = (p.speed / self.speed_norm) if progressing else 0.0
            self.prev[a] = c
        return out


class MTBoostReward(RewardFunction):
    """Reward each mini-turbo boost activation (0 -> >0 edge on the boost timer)."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reset(self, agents, state) -> None:
        self.prev = {a: state.players[a].mt_boost_timer for a in agents}

    def get_rewards(self, agents, state) -> Dict[int, float]:
        out = {}
        for a in agents:
            curr = state.players[a].mt_boost_timer
            out[a] = self.scale if (curr > 0 and self.prev[a] == 0) else 0.0
            self.prev[a] = curr
        return out


class OffroadPenalty(RewardFunction):
    """
    Penalize being off-road (grass), the agents' main failure mode. MKW lowers the
    SOFT speed limit below the HARD (kart-max) limit on rough terrain, so
        frac = max(0, hard - soft) / hard
    is ~0 on the track and grows toward 1 the more off-road the kart is. The
    per-step reward is -scale * frac. Threshold-free: exactly 0 on the road, so it
    can only ever discourage leaving it (never penalizes slow on-track driving).
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reset(self, agents, state) -> None:
        pass

    def get_rewards(self, agents, state) -> Dict[int, float]:
        out = {}
        for a in agents:
            p = state.players[a]
            hard = p.hard_speed_limit
            frac = max(0.0, hard - p.soft_speed_limit) / hard if hard > 1e-3 else 0.0
            out[a] = -self.scale * min(frac, 1.0)
        return out


# ── registry + builder ───────────────────────────────────────────────────────
_REGISTRY = {
    "RankReward": RankReward,
    "TeamRankReward": TeamRankReward,
    "RaceProgressReward": RaceProgressReward,
    "FinishReward": FinishReward,
    "LapReward": LapReward,
    "SpeedReward": SpeedReward,
    "ProgressGatedSpeedReward": ProgressGatedSpeedReward,
    "MTBoostReward": MTBoostReward,
    "OffroadPenalty": OffroadPenalty,
}


def build_reward(reward_cfg, gamma: float) -> CombinedReward:
    """
    Build a CombinedReward from the YAML `reward.reward_functions` list. Entries
    are either `Name: weight` or `Name: {weight: w, params: {...}}`. Potential-based
    components (RankReward, TeamRankReward) receive `gamma` automatically.
    """
    from omegaconf import OmegaConf

    weighted = []
    for item in reward_cfg.reward_functions:
        for name, value in item.items():
            cls = _REGISTRY[name]
            if isinstance(value, (int, float)):
                weight, params = float(value), {}
            else:
                weight = float(value.weight)
                params = OmegaConf.to_container(value.get("params", {})) or {}
            if name in ("RankReward", "TeamRankReward"):
                params.setdefault("gamma", gamma)
            weighted.append((cls(**params), weight))
    return CombinedReward(*weighted)
