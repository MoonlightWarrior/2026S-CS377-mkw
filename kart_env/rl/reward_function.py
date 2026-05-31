from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from .game_state import AgentID, KartGameState


class RewardFunction(ABC):
    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    @abstractmethod
    def get_rewards(
        self,
        agents: List[AgentID],
        state: KartGameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
    ) -> Dict[AgentID, float]:
        pass


class CombinedReward(RewardFunction):
    """Weighted sum of multiple reward functions, mirroring RLGym CombinedReward."""

    def __init__(self, *weighted_rewards: tuple[RewardFunction, float]):
        self.weighted_rewards = weighted_rewards

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        for fn, _ in self.weighted_rewards:
            fn.reset(agents, initial_state)

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        totals: Dict[AgentID, float] = {a: 0.0 for a in agents}
        for fn, weight in self.weighted_rewards:
            partial = fn.get_rewards(agents, state, is_terminated, is_truncated)
            for agent, r in partial.items():
                totals[agent] += weight * r
        return totals


class RaceProgressReward(RewardFunction):
    """
    Delta of MaxRaceCompletion — the primary forward-progress signal.
    Negative deltas (respawn) are clamped to zero.
    """

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev: Dict[AgentID, float] = {
            a: initial_state.players[a].max_race_completion for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            curr = state.players[agent].max_race_completion
            rewards[agent] = max(0.0, curr - self._prev.get(agent, 0.0))
            self._prev[agent] = curr
        return rewards


class SpeedReward(RewardFunction):
    """Small dense reward for maintaining speed — prevents standing still."""

    def __init__(self, speed_norm: float = 120.0):
        self.speed_norm = speed_norm

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        return {
            agent: state.players[agent].speed / self.speed_norm
            for agent in agents
        }


class FinishReward(RewardFunction):
    """One-time bonus when an agent finishes the race (StateBit & 32)."""

    def __init__(self, bonus: float = 10.0):
        self.bonus = bonus

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            finished = is_terminated[agent] and state.players[agent].is_finished
            rewards[agent] = self.bonus if finished else 0.0
        return rewards


class RacePositionReward(RewardFunction):
    """
    Reward proportional to race ranking — (n - rank) / n.
    Rank 1 = highest reward, last place = near zero.
    """

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        n = max(len(agents), 1)
        return {
            agent: (n - state.players[agent].race_position) / n
            for agent in agents
        }


class MTBoostReward(RewardFunction):
    """Small reward for landing a mini-turbo boost, encouraging drift technique."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev_boost: Dict[AgentID, int] = {
            a: initial_state.players[a].mt_boost_timer for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            curr = state.players[agent].mt_boost_timer
            prev = self._prev_boost.get(agent, 0)
            # Reward the moment a boost activates (timer goes from 0 → positive)
            rewards[agent] = self.scale if (curr > 0 and prev == 0) else 0.0
            self._prev_boost[agent] = curr
        return rewards


class SmoothDrivingReward(RewardFunction):
    """
    Dense reward in [0, 1] for low angular velocity when not drifting.

    Discrete-action agents tend to develop jittery steering that wastes speed
    through micro-corrections. This penalizes erratic angular velocity without
    punishing intentional high-yaw moves: the reward is skipped entirely during
    a drift (drift_state != 0) so it does not conflict with MTBoostReward.

    ang_vel_norm: angular velocity magnitude that maps to reward 0.0; tune to
    the typical non-drift yaw rate seen in your environment.
    """

    def __init__(self, ang_vel_norm: float = 2.0):
        self.ang_vel_norm = ang_vel_norm

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            p = state.players[agent]
            if p.drift_state != 0:
                rewards[agent] = 0.0
            else:
                ang_mag = sum(v ** 2 for v in p.angular_velocity) ** 0.5
                rewards[agent] = max(0.0, 1.0 - ang_mag / self.ang_vel_norm)
        return rewards


def _teammate(agent_id: AgentID) -> AgentID:
    """Return teammate for 2v2 layout: teams [0,1] vs [2,3]."""
    return (agent_id // 2) * 2 + (1 - agent_id % 2)


class TeamProgressReward(RewardFunction):
    """
    Own race-completion delta plus a fraction of the teammate's delta.
    Use this instead of RaceProgressReward in 2v2 mode to encourage cooperation.
    Teams assumed: [0,1] vs [2,3].
    """

    def __init__(self, teammate_weight: float = 0.3):
        self.teammate_weight = teammate_weight

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev: Dict[AgentID, float] = {
            a: initial_state.players[a].max_race_completion for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        deltas: Dict[AgentID, float] = {}
        for agent in agents:
            curr = state.players[agent].max_race_completion
            deltas[agent] = max(0.0, curr - self._prev.get(agent, 0.0))
            self._prev[agent] = curr

        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            mate = _teammate(agent)
            rewards[agent] = deltas[agent] + self.teammate_weight * deltas.get(mate, 0.0)
        return rewards


class LapCompletionReward(RewardFunction):
    """Sparse bonus each time an agent completes a new lap."""

    def __init__(self, bonus: float = 2.0):
        self.bonus = bonus

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev_lap: Dict[AgentID, int] = {
            a: initial_state.players[a].max_lap for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            curr_lap = state.players[agent].max_lap
            rewards[agent] = self.bonus if curr_lap > self._prev_lap.get(agent, 0) else 0.0
            self._prev_lap[agent] = curr_lap
        return rewards


class ItemUseReward(RewardFunction):
    """Small reward when an agent uses a held item (item field transitions non-zero → zero)."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev_item: Dict[AgentID, int] = {
            a: initial_state.players[a].item for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            curr = state.players[agent].item
            prev = self._prev_item.get(agent, 0)
            rewards[agent] = self.scale if (prev > 0 and curr == 0) else 0.0
            self._prev_item[agent] = curr
        return rewards


def _quat_forward(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """Rotate local forward (0,0,1) by quaternion — MKW convention."""
    return (
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )


class ForwardVelocityReward(RewardFunction):
    """
    Velocity projected onto the kart's forward direction, normalised by speed_norm.

    Returns a value in [-1, 1]; set clip_negative=True (default) to clamp to [0, 1]
    so reversing doesn't produce a negative signal that confuses the combined reward.
    This is strictly better than SpeedReward because it rewards going *forward* not
    just fast — an agent boosting in reverse gets zero, not a positive signal.

    speed_norm: maximum expected forward speed in the same units as PlayerState.speed.
    """

    def __init__(self, speed_norm: float = 120.0, clip_negative: bool = True):
        self.speed_norm = speed_norm
        self.clip_negative = clip_negative

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            p = state.players[agent]
            qx, qy, qz, qw = p.main_rotation
            fx, fy, fz = _quat_forward(qx, qy, qz, qw)
            vx, vy, vz = p.velocity
            proj = vx * fx + vy * fy + vz * fz
            r = proj / self.speed_norm
            rewards[agent] = max(0.0, r) if self.clip_negative else r
        return rewards


class MTChargeReward(RewardFunction):
    """
    Dense per-frame reward for increasing miniturbo_charge while drifting.

    Rewards the entire drift-charge loop (not just the moment of boost activation),
    giving the agent a continuous gradient toward efficient drift technique.
    Only fires while drift_state != 0 so it doesn't reward phantom charge ticks.

    charge_norm: max miniturbo_charge value — reward is delta / charge_norm.
    """

    def __init__(self, charge_norm: float = 270.0):
        self.charge_norm = charge_norm

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev: Dict[AgentID, int] = {
            a: initial_state.players[a].miniturbo_charge for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            p = state.players[agent]
            curr = p.miniturbo_charge
            prev = self._prev.get(agent, 0)
            if p.drift_state != 0:
                delta = curr - prev
                rewards[agent] = max(0.0, delta) / self.charge_norm
            else:
                rewards[agent] = 0.0
            self._prev[agent] = curr
        return rewards


class SMTChargeReward(RewardFunction):
    """
    Dense per-frame reward for increasing smt_charge (super mini-turbo) while drifting.

    Mirrors MTChargeReward but targets the higher-tier orange-sparks charge.
    Use alongside MTChargeReward to reward progression up the charge tiers.

    charge_norm: max smt_charge value — reward is delta / charge_norm.
    """

    def __init__(self, charge_norm: float = 270.0):
        self.charge_norm = charge_norm

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev: Dict[AgentID, int] = {
            a: initial_state.players[a].smt_charge for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            p = state.players[agent]
            curr = p.smt_charge
            prev = self._prev.get(agent, 0)
            if p.drift_state != 0:
                delta = curr - prev
                rewards[agent] = max(0.0, delta) / self.charge_norm
            else:
                rewards[agent] = 0.0
            self._prev[agent] = curr
        return rewards


class PositionImprovementReward(RewardFunction):
    """
    Bonus when an agent overtakes another kart (race_position decreases).

    Uses a per-agent delta so the agent is rewarded for each rank gained
    rather than for its absolute standing. Scale controls the bonus per rank gained.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        self._prev_pos: Dict[AgentID, int] = {
            a: initial_state.players[a].race_position for a in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            curr = state.players[agent].race_position
            prev = self._prev_pos.get(agent, curr)
            improvement = prev - curr  # positive when rank number drops (better place)
            rewards[agent] = self.scale * max(0, improvement)
            self._prev_pos[agent] = curr
        return rewards


class MushroomBoostActiveReward(RewardFunction):
    """
    Dense reward each frame while mushroom_boost_timer > 0.

    Pairs with ItemUseReward: ItemUseReward fires once on use, this provides
    a sustained signal for the duration of the boost so the agent learns to
    stay on track and exploit the speed window rather than using items wastefully.

    scale: reward per frame of active boost; keep small (default 0.01).
    """

    def __init__(self, scale: float = 0.01):
        self.scale = scale

    def get_rewards(self, agents, state, is_terminated, is_truncated):
        return {
            agent: self.scale if state.players[agent].mushroom_boost_timer > 0 else 0.0
            for agent in agents
        }
