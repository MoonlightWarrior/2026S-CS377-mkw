from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from .game_state import AgentID, KartGameState


class ObsBuilder(ABC):
    @abstractmethod
    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    @abstractmethod
    def build_obs(self, agent_id: AgentID, state: KartGameState) -> np.ndarray:
        pass

    @abstractmethod
    def get_obs_size(self) -> int:
        pass


class KartVectorObs(ObsBuilder):
    """
    Flat normalized vector combining self-player physics state and
    relative positions/progress of all other agents.

    Self dims (19):
        current_race_completion, max_race_completion, current_lap, speed,
        position (3), velocity (3), main_rotation (4), angular_velocity (3),
        drift_state, race_position

    Per-other-agent dims (6):
        relative position (3), speed, race_position, current_race_completion
    """

    SELF_DIM = 19
    PER_OTHER_DIM = 6

    def __init__(
        self,
        num_agents: int = 4,
        speed_norm: float = 120.0,
        pos_norm: float = 10000.0,
        ang_vel_norm: float = 3.0,
    ):
        self.num_agents = num_agents
        self.speed_norm = speed_norm
        self.pos_norm = pos_norm
        self.ang_vel_norm = ang_vel_norm

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    def get_obs_size(self) -> int:
        return self.SELF_DIM + (self.num_agents - 1) * self.PER_OTHER_DIM

    def build_obs(self, agent_id: AgentID, state: KartGameState) -> np.ndarray:
        me = state.players[agent_id]
        n = max(self.num_agents, 1)
        obs: List[float] = []

        obs += [
            me.current_race_completion / 3.0,
            me.max_race_completion / 3.0,
            me.current_lap / 3.0,
            me.speed / self.speed_norm,
            me.position[0] / self.pos_norm,
            me.position[1] / self.pos_norm,
            me.position[2] / self.pos_norm,
            me.velocity[0] / self.speed_norm,
            me.velocity[1] / self.speed_norm,
            me.velocity[2] / self.speed_norm,
            me.main_rotation[0],
            me.main_rotation[1],
            me.main_rotation[2],
            me.main_rotation[3],
            me.angular_velocity[0] / self.ang_vel_norm,
            me.angular_velocity[1] / self.ang_vel_norm,
            me.angular_velocity[2] / self.ang_vel_norm,
            me.drift_state / 3.0,
            me.race_position / n,
        ]

        for other_id, other in state.players.items():
            if other_id == agent_id:
                continue
            obs += [
                (other.position[0] - me.position[0]) / self.pos_norm,
                (other.position[1] - me.position[1]) / self.pos_norm,
                (other.position[2] - me.position[2]) / self.pos_norm,
                other.speed / self.speed_norm,
                other.race_position / n,
                other.current_race_completion / 3.0,
            ]

        # Zero-pad if fewer players than expected (e.g., online mode with 1 agent)
        expected = self.get_obs_size()
        if len(obs) < expected:
            obs += [0.0] * (expected - len(obs))

        return np.array(obs, dtype=np.float32)


class TeamVectorObs(ObsBuilder):
    """
    Extends KartVectorObs for 2v2 team play.

    Self dims (22): same 19 as KartVectorObs + item + miniturbo_charge + mt_boost_timer

    Per-other-agent dims (7): relative position (3), speed, race_completion, item, is_teammate flag

    Teams: [0,1] vs [2,3] (hardcoded for 4-agent 2v2 setup).
    race_position normalized by the full 12-kart field (MKW always tracks 12 positions).
    """

    SELF_DIM = 22
    PER_OTHER_DIM = 7

    def __init__(
        self,
        num_agents: int = 4,
        speed_norm: float = 120.0,
        pos_norm: float = 10000.0,
        ang_vel_norm: float = 3.0,
        item_norm: float = 100.0,
        charge_norm: float = 270.0,
        boost_norm: float = 100.0,
    ):
        self.num_agents = num_agents
        self.speed_norm = speed_norm
        self.pos_norm = pos_norm
        self.ang_vel_norm = ang_vel_norm
        self.item_norm = item_norm
        self.charge_norm = charge_norm
        self.boost_norm = boost_norm

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    def get_obs_size(self) -> int:
        return self.SELF_DIM + (self.num_agents - 1) * self.PER_OTHER_DIM

    def build_obs(self, agent_id: AgentID, state: KartGameState) -> np.ndarray:
        me = state.players[agent_id]
        n_field = max(state.race_info.player_count, 1)  # 12 in MKW
        obs: List[float] = []

        obs += [
            me.current_race_completion / 3.0,
            me.max_race_completion / 3.0,
            me.current_lap / 3.0,
            me.speed / self.speed_norm,
            me.position[0] / self.pos_norm,
            me.position[1] / self.pos_norm,
            me.position[2] / self.pos_norm,
            me.velocity[0] / self.speed_norm,
            me.velocity[1] / self.speed_norm,
            me.velocity[2] / self.speed_norm,
            me.main_rotation[0],
            me.main_rotation[1],
            me.main_rotation[2],
            me.main_rotation[3],
            me.angular_velocity[0] / self.ang_vel_norm,
            me.angular_velocity[1] / self.ang_vel_norm,
            me.angular_velocity[2] / self.ang_vel_norm,
            me.drift_state / 3.0,
            me.race_position / n_field,
            min(me.item / self.item_norm, 1.0),
            min(me.miniturbo_charge / self.charge_norm, 1.0),
            min(me.mt_boost_timer / self.boost_norm, 1.0),
        ]

        my_team = agent_id // 2
        for other_id, other in state.players.items():
            if other_id == agent_id:
                continue
            is_teammate = 1.0 if (other_id // 2 == my_team) else 0.0
            obs += [
                (other.position[0] - me.position[0]) / self.pos_norm,
                (other.position[1] - me.position[1]) / self.pos_norm,
                (other.position[2] - me.position[2]) / self.pos_norm,
                other.speed / self.speed_norm,
                other.current_race_completion / 3.0,
                min(other.item / self.item_norm, 1.0),
                is_teammate,
            ]

        expected = self.get_obs_size()
        if len(obs) < expected:
            obs += [0.0] * (expected - len(obs))

        return np.array(obs, dtype=np.float32)
