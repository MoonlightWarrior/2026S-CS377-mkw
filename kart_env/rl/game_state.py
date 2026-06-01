from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

AgentID = int
RawObs = Dict[str, Any]


@dataclass
class PlayerState:
    player_id: int
    current_race_completion: float
    max_race_completion: float
    current_lap: int
    max_lap: int
    speed: float
    position: tuple
    velocity: tuple
    angular_velocity: tuple
    main_rotation: tuple
    drift_state: int
    miniturbo_charge: int
    smt_charge: int
    race_position: int
    state_bit: int
    item: int
    mt_boost_timer: int
    mushroom_boost_timer: int
    start_boost_charge: float
    # speed caps: hard = kart's max; soft drops below it when off-road (grass).
    # Defaulted so existing direct constructions (tests) keep working.
    soft_speed_limit: float = 0.0
    hard_speed_limit: float = 0.0

    @classmethod
    def from_obs(cls, obs: RawObs) -> PlayerState:
        p = obs["PLAYER_INFO"]
        return cls(
            player_id=int(p["PlayerID"]),
            current_race_completion=float(p["CurrentRaceCompletion"]),
            max_race_completion=float(p["MaxRaceCompletion"]),
            current_lap=int(p["CurrentLap"]),
            max_lap=int(p["MaxLap"]),
            speed=float(p["Speed"]),
            position=tuple(float(x) for x in p["Position"]),
            velocity=tuple(float(x) for x in p["Velocity"]),
            angular_velocity=tuple(float(x) for x in p["AngularVelocity"]),
            main_rotation=tuple(float(x) for x in p["MainRotation"]),
            drift_state=int(p["DriftState"]),
            miniturbo_charge=int(p["MiniturboCharge"]),
            smt_charge=int(p["SMiniturboCharge"]),
            race_position=int(p["RacePosition"]),
            state_bit=int(p["StateBit"]),
            item=int(p["Item"]),
            mt_boost_timer=int(p["MTBoostTimer"]),
            mushroom_boost_timer=int(p["MushroomBoostTimer"]),
            start_boost_charge=float(p["startBoostCharge"]),
            soft_speed_limit=float(p["SoftSpeedLimit"]),
            hard_speed_limit=float(p["HardSpeedLimit"]),
        )

    @property
    def is_finished(self) -> bool:
        return bool(self.state_bit & 32)


@dataclass
class RaceInfo:
    frame_count: int
    course_id: int
    player_count: int

    @classmethod
    def from_obs(cls, obs: RawObs) -> RaceInfo:
        r = obs["RACE_INFO"]
        return cls(
            frame_count=int(r["FrameCount"]),
            course_id=int(r["CourseID"]),
            player_count=int(r["PlayerCount"]),
        )


@dataclass
class KartGameState:
    race_info: RaceInfo
    players: Dict[AgentID, PlayerState]

    @classmethod
    def from_obs_dict(cls, obs_dict: Dict[AgentID, RawObs]) -> KartGameState:
        race_info = None
        players: Dict[AgentID, PlayerState] = {}
        for agent_id, obs in obs_dict.items():
            players[agent_id] = PlayerState.from_obs(obs)
            if race_info is None:
                race_info = RaceInfo.from_obs(obs)
        return cls(race_info=race_info, players=players)
