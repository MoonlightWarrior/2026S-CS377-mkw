import functools
import os
from dataclasses import replace
from pettingzoo import ParallelEnv
from gymnasium import spaces
from typing import Any, Dict
import numpy as np

from .utils.dolphin import Dolphin
from .utils.dolphin_mem import DolphinMem
from .utils.kart_graphic_obs import KartGraphicObs
from .utils.helper import launch_game
from .utils.macro_helper import OptionType

ObsType = Dict[str, Any]
ActionType = Dict[str, Any]
AgentID = int


class KartEnvironment(ParallelEnv):
    metadata = {"name": "kart_environment"}

    def __init__(self, env_id: int = 0, options: OptionType | None = None):
        self.options = options if options is not None else OptionType()
        self._closed = False

        self.dolphins: list[Dolphin] = []
        self.dolphins_mem: list[DolphinMem] = []
        self.graphic_obss: list[KartGraphicObs] = []

        self.agents = [i for i in range(self.options.num_agents)]

        if self.options.online_mode:
            self.agent_mapper = lambda x: (
                x,
                0,
            )  # agent_id corresponds directly to dolphin index
            """mapping function from global agent_id to (dolphin_idx, agent_id_within_dolphin)"""

            for i in range(self.options.num_agents):
                _options = replace(
                    self.options,
                    num_agents=1,
                    character=[self.options.character[i]],
                    vehicle=[self.options.vehicle[i]],
                    drift_modes=[self.options.drift_modes[i]],
                )
                dolphin = Dolphin(instance_id=12 * env_id + i, options=_options)
                assert dolphin.dolphin_proc_pid is not None

                self.dolphins.append(dolphin)
                self.dolphins_mem.append(DolphinMem(dolphin.dolphin_proc_pid))
        else:
            self.agent_mapper = lambda x: (0, x)  # dolphin_idx, agent_id_within_dolphin

            dolphin = Dolphin(instance_id=12 * env_id, options=self.options)
            assert dolphin.dolphin_proc_pid is not None

            self.dolphins.append(dolphin)
            self.dolphins_mem.append(DolphinMem(dolphin.dolphin_proc_pid))

        launch_game(self, self.options)
        if os.environ.get("KART_NULL_RENDER") != "1":
            for dolphin in self.dolphins:
                self.graphic_obss.append(KartGraphicObs(dolphin.instance_id))

        # Advance frames until race memory is valid before taking the initial save state.
        for _ in range(6000):
            try:
                self.dolphins_mem[0].read_obs(self.options.num_agents)
                break
            except ValueError:
                self._send_actions({})
        self.save_slot(0)

    def __del__(self):
        self.close()

    def reset(
        self, seed=None, options: dict | None = {}
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        if options is None:
            options = {}

        if options.get("slot") is not None:
            self.load_slot(options["slot"])
        elif options.get("file") is not None:
            self.load_file(options["file"])
        else:
            self.load_slot(0)

        for _ in range(3000):
            try:
                observations = self._get_obs()
                break
            except ValueError:
                self._send_actions({})
        else:
            observations = self._get_obs()

        infos = {agent_id: {} for agent_id in self.agents}

        # TODO combine graphic_obs and vector_obs into a single observation dict

        return observations, infos

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        self._send_actions(actions)

        observations = self._get_obs()

        for agent_id in self.agents:
            rewards[agent_id] = 0.0

            termination = bool(observations[agent_id]["PLAYER_INFO"]["StateBit"] & 32)
            terminations[agent_id] = termination

            truncations[agent_id] = False

            infos[agent_id] = {}

        return observations, rewards, terminations, truncations, infos

    def close(self):
        if self._closed:
            return
        self._closed = True

        for graphic_obs in self.graphic_obss:
            graphic_obs.close()

        for dolphin in self.dolphins:
            dolphin.close()

    """HELPER FUNCTIONS"""

    def _get_obs(self) -> dict[AgentID, ObsType]:
        observations = {}

        if self.options.online_mode:
            for agent_id in self.agents:
                dolphin_idx, _ = self.agent_mapper(agent_id)
                raw_vector_obs = self.dolphins_mem[dolphin_idx].read_obs(0)
                raw_graphic_obs = self.graphic_obss[
                    dolphin_idx
                ].get()  # TODO give graphic obs correctly

                observation = {
                    "RACE_INFO": raw_vector_obs["RACE_INFO"],
                    "PLAYER_INFO": raw_vector_obs["PLAYER_INFO"][0],
                    "GRAPHIC_INFO": (
                        raw_graphic_obs[0],
                        raw_graphic_obs[1],
                        raw_graphic_obs[2],
                        raw_graphic_obs[3],
                    ),
                }
                observations[agent_id] = observation

        else:
            raw_vector_obs = self.dolphins_mem[0].read_obs(self.options.num_agents)
            _null = os.environ.get("KART_NULL_RENDER") == "1"
            raw_graphic_obs = None if _null else self.graphic_obss[0].get()
            # save_graphic_obs(raw_graphic_obs) # for DEBUG
            for agent_id in self.agents:
                observation = {
                    "RACE_INFO": raw_vector_obs["RACE_INFO"],
                    "PLAYER_INFO": raw_vector_obs["PLAYER_INFO"][agent_id],
                    "GRAPHIC_INFO": None if _null else (
                        raw_graphic_obs[0],
                        raw_graphic_obs[1],
                        raw_graphic_obs[2],
                        raw_graphic_obs[3 + agent_id],
                    ),
                }
                observations[agent_id] = observation

        return observations

    def _send_actions(self, actions: dict[AgentID, ActionType]):
        self.async_send_actions(actions)
        self.await_send_actions()

    def click(self, actions: dict[AgentID, ActionType], num_frame: int = 250):
        self._send_actions(actions)
        for _ in range(num_frame - 1):
            self._send_actions({})

    def async_send_actions(self, actions: dict[AgentID, ActionType]):
        new_actions = {
            idx: {} for idx in range(len(self.dolphins))
        }  # {dolphin_idx: {agent_id_within_dolphin: action}}
        for agent_id, action in actions.items():
            dolphin_idx, agent_id_within_dolphin = self.agent_mapper(agent_id)
            new_actions[dolphin_idx][agent_id_within_dolphin] = action

        for dolphin_idx, dolphin_actions in new_actions.items():
            self.dolphins[dolphin_idx].async_send_actions(dolphin_actions)

    def await_send_actions(self):
        for dolphin in self.dolphins:
            dolphin.await_send_actions()

    def await_step_done(self):
        for dolphin in self.dolphins:
            dolphin.await_send_actions()

    def async_save_file(self, path: str):
        for dolphin in self.dolphins:
            dolphin.async_save_file(path)

    def async_save_slot(self, slot: int):
        for dolphin in self.dolphins:
            dolphin.async_save_slot(slot)

    def await_save_done(self):
        for dolphin in self.dolphins:
            dolphin.await_save_done()

    def save_file(self, path: str):
        self.async_save_file(path)
        self.await_save_done()

    def save_slot(self, slot: int):
        self.async_save_slot(slot)
        self.await_save_done()

    def async_load_file(self, path: str):
        for dolphin in self.dolphins:
            dolphin.async_load_file(path)

    def async_load_slot(self, slot: int):
        for dolphin in self.dolphins:
            dolphin.async_load_slot(slot)

    def await_load_done(self):
        for dolphin in self.dolphins:
            dolphin.await_load_done()

    def load_file(self, path: str):
        self.async_load_file(path)
        self.await_load_done()

    def load_slot(self, slot: int):
        self.async_load_slot(slot)
        self.await_load_done()

    # fmt: off
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Space:  # type: ignore
        max_h, max_w = 480, 640
        image_space = spaces.Box(low=0, high=255, shape=(max_h, max_w, 4), dtype=np.uint8)

        return spaces.Dict({
            "RACE_INFO": spaces.Dict({
                "StageID": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "FrameCount": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "PlayerCount": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "CourseID": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "EngineClass": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
            }),
            "PLAYER_INFO": spaces.Dict({
                "PlayerID": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "LocalPlayerNum": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "RealControllerID": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "KartID": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "CharacterID": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),

                "CurrentRaceCompletion": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "MaxRaceCompletion": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "FirstKcpLapCompletion": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "NextCheckpointLapCompletion": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "NextCheckpointLapCompletionMax": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),

                "CurrentLap": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "MaxLap": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "currentKCP": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "maxKCP": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "StateBit": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),

                "SoftSpeedLimit": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "HardSpeedLimit": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),

                "Position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "Velocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "InternalVelocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "ExternalVelocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "AngularVelocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "Acceleration": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),

                "MainRotation": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),

                "Speed": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "AccelerationKartMove": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),

                "DriftState": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "MiniturboCharge": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "SMiniturboCharge": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "OffroadInvincibilityTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),

                "WheelieFrameCount": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "WheelieCooldownCount": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "LeanRot": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),

                "BitField0": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "BitField1": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "BitField2": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "BitField3": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "SurfaceFlag": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),

                "Hop": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),

                "MTBoostTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "AllMTCharge": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "MushroomBoostTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "TrickableTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "TrickCooldown": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "AirtimeCount": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),

                "RacePosition": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "FloorCollisionCount": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "RespawnTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "WallCollideFlag": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),

                "Item": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "ItemNum": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "PassiveItem": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
                "PassiveItemNum": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),

                "StarTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "ShockTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "BlooperInkTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "BlooperStateFlag": spaces.Box(0, np.iinfo(np.uint8).max, shape=(), dtype=np.uint8),
                "CrushTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),
                "MegaTimer": spaces.Box(0, np.iinfo(np.uint16).max, shape=(), dtype=np.uint16),

                "startBoostCharge": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "startBoostIdx": spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32),
            }),
            "GRAPHIC_INFO": spaces.Tuple((
                image_space,
                image_space,
                image_space,
                image_space
            ))
        })
    # fmt: on

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):  # type: ignore
        return spaces.Dict(
            {
                "A": spaces.Discrete(2),
                "B": spaces.Discrete(2),
                "X": spaces.Discrete(2),
                # "Y": spaces.Discrete(2),
                # "Z": spaces.Discrete(2),
                # "Start": spaces.Discrete(2),
                "Up": spaces.Discrete(2),
                "Down": spaces.Discrete(2),
                # "Left": spaces.Discrete(2),
                # "Right": spaces.Discrete(2),
                "L": spaces.Discrete(2),
                "R": spaces.Discrete(2),
                "StickX": spaces.Box(-1.0, 1.0, (), np.float32),
                # "StickY": spaces.Box(-1.0, 1.0, (), np.float32),
                # "CStickX": spaces.Box(-1.0, 1.0, (), np.float32),
                # "CStickY": spaces.Box(-1.0, 1.0, (), np.float32),
                # "TriggerLeft": spaces.Box(0.0, 1.0, (), np.float32),
                # "TriggerRight": spaces.Box(0.0, 1.0, (), np.float32),
            }
        )
