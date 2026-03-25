from __future__ import annotations

import gymnasium as gym
import numpy as np

from DolphinEnv import DolphinEnv


class PPOEnv:
    def __init__(
        self,
        num_envs: int,
        gamename: str = "LC",
        gamefile: str = "mkw.iso",
        project_folder=None,
        games_folder=None,
    ) -> None:
        self.env = DolphinEnv(
            num_envs=num_envs,
            gamename=gamename,
            gamefile=gamefile,
            project_folder=project_folder,
            games_folder=games_folder,
        )
        self.num_envs = num_envs
        self.single_action_space = self.env.action_space[0]
        self.action_space = self.single_action_space

        flat_obs_size = int(np.prod(self.env.observation_space.shape))
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_obs_size,),
            dtype=np.float32,
        )
        self.observation_space = self.single_observation_space

    def _flatten_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs.reshape(obs.shape[0], -1).astype(np.float32, copy=False)

    def reset(self):
        obs, infos = self.env.reset()
        return self._flatten_obs(obs), infos

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        self.env.step_async(actions)
        obs, rewards, dones, truns, infos = self.env.step_wait()

        episode_end = np.logical_or(dones, truns)
        infos = dict(infos)
        infos["episode_end"] = episode_end

        return self._flatten_obs(obs), rewards, dones, truns, infos
