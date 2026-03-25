from __future__ import annotations

import numpy as np


class PPOBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.valid_mask = np.zeros((rollout_steps, num_envs), dtype=np.bool_)

        self.advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise IndexError("PPOBuffer is full")

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones.astype(np.float32)
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.valid_mask[self.ptr] = valid_mask
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray) -> None:
        last_advantage = np.zeros(self.num_envs, dtype=np.float32)

        for t in range(self.rollout_steps - 1, -1, -1):
            if t == self.rollout_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * not_done - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * not_done * last_advantage
            self.advantages[t] = last_advantage

        self.returns = self.advantages + self.values

    def get_batches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ):
        total_steps = self.rollout_steps * self.num_envs

        obs = self.obs.reshape(total_steps, self.obs_dim)
        actions = self.actions.reshape(total_steps)
        log_probs = self.log_probs.reshape(total_steps)
        values = self.values.reshape(total_steps)
        advantages = self.advantages.reshape(total_steps)
        returns = self.returns.reshape(total_steps)
        valid_mask = self.valid_mask.reshape(total_steps)

        indices = np.flatnonzero(valid_mask)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), minibatch_size):
            batch_idx = indices[start:start + minibatch_size]
            yield {
                "obs": obs[batch_idx],
                "actions": actions[batch_idx],
                "log_probs": log_probs[batch_idx],
                "values": values[batch_idx],
                "advantages": advantages[batch_idx],
                "returns": returns[batch_idx],
            }
