from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 40,
        hidden_sizes: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        self.actor = self._build_mlp(obs_dim, hidden_sizes, action_dim)
        self.critic = self._build_mlp(obs_dim, hidden_sizes, 1)

    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_dist(self, obs: torch.Tensor) -> Categorical:
        logits = self.get_logits(obs)
        return Categorical(logits=logits)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(obs)

        return action, log_prob, entropy, value

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(obs)

        return log_prob, entropy, value
