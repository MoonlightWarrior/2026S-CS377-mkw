"""PPO policy and learner for Mario Kart Wii RL."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from .game_state import AgentID
from .policy import Policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp(in_size: int, out_size: int, hidden: List[int]) -> nn.Sequential:
    layers: list = []
    prev = in_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


def _gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GAE-λ advantage estimation over a single agent's sequential trajectory."""
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    return advantages


# ---------------------------------------------------------------------------
# Policy (actor)
# ---------------------------------------------------------------------------

class PPOPolicy(Policy):
    """
    Stochastic actor that satisfies the Policy interface.

    Pass to train.py in place of RandomPolicy:

        policy = PPOPolicy(obs_size, n_actions)
        learner = PPOLearner(policy, obs_size)
        ...
        trajectory, stats = collect_episode(..., policy=policy, ...)
        train_metrics = learner.update(trajectory, env.agents)
    """

    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        hidden: List[int] = [512, 512, 256],
    ):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.actor = _mlp(obs_size, n_actions, hidden)

    def get_actions(self, flat_obs: Dict[AgentID, np.ndarray]) -> Dict[AgentID, int]:
        agents = list(flat_obs.keys())
        obs_t = torch.tensor(
            np.stack([flat_obs[a] for a in agents]), dtype=torch.float32
        )
        with torch.no_grad():
            actions = Categorical(logits=self.actor(obs_t)).sample().tolist()
        return {a: actions[i] for i, a in enumerate(agents)}

    def save(self, path: str | Path) -> None:
        torch.save({"actor": self.actor.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, weights_only=True)
        # Support both full checkpoint (actor+critic) and actor-only saves.
        state = ckpt.get("actor", ckpt)
        self.actor.load_state_dict(state)


# ---------------------------------------------------------------------------
# Learner (critic + optimizers + PPO update)
# ---------------------------------------------------------------------------

class PPOLearner:
    """
    PPO learner. Owns the critic network and both optimizers.
    The actor lives inside ``policy`` so weights are shared with inference.

    Hyperparameters mirror config keys where applicable:
        actor_lr, critic_lr  ← policy_lr / critic_lr
        gamma, lam           ← gae_gamma / gae_lambda
        ppo_epochs           ← ppo_epochs
        mini_batch_size      ← ppo_batch_size (divided by n_agents)
        ent_coef             ← ppo_ent_coef
    """

    def __init__(
        self,
        policy: PPOPolicy,
        obs_size: int,
        hidden: List[int] = [512, 512, 256],
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        ppo_epochs: int = 2,
        mini_batch_size: int = 512,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.critic = _mlp(obs_size, 1, hidden)

        self.actor_optim = Adam(policy.actor.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, trajectory: dict, agents: list) -> Dict[str, float]:
        """
        Run a PPO update from a trajectory returned by collect_episode().

        GAE advantages are computed per-agent (preserving time order), then
        all agents' data is pooled for the mini-batch updates.

        Returns training metrics suitable for logging to W&B:
            train/actor_loss, train/critic_loss, train/entropy
        """
        obs_t, actions_t, advantages_t, returns_t, old_log_probs_t = \
            self._build_tensors(trajectory, agents)

        # Normalize advantages across the whole batch.
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        N = len(obs_t)
        total_actor_loss = total_critic_loss = total_entropy = 0.0
        n_batches = 0

        for _ in range(self.ppo_epochs):
            for idx in self._mini_batches(N):
                actor_loss, critic_loss, entropy = self._ppo_step(
                    obs_t[idx],
                    actions_t[idx],
                    advantages_t[idx],
                    returns_t[idx],
                    old_log_probs_t[idx],
                )
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy += entropy
                n_batches += 1

        denom = max(n_batches, 1)
        return {
            "train/actor_loss":  total_actor_loss / denom,
            "train/critic_loss": total_critic_loss / denom,
            "train/entropy":     total_entropy / denom,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full training state for resuming (actor + critic + optimizers)."""
        torch.save(
            {
                "actor":        self.policy.actor.state_dict(),
                "critic":       self.critic.state_dict(),
                "actor_optim":  self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, weights_only=True)
        self.policy.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_tensors(self, trajectory, agents):
        """Compute per-agent GAE, then concatenate across agents."""
        all_obs, all_actions, all_advantages, all_returns, all_old_lp = [], [], [], [], []

        for a in agents:
            traj = trajectory[a]
            obs      = torch.tensor(np.stack([t["obs"]      for t in traj]), dtype=torch.float32)
            actions  = torch.tensor([t["action"]             for t in traj], dtype=torch.long)
            rewards  = torch.tensor([t["reward"]             for t in traj], dtype=torch.float32)
            dones    = torch.tensor([t["done"]               for t in traj], dtype=torch.float32)
            next_obs = torch.tensor(np.stack([t["next_obs"] for t in traj]), dtype=torch.float32)

            with torch.no_grad():
                values      = self.critic(obs).squeeze(-1)
                next_values = self.critic(next_obs).squeeze(-1)
                advantages  = _gae(rewards, values, next_values, dones, self.gamma, self.lam)
                returns     = advantages + values
                old_lp      = Categorical(logits=self.policy.actor(obs)).log_prob(actions)

            all_obs.append(obs)
            all_actions.append(actions)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_old_lp.append(old_lp)

        return (
            torch.cat(all_obs),
            torch.cat(all_actions),
            torch.cat(all_advantages),
            torch.cat(all_returns),
            torch.cat(all_old_lp),
        )

    def _mini_batches(self, N: int):
        perm = torch.randperm(N)
        for start in range(0, N, self.mini_batch_size):
            yield perm[start : start + self.mini_batch_size]

    def _ppo_step(self, obs, actions, advantages, returns, old_log_probs):
        dist = Categorical(logits=self.policy.actor(obs))
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

        critic_loss = F.mse_loss(self.critic(obs).squeeze(-1), returns)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        (self.vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()
