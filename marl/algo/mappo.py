"""
Phase 2 — MAPPO (Yu et al., 2022) with a centralized critic (CTDE).

Difference from the Vlab IPPO baseline: the critic is a *centralized* value
function V(s) that consumes the full global state (CentralizedTeamState, 102-d),
while the shared actor pi(a | o) consumes only the restricted per-agent view
(AsymmetricTeamObs, 59-d). This is the defining property of MAPPO/CTDE:
centralized training, decentralized execution.

Trajectory format consumed by `MAPPOLearner.update` (produced by train_marl.py):

    trajectory[agent_id] = [
        {
          "actor_obs":         np.ndarray (obs_size,)      # actor input
          "global_state":      np.ndarray (state_size,)    # critic input
          "next_global_state": np.ndarray (state_size,)
          "action":            int
          "reward":            float
          "done":              bool
        }, ...
    ]

A single shared actor controls all agents (parameter sharing) — matching the
proposal's "single shared policy controls all four players".
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


def _mlp(in_size: int, out_size: int, hidden: List[int]) -> nn.Sequential:
    layers: list = []
    prev = in_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


def _gae(rewards, values, next_values, dones, gamma, lam):
    T = len(rewards)
    adv = torch.zeros(T)
    g = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        g = delta + gamma * lam * (1.0 - dones[t]) * g
        adv[t] = g
    return adv


class SharedActor:
    """Stochastic shared actor over the restricted observation."""

    def __init__(self, obs_size: int, n_actions: int, hidden: List[int], device="cpu"):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = device
        self.net = _mlp(obs_size, n_actions, hidden).to(device)

    @torch.no_grad()
    def get_actions(self, actor_obs: Dict[int, np.ndarray]) -> Dict[int, int]:
        agents = list(actor_obs.keys())
        x = torch.as_tensor(np.stack([actor_obs[a] for a in agents]),
                            dtype=torch.float32, device=self.device)
        a = Categorical(logits=self.net(x)).sample().tolist()
        return {ag: int(a[i]) for i, ag in enumerate(agents)}

    def save(self, path: str | Path) -> None:
        torch.save({"actor": self.net.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        self.net.load_state_dict(ckpt.get("actor", ckpt))

    def clone_frozen(self) -> "SharedActor":
        """A detached copy for the self-play opponent pool (Phase 4)."""
        import copy
        c = SharedActor(self.obs_size, self.n_actions,
                        hidden=[], device=self.device)
        c.net = copy.deepcopy(self.net)
        for p in c.net.parameters():
            p.requires_grad_(False)
        return c


class MAPPOLearner:
    """Owns the centralized critic + both optimizers; updates the shared actor."""

    def __init__(
        self,
        actor: SharedActor,
        state_size: int,
        critic_hidden: List[int],
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 4096,
        max_grad_norm: float = 0.5,
        device="cpu",
    ):
        self.actor = actor
        self.device = device
        self.critic = _mlp(state_size, 1, critic_hidden).to(device)   # CENTRALIZED
        self.actor_optim = Adam(actor.net.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lam = gamma, lam
        self.clip_eps, self.ent_coef, self.vf_coef = clip_eps, ent_coef, vf_coef
        self.ppo_epochs, self.mini_batch_size = ppo_epochs, mini_batch_size
        self.max_grad_norm = max_grad_norm

    def update(self, trajectory: dict, agents: list) -> Dict[str, float]:
        obs, states, actions, adv, returns, old_lp = self._build(trajectory, agents)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = len(obs)
        a_loss = c_loss = ent = 0.0
        nb = 0
        for _ in range(self.ppo_epochs):
            for idx in self._batches(N):
                al, cl, e = self._step(obs[idx], states[idx], actions[idx],
                                       adv[idx], returns[idx], old_lp[idx])
                a_loss += al; c_loss += cl; ent += e; nb += 1
        d = max(nb, 1)
        return {"train/actor_loss": a_loss / d,
                "train/critic_loss": c_loss / d,
                "train/entropy": ent / d}

    def _build(self, trajectory, agents):
        O, S, A, ADV, RET, OLP = [], [], [], [], [], []
        for a in agents:
            traj = trajectory[a]
            if not traj:
                continue
            o   = torch.as_tensor(np.stack([t["actor_obs"] for t in traj]), dtype=torch.float32, device=self.device)
            s   = torch.as_tensor(np.stack([t["global_state"] for t in traj]), dtype=torch.float32, device=self.device)
            ns  = torch.as_tensor(np.stack([t["next_global_state"] for t in traj]), dtype=torch.float32, device=self.device)
            act = torch.as_tensor([t["action"] for t in traj], dtype=torch.long, device=self.device)
            rew = torch.as_tensor([t["reward"] for t in traj], dtype=torch.float32, device=self.device)
            dn  = torch.as_tensor([float(t["done"]) for t in traj], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                v   = self.critic(s).squeeze(-1)          # centralized value
                nv  = self.critic(ns).squeeze(-1)
                adv = _gae(rew, v, nv, dn, self.gamma, self.lam)
                ret = adv + v
                olp = Categorical(logits=self.actor.net(o)).log_prob(act)
            O.append(o); S.append(s); A.append(act); ADV.append(adv); RET.append(ret); OLP.append(olp)
        return (torch.cat(O), torch.cat(S), torch.cat(A),
                torch.cat(ADV), torch.cat(RET), torch.cat(OLP))

    def _batches(self, N):
        perm = torch.randperm(N, device=self.device)
        for s in range(0, N, self.mini_batch_size):
            yield perm[s:s + self.mini_batch_size]

    def _step(self, obs, states, actions, adv, returns, old_lp):
        dist = Categorical(logits=self.actor.net(obs))
        lp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratio = (lp - old_lp).exp()
        s1 = ratio * adv
        s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
        actor_loss = -torch.min(s1, s2).mean() - self.ent_coef * entropy

        critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        (self.vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({"actor": self.actor.net.state_dict(),
                    "critic": self.critic.state_dict(),
                    "actor_optim": self.actor_optim.state_dict(),
                    "critic_optim": self.critic_optim.state_dict()}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        self.actor.net.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
