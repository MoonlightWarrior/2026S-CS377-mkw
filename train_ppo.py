from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from ppo_agent import PPOAgent
from ppo_buffer import PPOBuffer
from ppo_env import PPOEnv


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    num_envs = args.num_envs
    rollout_steps = args.rollout_steps
    total_updates = args.total_updates
    epochs = args.epochs
    minibatch_size = args.minibatch_size
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    learning_rate = args.learning_rate
    checkpoint_dir = args.checkpoint_dir
    load_checkpoint = args.load_checkpoint
    save_every = args.save_every
    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    env = PPOEnv(num_envs=num_envs)
    obs, _ = env.reset()
    obs_dim = obs.shape[1]
    action_dim = env.action_space.n
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.update(obs)
    obs = obs_rms.normalize(obs)

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    buffer = PPOBuffer(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_max_race_completion = np.zeros(args.num_envs, dtype=np.float32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    completed_race_completions: list[float] = []
    completed_successes = 0
    completed_timeouts = 0
    env_steps = 0
    start_time = time.time()
    start_update = 0

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(update: int) -> None:
        if checkpoint_dir is None:
            return
        torch.save(
            {
                "update": update,
                "env_steps": env_steps,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_rms_mean": obs_rms.mean,
                "obs_rms_var": obs_rms.var,
                "obs_rms_count": obs_rms.count,
            },
            os.path.join(checkpoint_dir, f"checkpoint_update_{update}.pt"),
        )

    if load_checkpoint is not None:
        # This checkpoint is produced by this project and includes optimizer and
        # numpy-based normalization stats, so it must be loaded as a full pickle.
        checkpoint = torch.load(load_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        start_update = int(checkpoint.get("update", 0))
        env_steps = int(checkpoint.get("env_steps", 0))

        if "obs_rms_mean" in checkpoint:
            obs_rms.mean = np.asarray(checkpoint["obs_rms_mean"], dtype=np.float64)
        if "obs_rms_var" in checkpoint:
            obs_rms.var = np.asarray(checkpoint["obs_rms_var"], dtype=np.float64)
        if "obs_rms_count" in checkpoint:
            obs_rms.count = float(checkpoint["obs_rms_count"])

        obs = obs_rms.normalize(obs)
        print(
            f"Loaded checkpoint: {load_checkpoint} "
            f"(resume_update={start_update}, env_steps={env_steps})"
        )

    if total_updates <= start_update:
        print(
            f"Nothing to do: total_updates={total_updates} is not greater than "
            f"checkpoint update={start_update}"
        )
        save_checkpoint(start_update)
        return

    for update in range(start_update + 1, total_updates + 1):
        buffer.reset()
        roll_returns: list[float] = []
        roll_lengths: list[int] = []
        roll_race_completions: list[float] = []
        roll_successes = 0
        roll_timeouts = 0
        action_counts = np.zeros(action_dim, dtype=np.int64)
        policy_entropies: list[float] = []

        for _ in range(rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(obs_tensor)
                _, entropy, _ = agent.evaluate_actions(obs_tensor, actions)

            actions_np = actions.cpu().numpy()
            action_counts += np.bincount(actions_np, minlength=action_dim)
            policy_entropies.append(float(entropy.mean().cpu().item()))
            next_obs, rewards, dones, truns, infos = env.step(actions_np)
            valid_mask = ~(np.asarray(infos["Ignore"], dtype=np.bool_) | np.asarray(infos["First"], dtype=np.bool_))
            stored_rewards = np.where(valid_mask, rewards, 0.0)
            stored_dones = np.where(valid_mask, np.logical_or(dones, truns), True)

            episode_returns[valid_mask] += rewards[valid_mask]
            episode_lengths[valid_mask] += 1
            race_completion_arr = infos.get("RaceCompletion")
            if race_completion_arr is not None:
                race_completion_arr = np.asarray(race_completion_arr, dtype=np.float32)
                episode_max_race_completion = np.maximum(
                    episode_max_race_completion, race_completion_arr
                )
            terminal = np.logical_or(dones, truns)
            for i in np.where(terminal & valid_mask)[0]:
                finished = bool(dones[i] and rewards[i] > 0.0)
                timed_out = bool(truns[i] or (dones[i] and rewards[i] <= 0.0))
                max_race_completion = float(episode_max_race_completion[i])

                roll_returns.append(float(episode_returns[i]))
                roll_lengths.append(int(episode_lengths[i]))
                roll_race_completions.append(max_race_completion)
                completed_returns.append(float(episode_returns[i]))
                completed_lengths.append(int(episode_lengths[i]))
                completed_race_completions.append(max_race_completion)
                roll_successes += int(finished)
                roll_timeouts += int(timed_out)
                completed_successes += int(finished)
                completed_timeouts += int(timed_out)

                episode_returns[i] = 0.0
                episode_lengths[i] = 0
                episode_max_race_completion[i] = 0.0

                print(
                    f"episode "
                    f"env={i} "
                    f"max_race_completion={max_race_completion:.3f} "
                    f"timeout={int(timed_out)} "
                    f"finish={int(finished)}"
                )

            buffer.add(
                obs=obs,
                actions=actions_np,
                rewards=stored_rewards,
                dones=stored_dones,
                log_probs=log_probs.cpu().numpy(),
                values=values.cpu().numpy(),
                valid_mask=valid_mask,
            )

            obs_rms.update(next_obs)
            obs = obs_rms.normalize(next_obs)
            env_steps += num_envs

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            last_values = agent.get_value(next_obs_tensor).cpu().numpy()

        buffer.compute_returns_and_advantages(last_values)

        advantages = buffer.advantages.reshape(-1)
        valid_rows = buffer.valid_mask.reshape(-1)
        if np.any(valid_rows):
            advantages = advantages[valid_rows]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            buffer.advantages[buffer.valid_mask] = advantages

        valid_sample_count = int(buffer.valid_mask.sum())

        for _ in range(epochs):
            for batch in buffer.get_batches(minibatch_size=minibatch_size):
                batch_obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
                batch_actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
                batch_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
                batch_advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
                batch_returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)

                new_log_probs, entropy, new_values = agent.evaluate_actions(batch_obs, batch_actions)
                log_ratio = new_log_probs - batch_log_probs
                ratio = log_ratio.exp()

                pg_loss_1 = -batch_advantages * ratio
                pg_loss_2 = -batch_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                value_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        rollout_episode_count = len(roll_returns)
        completion_rate = roll_successes / rollout_episode_count if rollout_episode_count else 0.0
        timeout_rate = roll_timeouts / rollout_episode_count if rollout_episode_count else 0.0
        avg_ep_len = float(np.mean(roll_lengths)) if roll_lengths else 0.0
        avg_ep_return = float(np.mean(roll_returns)) if roll_returns else 0.0
        mean_race_completion = float(np.mean(roll_race_completions)) if roll_race_completions else 0.0
        max_race_completion = float(np.max(roll_race_completions)) if roll_race_completions else 0.0
        mean_entropy = float(np.mean(policy_entropies)) if policy_entropies else 0.0
        action_probs = action_counts / max(action_counts.sum(), 1)
        top_actions = np.argsort(action_probs)[-3:][::-1]
        action_dist = ",".join(
            f"{int(action)}:{action_probs[action]:.3f}" for action in top_actions if action_probs[action] > 0.0
        )
        sps = int(env_steps / max(time.time() - start_time, 1e-6))

        print(
            f"update={update} "
            f"steps={env_steps} "
            f"episode_return={avg_ep_return:.3f} "
            f"race_completion_mean={mean_race_completion:.3f} "
            f"race_completion_max={max_race_completion:.3f} "
            f"completion_rate={completion_rate:.3f} "
            f"timeout_rate={timeout_rate:.3f} "
            f"avg_episode_length={avg_ep_len:.1f} "
            f"entropy={mean_entropy:.3f} "
            f"valid_samples={valid_sample_count} "
            f"action_dist={action_dist or 'n/a'} "
            f"sps={sps}"
        )

        if save_every > 0 and update % save_every == 0:
            save_checkpoint(update)

    save_checkpoint(total_updates)


if __name__ == "__main__":
    main()
