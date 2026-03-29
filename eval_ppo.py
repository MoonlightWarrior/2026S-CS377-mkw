from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from ppo_agent import PPOAgent
from ppo_env import PPOEnv


@dataclass
class ObsNormalizer:
    mean: np.ndarray
    var: np.ndarray

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, obs_dim: int) -> "ObsNormalizer":
        mean = np.asarray(
            checkpoint.get("obs_rms_mean", np.zeros(obs_dim, dtype=np.float64)),
            dtype=np.float64,
        )
        var = np.asarray(
            checkpoint.get("obs_rms_var", np.ones(obs_dim, dtype=np.float64)),
            dtype=np.float64,
        )
        return cls(mean=mean, var=np.maximum(var, 1e-6))

    def normalize(self, obs: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32, copy=False)


def checkpoint_progress_metrics(
    start_race_completion: float,
    max_race_completion: float,
) -> tuple[float, float, float]:
    start_checkpoint = min(4.0, np.ceil(max(start_race_completion, 1.0) * 10.0) / 10.0)
    relative_race_completion = max(0.0, max_race_completion - start_checkpoint)
    completion_percent = min(100.0, 100.0 * relative_race_completion / 3.0)
    return start_checkpoint, relative_race_completion, completion_percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PPO checkpoint on the Dolphin environment."
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint saved by train_ppo.py",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reset_mode", type=str, default="race_start")
    parser.add_argument(
        "--reset_savestate",
        type=str,
        default=None,
        help="Optional savestate path to use as the fixed race-start checkpoint.",
    )
    parser.add_argument(
        "--episode_timeout_steps",
        type=int,
        default=None,
        help="Optional episode timeout in steps. Default is no timeout for race_start.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of taking argmax actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)

    env = PPOEnv(
        num_envs=args.num_envs,
        reset_mode=args.reset_mode,
        reset_savestate=args.reset_savestate,
        episode_timeout_steps=args.episode_timeout_steps,
    )
    obs, _ = env.reset()
    obs_dim = obs.shape[1]
    action_dim = env.action_space.n

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    obs_normalizer = ObsNormalizer.from_checkpoint(checkpoint, obs_dim)
    obs = obs_normalizer.normalize(obs)

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episode_start_race_completion = np.full(args.num_envs, np.nan, dtype=np.float32)
    episode_max_race_completion = np.zeros(args.num_envs, dtype=np.float32)

    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    completed_max_race_completion: list[float] = []
    completed_relative_race_completion: list[float] = []
    completed_completion_percent: list[float] = []
    completed_successes = 0
    completed_timeouts = 0

    while len(completed_returns) < args.episodes:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            if args.stochastic:
                actions, _, _, _ = agent.get_action_and_value(obs_tensor)
            else:
                logits = agent.get_logits(obs_tensor)
                actions = torch.argmax(logits, dim=-1)

        next_obs, rewards, dones, truns, infos = env.step(actions.cpu().numpy())
        next_obs = obs_normalizer.normalize(next_obs)

        valid_mask = ~(
            np.asarray(infos["Ignore"], dtype=np.bool_)
            | np.asarray(infos["First"], dtype=np.bool_)
        )
        episode_returns[valid_mask] += rewards[valid_mask]
        episode_lengths[valid_mask] += 1

        race_completion = np.asarray(
            infos.get("RaceCompletion", np.zeros(args.num_envs, dtype=np.float32)),
            dtype=np.float32,
        )
        new_episode_mask = valid_mask & np.isnan(episode_start_race_completion)
        episode_start_race_completion[new_episode_mask] = race_completion[new_episode_mask]
        episode_max_race_completion = np.maximum(
            episode_max_race_completion, race_completion
        )

        terminal = np.logical_or(dones, truns)
        for i in np.where(terminal & valid_mask)[0]:
            finished = bool(dones[i] and rewards[i] > 0.0)
            timed_out = bool(truns[i] or (dones[i] and rewards[i] <= 0.0))
            start_race_completion = float(
                1.0 if np.isnan(episode_start_race_completion[i]) else episode_start_race_completion[i]
            )
            max_race_completion = float(episode_max_race_completion[i])
            start_checkpoint, relative_race_completion, completion_percent = checkpoint_progress_metrics(
                start_race_completion,
                max_race_completion,
            )

            completed_returns.append(float(episode_returns[i]))
            completed_lengths.append(int(episode_lengths[i]))
            completed_max_race_completion.append(max_race_completion)
            completed_relative_race_completion.append(relative_race_completion)
            completed_completion_percent.append(completion_percent)
            completed_successes += int(finished)
            completed_timeouts += int(timed_out)

            print(
                f"episode={len(completed_returns)} "
                f"env={i} "
                f"return={episode_returns[i]:.3f} "
                f"length={episode_lengths[i]} "
                f"start_checkpoint={start_checkpoint:.3f} "
                f"relative_race_completion={relative_race_completion:.3f} "
                f"completion_percent={completion_percent:.1f} "
                f"finish={int(finished)} "
                f"timeout={int(timed_out)}"
            )

            episode_returns[i] = 0.0
            episode_lengths[i] = 0
            episode_start_race_completion[i] = np.nan
            episode_max_race_completion[i] = 0.0

            if len(completed_returns) >= args.episodes:
                break

        obs = next_obs

    total_episodes = len(completed_returns)
    mean_return = float(np.mean(completed_returns)) if completed_returns else 0.0
    mean_length = float(np.mean(completed_lengths)) if completed_lengths else 0.0
    mean_race_completion = (
        float(np.mean(completed_max_race_completion))
        if completed_max_race_completion
        else 0.0
    )
    mean_relative_race_completion = (
        float(np.mean(completed_relative_race_completion))
        if completed_relative_race_completion
        else 0.0
    )
    mean_completion_percent = (
        float(np.mean(completed_completion_percent))
        if completed_completion_percent
        else 0.0
    )
    completion_rate = completed_successes / total_episodes if total_episodes else 0.0
    timeout_rate = completed_timeouts / total_episodes if total_episodes else 0.0

    print("\nEvaluation summary")
    print(f"episodes={total_episodes}")
    print(f"mean_return={mean_return:.3f}")
    print(f"mean_length={mean_length:.1f}")
    print(f"mean_max_race_completion={mean_race_completion:.3f}")
    print(f"mean_relative_completion={mean_relative_race_completion:.3f}")
    print(f"mean_completion_percent={mean_completion_percent:.1f}")
    print(f"completion_rate={completion_rate:.3f}")
    print(f"timeout_rate={timeout_rate:.3f}")


if __name__ == "__main__":
    main()
