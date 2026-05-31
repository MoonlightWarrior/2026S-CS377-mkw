"""
Pull a W&B run's history into a LaTeX-friendly CSV (+ optional plots) for the
report. Per-episode win is 0/1, so we also emit rolling-mean columns that show
the trend (win-rate vs.\ snapshots, smoothed reward/completion).

Recommended report workflow: CSV -> pgfplots/tikzplotlib for vector figures.

Run (inside the container, WANDB_API_KEY is set):
  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/analysis/fetch_wandb.py \
      --run checkcheckcheck/mario-kart-rl/4t0ck9gy --window 20
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

KEYS = [
    "train/timesteps", "train/iteration",
    "rollout/win_vs_snapshot", "rollout/mean_episode_reward",
    "rollout/mean_race_completion", "rollout/finish_rate",
    "rollout/mean_speed", "rollout/best_race_position",
    "train/actor_loss", "train/critic_loss", "train/entropy",
    "selfplay/pool_size",
]


def _rolling(xs, w):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="entity/project/run_id")
    ap.add_argument("--window", type=int, default=20, help="rolling-average window")
    ap.add_argument("--out", default=None, help="output dir (default: results/analysis/<run_id>)")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    run = api.run(args.run)
    rows = [{k: r.get(k) for k in KEYS} for r in run.scan_history(keys=KEYS)]
    rows = [r for r in rows if r.get("train/timesteps") is not None]
    rows.sort(key=lambda r: r["train/timesteps"])
    print(f"[wandb] fetched {len(rows)} logged steps from {args.run}")
    if not rows:
        print("[wandb] no rows yet — run may still be initializing.")
        return

    # rolling trends for the 0/1 win signal and the noisy reward/completion
    ts = [r["train/timesteps"] for r in rows]
    for src, dst in [("rollout/win_vs_snapshot", "win_rate_roll"),
                     ("rollout/mean_episode_reward", "reward_roll"),
                     ("rollout/mean_race_completion", "completion_roll")]:
        vals = [r.get(src) or 0.0 for r in rows]
        roll = _rolling(vals, args.window)
        for r, v in zip(rows, roll):
            r[dst] = v

    out_dir = Path(args.out) if args.out else Path("/cs377/marl/results/analysis") / args.run.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = KEYS + ["win_rate_roll", "reward_roll", "completion_roll"]
    with open(out_dir / "wandb_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"[wandb] wrote {out_dir}/wandb_history.csv ({len(rows)} rows)")

    if HAVE_MPL:
        def _plot(y, ylabel, fname, color="C0"):
            plt.figure(figsize=(5, 3.2))
            plt.plot(ts, [r.get(y) for r in rows], color=color)
            plt.xlabel("timesteps"); plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=150); plt.close()
        _plot("win_rate_roll", f"win rate vs snapshot ({args.window}-ep avg)", "win_rate.png", "C2")
        _plot("reward_roll", f"mean episode reward ({args.window}-ep avg)", "reward.png", "C0")
        _plot("completion_roll", f"race completion ({args.window}-ep avg)", "completion.png", "C1")
        print(f"[wandb] wrote win_rate.png, reward.png, completion.png")

    last = rows[-1]
    print(f"[wandb] latest: ts={last['train/timesteps']:,}  "
          f"win_rate(roll)={last.get('win_rate_roll'):.2f}  "
          f"reward(roll)={last.get('reward_roll'):.2f}  "
          f"completion(roll)={last.get('completion_roll'):.2f}")


if __name__ == "__main__":
    main()
