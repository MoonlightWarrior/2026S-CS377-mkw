"""Top-down trajectory visualization for CPU AI BC data.

Reads an NPZ produced by `collect_distill_data.py` and renders:
  * Luigi Circuit track outline (CKPT left/right segments)
  * ENPT polyline (CPU AI ideal path)
  * Per-slot kart trajectories color-coded
  * Start (circle) / end (X) markers per slot

Optional `--slot N` adds a side panel with stickX, button raster, and
RaceCompletion timeseries for that slot.

Optional `--animate` produces an MP4 with fading trails.

Usage:
    uv run python viz_cpu_movement.py \
        --npz instance_info/distill_itembit.npz \
        [--course ~/Statistical-Physics-ing-Kinoko/output_dir/beginner_course.json] \
        [--slot 4] [--animate] [--out viz.png]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

POS_X_IDX = 16
POS_Z_IDX = 18

BUTTON_ACCEL = 1 << 0
BUTTON_BRAKE = 1 << 1
BUTTON_ITEM  = 1 << 2
BUTTON_DRIFT = 1 << 3

DEFAULT_COURSE = Path.home() / "Statistical-Physics-ing-Kinoko/output_dir/beginner_course.json"
DEFAULT_INSTANCE_INFO = Path(__file__).resolve().parent / "instance_info"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, default=None,
                   help="NPZ from collect_distill_data.py. Defaults to largest in instance_info/.")
    p.add_argument("--course", type=Path, default=DEFAULT_COURSE,
                   help="Course JSON (Kinoko-style, with checkpoints + enemy_points).")
    p.add_argument("--slot", type=int, default=None,
                   help="If set, add an action-stream side panel for this slot.")
    p.add_argument("--animate", action="store_true",
                   help="Render an MP4/GIF instead of (or in addition to) the static PNG.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output file. Default: viz_<npz_stem>.png (or .mp4 for --animate).")
    p.add_argument("--motion_threshold", type=float, default=50.0,
                   help="Skip slot if max(|dx|)+max(|dz|) < this (TYPE_NONE detection).")
    return p.parse_args()


def default_npz() -> Path:
    candidates = list(DEFAULT_INSTANCE_INFO.glob("distill_*.npz"))
    if not candidates:
        raise SystemExit(f"No NPZ found in {DEFAULT_INSTANCE_INFO}")
    return max(candidates, key=lambda p: p.stat().st_size)


def load_course(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Course JSON not found: {path}")
    with path.open() as f:
        return json.load(f)


def plot_track(ax, course: dict) -> None:
    """Draw CKPT left/right edges + ENPT polyline."""
    cps = course.get("checkpoints", [])
    if cps:
        left  = np.array([[c["left_x"],  c["left_z"]]  for c in cps])
        right = np.array([[c["right_x"], c["right_z"]] for c in cps])

        # Close the loop (LC is a closed ring).
        def close(a: np.ndarray) -> np.ndarray:
            return np.vstack([a, a[:1]])

        ax.plot(close(left)[:, 0],  close(left)[:, 1],  color="#888", lw=0.9, label="CKPT left")
        ax.plot(close(right)[:, 0], close(right)[:, 1], color="#888", lw=0.9, label="CKPT right")
        segs = np.stack([left, right], axis=1)
        ax.add_collection(LineCollection(segs, colors="#ccc", linewidths=0.6, zorder=0))

    enpts = course.get("enemy_points", [])
    if enpts:
        ex = np.array([p["pos_x"] for p in enpts])
        ez = np.array([p["pos_z"] for p in enpts])
        ex_c = np.append(ex, ex[0])
        ez_c = np.append(ez, ez[0])
        ax.plot(ex_c, ez_c, color="#4477aa", lw=1.0, linestyle="--",
                alpha=0.7, label="ENPT (CPU ideal)")

    ax.set_aspect("equal")
    ax.invert_yaxis()  # +Z down so layout matches in-game minimap
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.2)


def slot_traces(d: dict) -> dict[int, dict]:
    """Group long-format NPZ rows by slot. Returns {slot: {'step', 'x', 'z', ...}}."""
    out: dict[int, dict] = {}
    slot_arr = d["slot"]
    obs = d["obs"]
    step = d["step"]
    stickX = d["stickX"]
    buttons = d["buttons"]
    action = d["action"]
    rc = d["rc"]
    for s in sorted(np.unique(slot_arr).tolist()):
        mask = slot_arr == s
        # rows are appended in (step, slot) order — sort by step to be safe
        idx = np.where(mask)[0]
        idx = idx[np.argsort(step[idx])]
        out[int(s)] = {
            "step":    step[idx],
            "x":       obs[idx, POS_X_IDX],
            "z":       obs[idx, POS_Z_IDX],
            "stickX":  stickX[idx],
            "buttons": buttons[idx],
            "action":  action[idx],
            "rc":      rc[idx],
        }
    return out


def is_moving(trace: dict, threshold: float) -> bool:
    if len(trace["x"]) < 2:
        return False
    dx = float(np.max(trace["x"]) - np.min(trace["x"]))
    dz = float(np.max(trace["z"]) - np.min(trace["z"]))
    return (dx + dz) > threshold


def render_static(course: dict, traces: dict, args: argparse.Namespace) -> Path:
    has_panel = args.slot is not None
    if has_panel and args.slot not in traces:
        raise SystemExit(f"--slot {args.slot} not in NPZ; available: {sorted(traces)}")

    if has_panel:
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        gs = fig.add_gridspec(3, 5)
        ax_map = fig.add_subplot(gs[:, :3])
        ax_stick = fig.add_subplot(gs[0, 3:])
        ax_btn   = fig.add_subplot(gs[1, 3:])
        ax_rc    = fig.add_subplot(gs[2, 3:])
    else:
        fig, ax_map = plt.subplots(figsize=(12, 10), constrained_layout=True)

    plot_track(ax_map, course)

    cmap = plt.get_cmap("tab20")
    slot_summaries = []
    for color_i, (s, tr) in enumerate(traces.items()):
        if not is_moving(tr, args.motion_threshold):
            slot_summaries.append((s, "(stationary)"))
            continue
        color = cmap(color_i % cmap.N)
        ax_map.plot(tr["x"], tr["z"], color=color, lw=1.2, alpha=0.85,
                    label=f"slot {s}")
        ax_map.plot(tr["x"][0],  tr["z"][0],  marker="o", color=color, ms=8,
                    markeredgecolor="white", zorder=5)
        ax_map.plot(tr["x"][-1], tr["z"][-1], marker="X", color=color, ms=10,
                    markeredgecolor="white", zorder=5)
        slot_summaries.append((s, f"rc max={float(tr['rc'].max()):.2f}"))

    title_lines = [
        f"NPZ: {args.npz.name}  |  rows: {sum(len(tr['step']) for tr in traces.values())}",
        "  ".join(f"slot{s}: {info}" for s, info in slot_summaries),
    ]
    ax_map.set_title("\n".join(title_lines), fontsize=10)
    ax_map.legend(loc="upper right", fontsize=8, ncols=2)

    if has_panel:
        tr = traces[args.slot]
        t = tr["step"]
        # stickX
        ax_stick.plot(t, tr["stickX"], color="#222", lw=0.8)
        ax_stick.axhline(0, color="#999", lw=0.5)
        ax_stick.set_ylabel("stickX")
        ax_stick.set_ylim(-1.1, 1.1)
        ax_stick.set_title(f"slot {args.slot} — action stream", fontsize=10)
        ax_stick.grid(True, alpha=0.2)

        # button raster: 4 horizontal bars
        btn = tr["buttons"].astype(np.int64)
        bit_names = ["accel", "brake", "item", "drift"]
        bit_masks = [BUTTON_ACCEL, BUTTON_BRAKE, BUTTON_ITEM, BUTTON_DRIFT]
        for i, (name, mask) in enumerate(zip(bit_names, bit_masks)):
            on = (btn & mask) != 0
            ax_btn.fill_between(t, i + 0.05, i + 0.95, where=on, step="post",
                                color=f"C{i}", alpha=0.6)
        ax_btn.set_yticks([i + 0.5 for i in range(4)])
        ax_btn.set_yticklabels(bit_names)
        ax_btn.set_ylim(-0.1, 4.1)
        ax_btn.set_ylabel("buttons")
        ax_btn.grid(True, alpha=0.2)

        # RaceCompletion
        ax_rc.plot(t, tr["rc"], color="#225522", lw=1.0)
        ax_rc.set_ylabel("RaceCompletion")
        ax_rc.set_xlabel("master step")
        ax_rc.grid(True, alpha=0.2)

    out = args.out
    if out is None:
        stem = args.npz.stem
        suffix = f"_slot{args.slot}" if has_panel else ""
        # Prefer NPZ dir; fall back to script dir if read-only (instance_info/
        # may be docker-root-owned).
        candidate = args.npz.parent / f"viz_{stem}{suffix}.png"
        out = candidate if os.access(args.npz.parent, os.W_OK) else \
              Path(__file__).resolve().parent / f"viz_{stem}{suffix}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def render_animation(course: dict, traces: dict, args: argparse.Namespace) -> Path:
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    moving = {s: tr for s, tr in traces.items()
              if is_moving(tr, args.motion_threshold)}
    if not moving:
        raise SystemExit("No moving slots to animate.")

    # Align timelines on master step. Each slot's `step` may have gaps (e.g.
    # perturbation rows dropped). Build a per-slot lookup, then render at the
    # union of steps.
    all_steps = sorted(set().union(*(set(tr["step"].tolist()) for tr in moving.values())))
    lookup = {s: {int(st): (float(x), float(z))
                   for st, x, z in zip(tr["step"], tr["x"], tr["z"])}
              for s, tr in moving.items()}

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    plot_track(ax, course)

    cmap = plt.get_cmap("tab20")
    colors = {s: cmap(i % cmap.N) for i, s in enumerate(moving)}

    # One LineCollection per slot for the fading trail; we update its segments per frame.
    trails = {s: LineCollection([], colors=[colors[s]] * 30, linewidths=1.4)
              for s in moving}
    dots = {s: ax.plot([], [], marker="o", color=colors[s], ms=6,
                       markeredgecolor="white", zorder=5, label=f"slot {s}")[0]
            for s in moving}
    for tc in trails.values():
        ax.add_collection(tc)

    ax.legend(loc="upper right", fontsize=8, ncols=2)
    title = ax.set_title("")

    TRAIL = 30

    def init():
        return list(dots.values()) + list(trails.values()) + [title]

    def update(frame_i: int):
        step_now = all_steps[frame_i]
        title.set_text(f"step {step_now}  ({frame_i + 1}/{len(all_steps)})")
        for s in moving:
            # Build fading trail from steps in (step_now - TRAIL, step_now]
            recent = [lookup[s].get(st) for st in all_steps[max(0, frame_i - TRAIL):frame_i + 1]]
            recent = [p for p in recent if p is not None]
            if not recent:
                dots[s].set_data([], [])
                trails[s].set_segments([])
                continue
            xs, zs = zip(*recent)
            dots[s].set_data([xs[-1]], [zs[-1]])
            if len(recent) >= 2:
                segs = [[(xs[k], zs[k]), (xs[k + 1], zs[k + 1])] for k in range(len(recent) - 1)]
                # alpha fade from old (transparent) to recent (opaque)
                alphas = np.linspace(0.1, 0.9, len(segs))
                trails[s].set_segments(segs)
                trails[s].set_alpha(None)
                trails[s].set_color([(*colors[s][:3], a) for a in alphas])
            else:
                trails[s].set_segments([])
        return list(dots.values()) + list(trails.values()) + [title]

    anim = FuncAnimation(fig, update, init_func=init, frames=len(all_steps),
                         blit=False, interval=50)

    out = args.out
    if out is None:
        candidate = args.npz.parent / f"viz_{args.npz.stem}.mp4"
        out = candidate if os.access(args.npz.parent, os.W_OK) else \
              Path(__file__).resolve().parent / f"viz_{args.npz.stem}.mp4"

    try:
        writer = FFMpegWriter(fps=20, bitrate=2000)
        anim.save(str(out), writer=writer, dpi=120)
    except Exception as exc:
        print(f"[anim] FFMpegWriter failed ({exc}); falling back to GIF (PillowWriter).")
        out = out.with_suffix(".gif")
        anim.save(str(out), writer=PillowWriter(fps=20), dpi=100)

    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    if args.npz is None:
        args.npz = default_npz()
    if not args.npz.exists():
        raise SystemExit(f"NPZ not found: {args.npz}")

    d = np.load(args.npz)
    course = load_course(args.course)
    traces = slot_traces(d)

    print(f"[viz] NPZ {args.npz.name}  slots: {sorted(traces)}  "
          f"rows: {sum(len(tr['step']) for tr in traces.values())}")

    if args.animate:
        out = render_animation(course, traces, args)
    else:
        out = render_static(course, traces, args)
    print(f"[viz] saved → {out}")


if __name__ == "__main__":
    main()
