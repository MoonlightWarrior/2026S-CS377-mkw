"""Curses TUI for scrubbing through a BC NPZ frame-by-frame.

Lets you inspect every obs column for every slot, frame by frame, to
debug the CPU AI data (or BC labels) when the trajectory viz looks off.

Two views:
  * Single-slot (default): all 78 obs fields grouped into named panels.
  * All-slots (TAB): 12-row table, key fields only.

Navigation:
  ← / →            previous / next master step
  Shift+← / Shift+→ jump 10 steps
  ↑ / ↓            cycle active slot
  TAB              toggle single ↔ all-slots view
  g                prompt for frame number then jump
  q                quit

Usage:
    uv run python inspect_cpu_tui.py \
        --npz instance_info/distill_itembit.npz \
        [--slot 4] [--frame 0]
"""
from __future__ import annotations

import argparse
import curses
from pathlib import Path

import numpy as np

POS_X_IDX = 16
POS_Z_IDX = 18

# Field labels for the 78-dim per-kart obs slice. Mirrors the layout in
# DolphinScript.py _lite_per_kart_obs (line 342+); kept as a local table to
# keep this script standalone.
FIELD_LABELS = [
    "PlayerID",          # 0
    "LocalPlayerNum",    # 1
    "RealControllerID",  # 2
    "KartID",            # 3
    "CharacterID",       # 4
    "RaceCompletion",    # 5
    "MaxRaceCompletion", # 6
    "FirstKcpLapCompl",  # 7
    "NextChkptLapCompl", # 8
    "NextChkptLapCompMx",# 9
    "currentLap",        # 10
    "MaxLap",            # 11
    "currentKCP",        # 12
    "maxKCP",            # 13
    "SoftSpeedLimit",    # 14
    "HardSpeedLimit",    # 15
    "pos_x",             # 16
    "pos_y",             # 17
    "pos_z",             # 18
    "vel_x",             # 19
    "vel_y",             # 20
    "vel_z",             # 21
    "intVel_x",          # 22
    "intVel_y",          # 23
    "intVel_z",          # 24
    "extVel_x",          # 25
    "extVel_y",          # 26
    "extVel_z",          # 27
    "angVel_x",          # 28
    "angVel_y",          # 29
    "angVel_z",          # 30
    "accel_x",           # 31
    "accel_y",           # 32
    "accel_z",           # 33
    "mainRot_x",         # 34
    "mainRot_y",         # 35
    "mainRot_w",         # 36  (WiiRL stores w before z!)
    "mainRot_z",         # 37
    "speed",             # 38
    "accel_KartMove",    # 39
    "DriftState",        # 40
    "miniturboCharge",   # 41
    "SMiniturboCharge",  # 42
    "offroadInvinc",     # 43
    "wheelieFrames",     # 44
    "wheelieCooldown",   # 45
    "leanRot",           # 46
    "BitField0",         # 47
    "BitField1",         # 48
    "BitField2",         # 49
    "BitField3",         # 50
    "surfaceFlags",      # 51
    "HopVector_x",       # 52
    "HopVector_y",       # 53
    "HopVector_z",       # 54
    "mt_boost_timer",    # 55
    "allmt",             # 56
    "mush_and_boost",    # 57
    "trickableTimer",    # 58
    "trick_cooldown",    # 59
    "airtime",           # 60
    "race_position",     # 61
    "floor_collision",   # 62
    "respawn_timer",     # 63
    "wall_collide",      # 64
    "Item",              # 65
    "ItemNum",           # 66
    "PassiveItem",       # 67
    "PassiveItemNum",    # 68
    "StarTimer",         # 69
    "ShockTimer",        # 70
    "BlooperInkTimer",   # 71
    "BlooperStateFlag",  # 72
    "CrushTimer",        # 73
    "MegaTimer",         # 74
    "StateBit",          # 75
    "startBoostCharge",  # 76
    "startBoostIdx",     # 77
]
assert len(FIELD_LABELS) == 78

# Logical groupings for the single-slot panel layout.
GROUPS = [
    ("Race",       list(range(0, 14))),
    ("Limits",     list(range(14, 16))),
    ("Physics",    list(range(16, 38))),
    ("KartMove",   list(range(38, 47))),
    ("BitFields",  list(range(47, 51))),
    ("Collide",    [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]),
    ("Items",      [65, 66, 67, 68]),
    ("Timers",     [69, 70, 71, 72, 73, 74]),
    ("Misc",       [75, 76, 77]),
]

BUTTON_BITS = [
    (1 << 0, "accel"),
    (1 << 1, "brake"),
    (1 << 2, "item"),
    (1 << 3, "drift"),
]

BITFIELD_ANNOTATIONS = {
    # field_idx → list of (bit_in_word_msb_from_0, label) annotations
    50: [(3, "wallCol")],
    49: [(31 - 31, "wheelie")],  # bit 31 LSB
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--slot", type=int, default=None,
                   help="Initial active slot (default: first slot in NPZ).")
    p.add_argument("--frame", type=int, default=0)
    return p.parse_args()


def load_npz(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"NPZ not found: {path}")
    d = np.load(path)
    return {k: d[k] for k in d.keys()}


def index_by_step_slot(d: dict) -> dict:
    """Build {step → {slot → row_idx}}.

    Rows in the NPZ are long-format (one row per (step, slot)). For the TUI
    we want O(1) lookup by (step, slot).
    """
    steps = d["step"]
    slots = d["slot"]
    index: dict[int, dict[int, int]] = {}
    for i, (s, k) in enumerate(zip(steps.tolist(), slots.tolist())):
        index.setdefault(int(s), {})[int(k)] = i
    return index


def format_value(field_idx: int, val) -> str:
    name = FIELD_LABELS[field_idx]
    if name.startswith("BitField") or name == "surfaceFlags":
        v = int(val) & 0xFFFFFFFF
        return f"0x{v:08x}"
    if name in {"wall_collide", "Item", "ItemNum", "PassiveItem", "PassiveItemNum",
                "StateBit", "currentKCP", "maxKCP", "currentLap", "MaxLap",
                "race_position", "DriftState", "miniturboCharge", "SMiniturboCharge",
                "offroadInvinc", "wheelieFrames", "wheelieCooldown", "mt_boost_timer",
                "allmt", "mush_and_boost", "trickableTimer", "trick_cooldown",
                "airtime", "floor_collision", "respawn_timer", "StarTimer",
                "ShockTimer", "BlooperInkTimer", "BlooperStateFlag", "CrushTimer",
                "MegaTimer", "startBoostIdx", "PlayerID", "LocalPlayerNum",
                "RealControllerID", "KartID", "CharacterID"}:
        return f"{int(val):>10d}"
    return f"{float(val):>10.3f}"


def render_single(stdscr, state) -> None:
    h, w = stdscr.getmaxyx()
    d = state["d"]
    step = state["steps_sorted"][state["frame_i"]]
    slot = state["slot"]
    row_idx = state["index"].get(step, {}).get(slot)

    prev_step = state["steps_sorted"][state["frame_i"] - 1] if state["frame_i"] > 0 else None
    prev_row = state["index"].get(prev_step, {}).get(slot) if prev_step is not None else None

    stdscr.erase()

    # Header
    npz_name = state["args"].npz.name
    n_steps = len(state["steps_sorted"])
    race_info = d["race_info"][row_idx] if row_idx is not None else np.zeros(5)
    header = (f" NPZ: {npz_name}  frame {state['frame_i']+1}/{n_steps} (step {step})  "
              f"race_stage={int(race_info[0])}  FrameCount={int(race_info[1])}  "
              f"slot={slot}")
    stdscr.addnstr(0, 0, header.ljust(w-1), w-1, curses.A_REVERSE)

    available_slots = sorted({s for ss in state["index"].values() for s in ss})
    helpline = (f" slots in NPZ: {available_slots}   "
                "[←/→ frame  Shift+←/→ jump 10  ↑/↓ slot  TAB all-view  g goto  q quit]")
    stdscr.addnstr(1, 0, helpline.ljust(w-1)[:w-1], w-1, curses.A_DIM)

    if row_idx is None:
        stdscr.addstr(3, 2, f"(no data for slot {slot} at step {step})",
                      curses.color_pair(2))
        action_line = ""
    else:
        obs = d["obs"][row_idx]
        prev_obs = d["obs"][prev_row] if prev_row is not None else None
        stickX = float(d["stickX"][row_idx])
        stickY = float(d["stickY"][row_idx])
        btn = int(d["buttons"][row_idx])
        action = int(d["action"][row_idx])
        rc = float(d["rc"][row_idx])
        btn_names = [name for mask, name in BUTTON_BITS if btn & mask]

        # Action / control summary line
        action_line = (f" action={action:>3d}  stickX={stickX:+.3f}  stickY={stickY:+.3f}  "
                       f"buttons=0b{btn:04b} ({'+'.join(btn_names) or '—'})  rc={rc:.3f}")

        # Two-column body of grouped panels
        col_w = max(36, w // 2 - 2)
        cols = [(2, []), (col_w + 4, [])]  # left col x=2, right col starts at col_w+4

        # Distribute groups across the two columns by total row count.
        col_rows = [0, 0]
        for label, indices in GROUPS:
            target = 0 if col_rows[0] <= col_rows[1] else 1
            cols[target][1].append((label, indices))
            col_rows[target] += len(indices) + 2

        max_y = h - 3
        for col_x, groups in cols:
            y = 3
            for label, indices in groups:
                if y >= max_y - 1:
                    break
                stdscr.addnstr(y, col_x, f"── {label} ──".ljust(col_w),
                               max(0, w - col_x - 1), curses.A_BOLD)
                y += 1
                for fi in indices:
                    if y >= max_y - 1:
                        break
                    val_str = format_value(fi, obs[fi])
                    label_str = FIELD_LABELS[fi]
                    line = f" {label_str:<18s} {val_str}"
                    changed = (prev_obs is not None and
                               not np.isclose(float(obs[fi]), float(prev_obs[fi]),
                                              atol=1e-6, rtol=1e-6))

                    # Bitfield annotation: list active known bits.
                    extra = ""
                    if FIELD_LABELS[fi].startswith("BitField"):
                        v = int(obs[fi]) & 0xFFFFFFFF
                        active_bits = []
                        # MSB-from-0 PowerPC convention: bit N → (1 << (31 - N))
                        if fi == 50:  # BitField3
                            if v & (1 << (31 - 3)):
                                active_bits.append("wallCol")
                        if fi == 49:  # BitField2
                            if v & (1 << 31):
                                active_bits.append("wheelie")
                        if active_bits:
                            extra = f"  ← {','.join(active_bits)}"
                    line = (line + extra).ljust(col_w)
                    attr = curses.color_pair(1) | curses.A_BOLD if changed else 0
                    stdscr.addnstr(y, col_x, line, max(0, w - col_x - 1), attr)
                    y += 1
                y += 1

    if action_line:
        stdscr.addnstr(h - 2, 0, action_line.ljust(w-1)[:w-1], w-1,
                       curses.color_pair(3) | curses.A_BOLD)

    stdscr.refresh()


def render_all(stdscr, state) -> None:
    h, w = stdscr.getmaxyx()
    d = state["d"]
    step = state["steps_sorted"][state["frame_i"]]
    slots = sorted({s for ss in state["index"].values() for s in ss})

    stdscr.erase()
    n_steps = len(state["steps_sorted"])
    race_info_row = None
    for s in slots:
        ri = state["index"].get(step, {}).get(s)
        if ri is not None:
            race_info_row = d["race_info"][ri]
            break

    rs = int(race_info_row[0]) if race_info_row is not None else 0
    fc = int(race_info_row[1]) if race_info_row is not None else 0

    header = (f" NPZ: {state['args'].npz.name}  frame {state['frame_i']+1}/{n_steps} "
              f"(step {step})  race_stage={rs}  FrameCount={fc}  ALL-SLOTS VIEW")
    stdscr.addnstr(0, 0, header.ljust(w-1), w-1, curses.A_REVERSE)
    help_ = " [←/→ frame  Shift+←/→ jump 10  TAB single-view  g goto  q quit]"
    stdscr.addnstr(1, 0, help_.ljust(w-1)[:w-1], w-1, curses.A_DIM)

    # Column header
    cols = ["slot", "rc", "pos_x", "pos_z", "speed", "stickX", "btn", "drift",
            "item", "KCP", "action"]
    col_widths = [4, 7, 10, 10, 8, 8, 6, 6, 4, 4, 7]
    y = 3
    line = " ".join(c.rjust(w_) for c, w_ in zip(cols, col_widths))
    stdscr.addnstr(y, 1, line, w-2, curses.A_BOLD)
    y += 1
    stdscr.addnstr(y, 1, "-" * min(w - 2, sum(col_widths) + len(col_widths)), w-2)
    y += 1

    for s in slots:
        if y >= h - 2:
            break
        row_idx = state["index"].get(step, {}).get(s)
        if row_idx is None:
            stdscr.addnstr(y, 1, f" slot {s:<2} — (no data)", w-2, curses.A_DIM)
            y += 1
            continue
        obs = d["obs"][row_idx]
        rc = float(d["rc"][row_idx])
        speed = float(obs[38])
        stickX = float(d["stickX"][row_idx])
        btn = int(d["buttons"][row_idx])
        drift = int(obs[40])
        item = int(obs[65])
        kcp = int(obs[12])
        action = int(d["action"][row_idx])
        vals = [
            f"{s:>{col_widths[0]}d}",
            f"{rc:>{col_widths[1]}.3f}",
            f"{float(obs[POS_X_IDX]):>{col_widths[2]}.1f}",
            f"{float(obs[POS_Z_IDX]):>{col_widths[3]}.1f}",
            f"{speed:>{col_widths[4]}.2f}",
            f"{stickX:>+{col_widths[5]}.3f}",
            f"{btn:>{col_widths[6]}d}",
            f"{drift:>{col_widths[7]}d}",
            f"{item:>{col_widths[8]}d}",
            f"{kcp:>{col_widths[9]}d}",
            f"{action:>{col_widths[10]}d}",
        ]
        line = " ".join(vals)
        attr = curses.color_pair(3) if s == state["slot"] else 0
        stdscr.addnstr(y, 1, line, w-2, attr)
        y += 1

    stdscr.refresh()


def goto_prompt(stdscr, state) -> None:
    h, w = stdscr.getmaxyx()
    stdscr.addnstr(h - 1, 0, " goto frame: ".ljust(w - 1), w - 1,
                   curses.color_pair(3) | curses.A_BOLD)
    curses.echo()
    curses.curs_set(1)
    try:
        s = stdscr.getstr(h - 1, 13, 12).decode("utf-8").strip()
        if s:
            n = int(s)
            n = max(0, min(n, len(state["steps_sorted"]) - 1))
            state["frame_i"] = n
    except (ValueError, KeyboardInterrupt):
        pass
    finally:
        curses.noecho()
        curses.curs_set(0)


def tui(stdscr, state) -> None:
    curses.curs_set(0)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_YELLOW, -1)  # changed
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # dim msg
        curses.init_pair(3, curses.COLOR_GREEN, -1)   # active / action

    while True:
        try:
            if state["view"] == "single":
                render_single(stdscr, state)
            else:
                render_all(stdscr, state)
        except curses.error:
            # Terminal too small — render nothing, wait for resize.
            pass

        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            return

        n_steps = len(state["steps_sorted"])
        slots = sorted({s for ss in state["index"].values() for s in ss})

        if ch in (ord("q"), 27):  # 27 = ESC
            return
        elif ch in (curses.KEY_RIGHT, ord("l")):
            state["frame_i"] = min(state["frame_i"] + 1, n_steps - 1)
        elif ch in (curses.KEY_LEFT, ord("h")):
            state["frame_i"] = max(state["frame_i"] - 1, 0)
        elif ch == curses.KEY_SRIGHT:
            state["frame_i"] = min(state["frame_i"] + 10, n_steps - 1)
        elif ch == curses.KEY_SLEFT:
            state["frame_i"] = max(state["frame_i"] - 10, 0)
        elif ch == curses.KEY_DOWN:
            i = slots.index(state["slot"]) if state["slot"] in slots else 0
            state["slot"] = slots[(i + 1) % len(slots)]
        elif ch == curses.KEY_UP:
            i = slots.index(state["slot"]) if state["slot"] in slots else 0
            state["slot"] = slots[(i - 1) % len(slots)]
        elif ch == ord("\t"):
            state["view"] = "all" if state["view"] == "single" else "single"
        elif ch == ord("g"):
            goto_prompt(stdscr, state)
        elif ch == curses.KEY_RESIZE:
            pass  # render loop will pick up new size


def main() -> None:
    args = parse_args()
    d = load_npz(args.npz)
    index = index_by_step_slot(d)
    steps_sorted = sorted(index.keys())
    if not steps_sorted:
        raise SystemExit("NPZ has zero rows.")

    initial_slot = args.slot
    if initial_slot is None:
        initial_slot = sorted({s for ss in index.values() for s in ss})[0]

    state = {
        "d": d,
        "index": index,
        "steps_sorted": steps_sorted,
        "frame_i": max(0, min(args.frame, len(steps_sorted) - 1)),
        "slot": initial_slot,
        "view": "single",
        "args": args,
    }

    curses.wrapper(tui, state)


if __name__ == "__main__":
    main()
