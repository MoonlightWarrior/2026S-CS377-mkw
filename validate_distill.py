"""Sanity-check the distill NPZ produced by collect_distill_data.py.

Verifies five things requested for the feature/cpu-distill validation step:

  1. Drift bit (BUTTON_DRIFT, mButtons bit 3) correlates with non-zero
     DriftState (per-kart obs idx 40).
  2. Item bit (BUTTON_ITEM, mButtons bit 2) shows up at all and, when set,
     coincides with ItemNum>0 (obs idx 66) — i.e. the AI is actually pressing
     it on something it owns.
  3. Per-kart x,z (obs idx 16 / 18) move over time — i.e. the karts are on
     the track and the chain isn't returning frozen positions.
  4. Action histogram across all 20 labels.
  5. mStick.x range is in [-1, 1] (chain integrity / no garbage reads).

Output is plain text — no plotting, no matplotlib dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PER_KART_OBS_DIMS = 78
DRIFT_STATE_IDX   = 40
ITEM_IDX          = 65
ITEM_NUM_IDX      = 66
POS_X_IDX         = 16
POS_Y_IDX         = 17
POS_Z_IDX         = 18

BUTTON_ACCEL = 1 << 0
BUTTON_BRAKE = 1 << 1
BUTTON_ITEM  = 1 << 2
BUTTON_DRIFT = 1 << 3

N_ACTIONS = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="NPZ produced by collect_distill_data.py")
    return p.parse_args()


def fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num}/{den} ({100*num/den:5.2f}%)"


def main() -> None:
    args = parse_args()
    path = Path(args.data).resolve()
    if not path.exists():
        raise SystemExit(f"[validate] not found: {path}")

    data = np.load(path)
    obs      = data["obs"]
    action   = data["action"]
    stickX   = data["stickX"]
    stickY   = data["stickY"]
    buttons  = data["buttons"].astype(np.int64)
    slot     = data["slot"]
    step     = data["step"]
    rc       = data["rc"]
    n        = obs.shape[0]
    print(f"[validate] {path}")
    print(f"[validate] rows={n}  obs.shape={obs.shape}  uniq slots={sorted(set(slot.tolist()))}")
    print(f"[validate] step range=[{step.min()}..{step.max()}]")

    # =========================================================================
    # 5. Stick range — do this first since it's a hard correctness gate.
    # =========================================================================
    print("\n=== (5) stickX / stickY range ===")
    sx_min, sx_max = float(stickX.min()), float(stickX.max())
    sy_min, sy_max = float(stickY.min()), float(stickY.max())
    print(f"  stickX: min={sx_min:+.4f}  max={sx_max:+.4f}  mean={float(stickX.mean()):+.4f}  std={float(stickX.std()):.4f}")
    print(f"  stickY: min={sy_min:+.4f}  max={sy_max:+.4f}  mean={float(stickY.mean()):+.4f}  std={float(stickY.std()):.4f}")
    sx_in_range = (-1.001 <= stickX) & (stickX <= 1.001)
    sy_in_range = (-1.001 <= stickY) & (stickY <= 1.001)
    print(f"  stickX in [-1,1]: {fmt_pct(int(sx_in_range.sum()), n)}")
    print(f"  stickY in [-1,1]: {fmt_pct(int(sy_in_range.sum()), n)}")
    nonzero_sx = int((np.abs(stickX) > 0.05).sum())
    print(f"  |stickX|>0.05:   {fmt_pct(nonzero_sx, n)}")

    # =========================================================================
    # 4. Action histogram
    # =========================================================================
    print("\n=== (4) action histogram (label → count, %) ===")
    counts = np.bincount(action, minlength=N_ACTIONS)
    for a, c in enumerate(counts):
        if c == 0:
            continue
        # decode for readability
        stick_idx = a // 4
        r_idx     = (a % 4) // 2
        l_idx     = a % 2
        print(f"  action {a:2d}  stick_idx={stick_idx} R={r_idx} L={l_idx}  count={c:6d}  ({100*c/n:5.2f}%)")
    nz_actions = int((counts > 0).sum())
    print(f"  unique action labels seen: {nz_actions}/{N_ACTIONS}")

    # =========================================================================
    # 1. Drift correlation
    # =========================================================================
    print("\n=== (1) BUTTON_DRIFT (bit 3) vs DriftState (obs idx 40) ===")
    drift_state = obs[:, DRIFT_STATE_IDX]
    drift_bit   = ((buttons & BUTTON_DRIFT) != 0)
    drift_active = (drift_state > 0.5)  # u16; non-zero means in some drift sub-state
    print(f"  drift_bit set:          {fmt_pct(int(drift_bit.sum()), n)}")
    print(f"  DriftState > 0:         {fmt_pct(int(drift_active.sum()), n)}")
    if drift_bit.sum() > 0:
        # of the rows where bit is set, what fraction also have DriftState>0?
        agree = int((drift_bit & drift_active).sum())
        print(f"  bit=1 & state>0:        {fmt_pct(agree, int(drift_bit.sum()))}  (precision-like)")
    if drift_active.sum() > 0:
        agree2 = int((drift_bit & drift_active).sum())
        print(f"  state>0 & bit=1:        {fmt_pct(agree2, int(drift_active.sum()))}  (recall-like)")
    # Confusion matrix
    tp = int((drift_bit & drift_active).sum())
    fp = int((drift_bit & ~drift_active).sum())
    fn = int((~drift_bit & drift_active).sum())
    tn = int((~drift_bit & ~drift_active).sum())
    print(f"  confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    # Show some unique DriftState values
    uniq_drift_state = np.unique(drift_state.astype(np.int32))[:20]
    print(f"  unique DriftState vals (first 20): {uniq_drift_state.tolist()}")

    # =========================================================================
    # 2. Item correlation
    # =========================================================================
    print("\n=== (2) BUTTON_ITEM (bit 2) vs ItemNum (obs idx 66) ===")
    item_bit  = ((buttons & BUTTON_ITEM) != 0)
    item_have = (obs[:, ITEM_NUM_IDX] > 0.5)
    item_kind = obs[:, ITEM_IDX]
    print(f"  item_bit set:           {fmt_pct(int(item_bit.sum()), n)}")
    print(f"  ItemNum > 0:            {fmt_pct(int(item_have.sum()), n)}")
    if item_bit.sum() > 0:
        agree = int((item_bit & item_have).sum())
        print(f"  bit=1 & ItemNum>0:      {fmt_pct(agree, int(item_bit.sum()))}  "
              f"(if low, AI may press item without holding one — or wrong bit)")
    uniq_item = np.unique(item_kind.astype(np.int32))[:20]
    print(f"  unique Item vals (first 20): {uniq_item.tolist()}")

    print("\n  also showing brake / accel for reference (sanity bits 0, 1):")
    accel_bit = ((buttons & BUTTON_ACCEL) != 0)
    brake_bit = ((buttons & BUTTON_BRAKE) != 0)
    print(f"  accel_bit set:          {fmt_pct(int(accel_bit.sum()), n)}  (CPU should accel almost always)")
    print(f"  brake_bit set:          {fmt_pct(int(brake_bit.sum()), n)}  (CPU rarely brakes)")

    # =========================================================================
    # 3. Position drift over time — per-slot
    # =========================================================================
    print("\n=== (3) Per-slot kart position motion (x,z) ===")
    # group rows by slot, sort by step within slot, measure spread + step-to-step delta
    slots_seen = sorted(set(slot.tolist()))
    print(f"  slots: {slots_seen}")
    print(f"  {'slot':<6}{'n':>6}{'x_min':>10}{'x_max':>10}{'z_min':>10}{'z_max':>10}"
          f"{'mean_dx':>10}{'mean_dz':>10}{'rc_min':>8}{'rc_max':>8}")
    for s in slots_seen:
        mask = (slot == s)
        if mask.sum() < 2:
            continue
        idx = np.where(mask)[0]
        # ensure sort by step for proper deltas
        order = np.argsort(step[idx])
        idx = idx[order]
        xs = obs[idx, POS_X_IDX]
        zs = obs[idx, POS_Z_IDX]
        rcs = rc[idx]
        dx = float(np.mean(np.abs(np.diff(xs))))
        dz = float(np.mean(np.abs(np.diff(zs))))
        print(f"  {int(s):<6}{int(mask.sum()):>6}"
              f"{float(xs.min()):>10.1f}{float(xs.max()):>10.1f}"
              f"{float(zs.min()):>10.1f}{float(zs.max()):>10.1f}"
              f"{dx:>10.3f}{dz:>10.3f}"
              f"{float(rcs.min()):>8.3f}{float(rcs.max()):>8.3f}")

    print("\n[validate] done.")


if __name__ == "__main__":
    main()
