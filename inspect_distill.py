"""Quick offline inspection of a distill NPZ to investigate three open issues:

  (1) ITEM bit position — show all unique mButtons values + per-bit popcount,
      so we can tell which bits are the AI ever exercises.
  (2) lite-path DriftState (obs idx 40) garbage — print value distribution.
  (3) lite-path Item / ItemNum (obs idx 65 / 66) garbage — same.

No memory access; pure NPZ dump.
"""
from __future__ import annotations
import argparse
import numpy as np

OBS_DRIFT_IDX = 40
OBS_ITEM_IDX = 65
OBS_ITEMNUM_IDX = 66
OBS_PASSIVE_IDX = 67
OBS_PASSIVE_NUM_IDX = 68


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    z = np.load(args.data)

    obs       = z["obs"]
    buttons   = z["buttons"].astype(np.int64)
    stickX    = z["stickX"]
    slot      = z["slot"]
    step      = z["step"]
    n         = obs.shape[0]
    print(f"NPZ rows = {n}")

    # === (1) mButtons bit-level distribution ===
    print("\n=== (1) mButtons bit popcount across all 16 bits ===")
    for b in range(16):
        m = 1 << b
        c = int(((buttons & m) != 0).sum())
        if c == 0:
            continue
        print(f"  bit {b:2d} (0x{m:04x})  set in {c:6d} / {n} rows  ({100*c/n:5.2f}%)")

    print("\n=== unique mButtons values seen (top 20) ===")
    uniq, counts = np.unique(buttons, return_counts=True)
    order = np.argsort(-counts)
    for k in order[:20]:
        v = int(uniq[k])
        c = int(counts[k])
        # decode bit positions
        bits = [b for b in range(16) if v & (1 << b)]
        print(f"  buttons=0x{v:04x}  bits={bits}  count={c:6d}  ({100*c/n:5.2f}%)")

    # === (2) DriftState distribution per slot ===
    print("\n=== (2) DriftState (obs idx 40) per slot ===")
    drift = obs[:, OBS_DRIFT_IDX]
    print(f"  overall: min={float(drift.min()):.0f}  max={float(drift.max()):.0f}  "
          f"mean={float(drift.mean()):.1f}")
    real_drift_range = (drift >= 0) & (drift <= 16)
    print(f"  in [0..16] (valid DriftState range): {int(real_drift_range.sum())}/{n}  "
          f"({100*real_drift_range.sum()/n:.2f}%)")
    for s in sorted(set(slot.tolist())):
        m = (slot == s)
        d = drift[m]
        if d.size == 0:
            continue
        print(f"  slot {int(s):2d}: n={int(d.size):4d}  range=[{float(d.min()):.0f}..{float(d.max()):.0f}]  "
              f"mean={float(d.mean()):.1f}  first 5 values={d[:5].astype(int).tolist()}")

    # === (3) Item / ItemNum distribution ===
    print("\n=== (3) Item (obs idx 65) values seen — top 10 unique ===")
    item = obs[:, OBS_ITEM_IDX]
    # Reinterpret garbage interpretation: float→raw bytes→int.
    # obs is float32; item value is stored as float(u32). For valid IDs 0..18 this
    # round-trips exactly. For pointer-shaped reads it loses precision but the
    # magnitude / sign tells us a lot.
    uniq, counts = np.unique(item, return_counts=True)
    order = np.argsort(-counts)
    for k in order[:10]:
        v = float(uniq[k])
        c = int(counts[k])
        v_int = int(v) if -2**31 <= v < 2**31 else None
        # Also try f32 reinterpretation: pack as f32, unpack as u32, see hex.
        as_u32 = int(np.array([v], dtype=np.float32).view(np.uint32)[0])
        print(f"  Item={v:>16.4f}  count={c:6d}  int_view={v_int}  hex_as_f32={as_u32:#010x}")

    print("\n=== ItemNum (obs idx 66) ===")
    inum = obs[:, OBS_ITEMNUM_IDX]
    valid_count = (inum >= 0) & (inum <= 10)
    print(f"  range=[{float(inum.min())}, {float(inum.max())}]   in [0..10]: "
          f"{int(valid_count.sum())}/{n}")
    print(f"  unique (top 10): {sorted(set(inum.tolist()))[:10]}")

    # === (3b) Per-slot first 5 (Item, ItemNum) pairs to look for patterns ===
    print("\n=== per-slot first 5 (Item, ItemNum) pairs ===")
    for s in sorted(set(slot.tolist())):
        m = (slot == s)
        idx = np.where(m)[0][:5]
        pairs = list(zip(item[idx].tolist(), inum[idx].tolist()))
        print(f"  slot {int(s):2d}: {pairs}")


if __name__ == "__main__":
    main()
