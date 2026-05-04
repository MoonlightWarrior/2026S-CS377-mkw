"""Investigate which mButtons bit (if any) corresponds to the AI's
'press item' button by correlating bit transitions with ItemCount drops.

Methodology:
  - Sort rows by (slot, step). Within each slot, look at consecutive frames.
  - "Item consumption event" = ItemCount[t-1] > 0 AND ItemCount[t] == ItemCount[t-1] - 1
    (or smaller for triple items going from 3 → 2 → 1 → 0).
  - For each consumption event, print mButtons at t-1 and t, and the (Item, ItemCount)
    transition.
  - Reverse direction: for each frame where mButtons & 0x4 (bit 2), print the
    (Item, ItemCount) before/after.
  - Per-bit aggregate: for each of the 16 bits, count how often the bit is set in
    the frame *immediately before* a consumption event, and compare against base
    rate (bit set probability across all frames).

If bit 2 is the item button, frames immediately before a consumption should have
bit 2 set far more often than the base rate.
"""
from __future__ import annotations
import argparse
import numpy as np

OBS_ITEM_IDX = 65
OBS_ITEMNUM_IDX = 66

# For readability
ITEM_NAMES = {
    -1: "None", 0: "Green", 1: "Red", 2: "Banana", 3: "FIB",
    4: "Mushroom", 5: "TripShroom", 6: "Bomb", 7: "Blue", 8: "Shock",
    9: "Star", 10: "Golden", 11: "Mega", 12: "Blooper", 13: "POW",
    14: "TC", 15: "Bill", 16: "TripGreen", 17: "TripRed", 18: "TripBanana",
    19: "Unused", 20: "NoItem",
}

# u32 reads — when the field is signed -1, it appears as 4294967295 in unsigned.
def _item_id_normalize(v: float) -> int:
    iv = int(v)
    if iv == 4294967295:
        return -1
    return iv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    z = np.load(args.data)

    obs    = z["obs"]
    btns   = z["buttons"].astype(np.int64)
    slot   = z["slot"]
    step   = z["step"]
    n      = obs.shape[0]
    item   = obs[:, OBS_ITEM_IDX]
    inum   = obs[:, OBS_ITEMNUM_IDX].astype(np.int32)
    print(f"NPZ rows = {n}")

    # Group by slot, sort by step within slot, walk consecutive pairs.
    consume_events = []        # list of dicts
    bit2_events    = []        # list of dicts
    consumed_btn_prev_counts = np.zeros(16, dtype=np.int64)
    consumed_btn_now_counts  = np.zeros(16, dtype=np.int64)
    n_consume = 0

    for s in sorted(set(slot.tolist())):
        m = (slot == s)
        idx = np.where(m)[0]
        order = np.argsort(step[idx])
        idx = idx[order]
        for k in range(1, len(idx)):
            prev = idx[k - 1]
            cur  = idx[k]
            # Consecutive in step? (Skip if there's a gap.)
            if int(step[cur]) - int(step[prev]) != 1:
                continue
            inum_p = int(inum[prev])
            inum_c = int(inum[cur])
            btn_p = int(btns[prev])
            btn_c = int(btns[cur])
            item_p = _item_id_normalize(item[prev])
            item_c = _item_id_normalize(item[cur])

            # Item consumption: count drops by 1 (most common)
            if inum_p > 0 and inum_c < inum_p:
                n_consume += 1
                consume_events.append((int(s), int(step[prev]), int(step[cur]),
                                        inum_p, inum_c,
                                        ITEM_NAMES.get(item_p, "?"),
                                        ITEM_NAMES.get(item_c, "?"),
                                        btn_p, btn_c))
                for b in range(16):
                    if btn_p & (1 << b):
                        consumed_btn_prev_counts[b] += 1
                    if btn_c & (1 << b):
                        consumed_btn_now_counts[b] += 1

            # Bit 2 fire (item button) events
            if (btn_p & 0x4) == 0 and (btn_c & 0x4) != 0:  # rising edge
                bit2_events.append((int(s), int(step[prev]), int(step[cur]),
                                     inum_p, inum_c,
                                     ITEM_NAMES.get(item_p, "?"),
                                     ITEM_NAMES.get(item_c, "?"),
                                     btn_p, btn_c))

    # === Summary 1: consumption events ===
    print(f"\n=== item-consumption events (ItemCount[t-1] > ItemCount[t]): {n_consume} ===")
    if n_consume == 0:
        print("  (no consumption events in this dataset — collect more or use later-race savestate)")
    for ev in consume_events[:30]:
        s, sp, sc, ip, ic, name_p, name_c, bp, bc = ev
        bits_p = [b for b in range(16) if bp & (1 << b)]
        bits_c = [b for b in range(16) if bc & (1 << b)]
        print(f"  slot {s:2d}  step {sp}→{sc}  item {name_p}({ip})→{name_c}({ic})  "
              f"btn {bp:#06x}{bits_p} → {bc:#06x}{bits_c}")
    if len(consume_events) > 30:
        print(f"  ... ({len(consume_events) - 30} more)")

    # === Summary 2: per-bit popcount in consumption frames vs base rate ===
    print("\n=== bit popcount around consumption events ===")
    base_rate = ((btns[:, None] & (1 << np.arange(16))) != 0).sum(axis=0)
    n_total = n
    print(f"  {'bit':<5}{'base_set/N':<18}{'set@prev/N_cons':<22}{'set@cur/N_cons':<22}")
    for b in range(16):
        if base_rate[b] == 0 and consumed_btn_prev_counts[b] == 0:
            continue
        base_pct = 100 * base_rate[b] / n_total
        prev_pct = 100 * consumed_btn_prev_counts[b] / max(n_consume, 1)
        cur_pct  = 100 * consumed_btn_now_counts[b] / max(n_consume, 1)
        print(f"  {b:<5}{f'{int(base_rate[b])}/{n_total} ({base_pct:5.2f}%)':<18}"
              f"{f'{int(consumed_btn_prev_counts[b])}/{n_consume} ({prev_pct:5.2f}%)':<22}"
              f"{f'{int(consumed_btn_now_counts[b])}/{n_consume} ({cur_pct:5.2f}%)':<22}")

    # === Summary 3: bit2 rising-edge events (purported item button) ===
    print(f"\n=== bit-2 rising edges: {len(bit2_events)} ===")
    for ev in bit2_events[:20]:
        s, sp, sc, ip, ic, name_p, name_c, bp, bc = ev
        print(f"  slot {s:2d}  step {sp}→{sc}  item {name_p}({ip})→{name_c}({ic})  "
              f"btn {bp:#06x} → {bc:#06x}")
    if len(bit2_events) > 20:
        print(f"  ... ({len(bit2_events) - 20} more)")

    # === Summary 4: item type distribution at consumption ===
    print("\n=== item types being consumed ===")
    from collections import Counter
    consume_types = Counter()
    for ev in consume_events:
        _, _, _, _, _, name_p, _, _, _ = ev
        consume_types[name_p] += 1
    for name, c in consume_types.most_common():
        print(f"  {name:<12}  {c}")


if __name__ == "__main__":
    main()
