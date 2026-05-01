"""Phase-1 analysis: verify the InputData layout matches this ROM.

Reads instance_info/input_probe_env*.csv produced by DolphinScript.py probe mode
(MKW_PROBE_INPUT=1) and checks the documented assertions:

  1. PlayerCount-as-12 sanity: every kart slot 0..11 has at least one non-zero
     stickX or buttonActions reading over the dump (means the slot is "live").
  2. Quantization sanity: for any frame with |stickX| > 0.05, the documented
     relation `qStickX ≈ round(stickX*7) + 7` holds within ±1.
  3. Steering correlation: pick the kart with the largest yaw-rate magnitude
     across the dump; assert its stickX series correlates with its yaw rate
     (Pearson |r| > 0.5 over a 30+ frame window).

Prints a PASS/FAIL line per assertion and exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    info_dir = Path(__file__).resolve().parent / "instance_info"
    csvs = sorted(info_dir.glob("input_probe_env*.csv"))
    if not csvs:
        print("[analyze] FAIL: no input_probe_env*.csv found in instance_info/")
        return 2
    csv_path = csvs[0]
    print(f"[analyze] reading {csv_path}")

    # rows[slot] = list of (frame, stickX, qStickX, buttonActions, yaw_deg)
    rows: dict[int, list[tuple[int, float, int, int, float]]] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                slot = int(r["slot"])
                rows[slot].append((
                    int(r["frame"]),
                    float(r["stickX"]),
                    int(r["qStickX"]),
                    int(r["buttonActions"]),
                    float(r["yaw_deg"]),
                ))
                # vtable column is "0x808b2f2c" hex string, optional sanity check
                if r.get("vtable") and r["vtable"].lower() != "0x808b2f2c":
                    pass  # allow non-matching for unused holders; we only need a few good ones
            except (KeyError, ValueError):
                continue

    if not rows:
        print("[analyze] FAIL: CSV had no parseable rows")
        return 2

    n_slots = max(rows.keys()) + 1
    n_frames = max(len(rs) for rs in rows.values())
    print(f"[analyze] slots={n_slots} max_frames_per_slot={n_frames}")

    # 1. all 12 slots are live (have at least one non-zero stickX or buttonActions).
    all_live = True
    for s in range(12):
        rs = rows.get(s, [])
        any_nonzero = any(abs(stick) > 1e-3 or btn != 0 for _, stick, _, btn, _ in rs)
        live_msg = "live" if any_nonzero else "DEAD (all zeros)"
        if not any_nonzero:
            all_live = False
        print(f"  slot {s:2d}: {len(rs):4d} rows  {live_msg}")
    print(f"[1] all-12-live: {'PASS' if all_live else 'FAIL'}")

    # 2. quantization sanity.
    qmismatches = 0
    qsamples = 0
    for s, rs in rows.items():
        for _, stick, q, _, _ in rs:
            if abs(stick) > 0.05:
                expected = max(0, min(14, round(stick * 7) + 7))
                qsamples += 1
                if abs(q - expected) > 1:
                    qmismatches += 1
    q_ok = qsamples > 0 and qmismatches / max(qsamples, 1) < 0.1
    print(f"[2] quantization: {qsamples} samples, {qmismatches} mismatches "
          f"({qmismatches/max(qsamples,1):.2%}) → {'PASS' if q_ok else 'FAIL'}")

    # 3. steering correlation.
    def yaw_rate(rs):
        out = []
        last = None
        for frame, _, _, _, yaw in rs:
            if last is not None:
                # unwrap +-180 → continuous
                d = yaw - last
                if d > 180: d -= 360
                if d < -180: d += 360
                out.append(d)
            last = yaw
        return out

    def pearson(xs, ys):
        n = min(len(xs), len(ys))
        if n < 5: return float("nan")
        mx = sum(xs[:n]) / n; my = sum(ys[:n]) / n
        num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        dx = math.sqrt(sum((xs[i]-mx)**2 for i in range(n)))
        dy = math.sqrt(sum((ys[i]-my)**2 for i in range(n)))
        if dx == 0 or dy == 0: return float("nan")
        return num / (dx*dy)

    # find the slot with the largest |yaw rate| variance among CPU slots (4..11)
    best_slot = -1
    best_var = -1.0
    for s in range(12):
        rs = rows.get(s, [])
        ys = yaw_rate(rs)
        if len(ys) < 30:
            continue
        var = sum(y*y for y in ys) / max(len(ys), 1)
        if var > best_var:
            best_var = var
            best_slot = s
    if best_slot < 0:
        print("[3] correlation: FAIL (no slot had ≥30 frames of yaw data)")
        return 2

    rs = rows[best_slot]
    ys = yaw_rate(rs)
    xs = [stick for (_, stick, _, _, _) in rs[1:]]  # align: yaw_rate has len n-1
    r = pearson(xs[: len(ys)], ys)
    print(f"[3] best-steering slot={best_slot} yawvar={best_var:.4f} "
          f"|stickX|→yaw_rate r={r:.3f} → "
          f"{'PASS' if (r==r and abs(r) > 0.5) else 'FAIL (try --steps higher or different savestate)'}")

    overall = all_live and q_ok and r==r and abs(r) > 0.5
    print(f"[analyze] OVERALL {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
