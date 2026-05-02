"""Compare per-kart trajectories from a `run_random_12p_<phase>.csv` to AI-line baseline.

Specifically tests whether our memory-write overrides are *actually* affecting the
kart's behavior. The AI in MKW races a known route; if our writes work, kart
trajectories diverge from the AI line. If our writes are no-ops (AI still
driving), every kart traces the AI route regardless of the action we sent.

For an "all_left" phase (every kart commanded `stickX=-1.0, accel`), the
expected outcomes are:

  * Override works   → karts swerve left, |Δposition| accumulates leftward
                       (low z growth, some x drift); none reach late checkpoints.
  * Override no-op   → every kart races the AI line normally; RaceCompletion
                       progresses identically to a free-AI run.

Usage:
    python analyze_trajectories.py [--csv instance_info/run_random_12p_all_left.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--env", type=int, default=0)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    if args.csv is None:
        # default: any run_random_12p_*.csv in instance_info/
        cands = sorted((here / "instance_info").glob("run_random_12p_*.csv"))
        if not cands:
            print("[analyze] no run_random_12p_*.csv found")
            return
        csv_path = cands[-1]
    else:
        csv_path = Path(args.csv)
    print(f"[analyze] reading {csv_path}")

    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if int(r["env"]) == args.env]
    if not rows:
        print("[analyze] no rows for env", args.env)
        return

    by_kart = defaultdict(list)  # kart -> list of (step, rc, x, z, race_pos)
    for r in rows:
        by_kart[int(r["kart"])].append((
            int(r["step"]),
            float(r["RaceCompletion"]),
            float(r["kart_x"]),
            float(r["kart_z"]),
            int(r["race_pos"]),
        ))

    # Total path length (sum of |Δposition| step over step) and RC delta
    print(f"\n{'kart':>4}  {'rc0':>6}  {'rcN':>6}  {'Δrc':>7}  {'pathlen':>10}  {'mean|Δx|':>10}  {'mean|Δz|':>10}  {'rp_first':>8} {'rp_last':>8}")
    for k in sorted(by_kart):
        seq = by_kart[k]
        if len(seq) < 2:
            continue
        rc0, rcN = seq[0][1], seq[-1][1]
        dxs = [seq[i][2] - seq[i - 1][2] for i in range(1, len(seq))]
        dzs = [seq[i][3] - seq[i - 1][3] for i in range(1, len(seq))]
        pathlen = sum(math.hypot(dx, dz) for dx, dz in zip(dxs, dzs))
        m_dx = sum(abs(dx) for dx in dxs) / len(dxs)
        m_dz = sum(abs(dz) for dz in dzs) / len(dzs)
        rp_first = seq[0][4]
        rp_last = seq[-1][4]
        print(f"{k:>4}  {rc0:>6.3f}  {rcN:>6.3f}  {rcN - rc0:>+7.3f}  {pathlen:>10.0f}  {m_dx:>10.2f}  {m_dz:>10.2f}  {rp_first:>8} {rp_last:>8}")

    # Pairwise trajectory similarity: for each pair (k0, k1) compute mean (Δx,Δz)
    # cosine similarity. If override works and all karts get the same input, they
    # should all move similarly. If AI drives, kart 0 (forced via GC) should look
    # different from karts 1-11 (memory writes — possibly no-op).
    karts = sorted(by_kart)
    if len(karts) >= 2 and len(by_kart[karts[0]]) >= 30:
        print("\nPairwise mean cosine similarity of step-by-step (Δx, Δz) vs kart 0:")
        steps0 = by_kart[karts[0]]
        v0 = [(steps0[i][2] - steps0[i - 1][2], steps0[i][3] - steps0[i - 1][3])
              for i in range(1, len(steps0))]
        for k in karts[1:]:
            seq = by_kart[k]
            if len(seq) < len(v0) + 1:
                continue
            v = [(seq[i][2] - seq[i - 1][2], seq[i][3] - seq[i - 1][3])
                 for i in range(1, len(v0) + 1)]
            num = sum(a * c + b * d for (a, b), (c, d) in zip(v0, v))
            d0 = math.sqrt(sum(a * a + b * b for a, b in v0))
            d1 = math.sqrt(sum(c * c + d * d for c, d in v))
            cos = (num / (d0 * d1)) if d0 > 1e-9 and d1 > 1e-9 else float("nan")
            print(f"  cos(kart0, kart{k:>2}) = {cos:+.4f}")


if __name__ == "__main__":
    main()
