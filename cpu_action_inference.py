"""Discretize CPU AI's per-kart input into the 20-action discrete label.

The slave (DolphinScript.py:send_transition) reads each kart's
`KPad::mRaceInputState` directly via the chain documented in
analysis/12phack/11-cpu-distill.md and forwards mStick.x / mStick.y / mButtons
to the master in the info dict. This module discretizes those raw values into
the action int that DolphinInstance._decode_action consumes.

Action encoding (matches DolphinInstance with n_actions=20, stride=4):
    action = stick_idx * 4 + r_idx * 2 + l_idx
    stick_idx ∈ {0..4} → stickX_values = [-1, -0.4, 0, 0.4, 1]
    r_idx     ∈ {0,1} → drift trigger (R)
    l_idx     ∈ {0,1} → item use (L)
    up_idx is hard-coded to 0 (up_values = [True])

We discretize stickX by nearest-neighbour in the 5-bin grid:
    cuts = [-0.7, -0.2, 0.2, 0.7]   # 4 cuts → 5 bins centered at the 5 stickX values

For R (drift) and L (item) we read from mButtons. The bit layout for MKW's
KPadRaceInputState::mButtons is documented in riidefi/mkw decomp:
    bit 0  accel
    bit 1  brake
    bit 2  item       ← L
    bit 3  drift      ← R
(Same field for real and CPU karts; KPadAIController publishes mCpuStick into
mRaceInputState each frame, including the buttons mask the AI selected.)
"""

from __future__ import annotations

import numpy as np

# Stick discretisation thresholds applied to mStick.x ∈ [-1, 1].
# Tuned to map cleanly onto the 5 stickX bins {-1, -0.4, 0, 0.4, 1}.
STICK_THRESHOLDS = (-0.7, -0.2, 0.2, 0.7)

# mButtons bit layout (riidefi/mkw KPadController.hpp / KPadRaceInputState).
BUTTON_ACCEL = 1 << 0
BUTTON_BRAKE = 1 << 1
BUTTON_ITEM  = 1 << 2
BUTTON_DRIFT = 1 << 3

# Action layout constants (kept in sync with DolphinScript._decode_action:
# stickX_values=5, r_values=2, up_values=1, l_values=2 → 20).
N_STICK_BINS = 5
N_R_BINS     = 2
N_L_BINS     = 2
N_ACTIONS    = N_STICK_BINS * N_R_BINS * N_L_BINS  # = 20
STICK_STRIDE = N_R_BINS * N_L_BINS                 # = 4 (up_values length is 1)


def _stick_bin_scalar(value: float) -> int:
    for k, cut in enumerate(STICK_THRESHOLDS):
        if value < cut:
            return k
    return N_STICK_BINS - 1


def encode_action(stick_x: float, mButtons: int) -> int:
    """Discretize a single (mStick.x, mButtons) pair into the 20-action label."""
    stick_idx = _stick_bin_scalar(float(stick_x))
    r_idx = 1 if (int(mButtons) & BUTTON_DRIFT) else 0
    l_idx = 1 if (int(mButtons) & BUTTON_ITEM)  else 0
    return stick_idx * STICK_STRIDE + r_idx * N_L_BINS + l_idx


def encode_actions_batch(stick_x: np.ndarray, mButtons: np.ndarray) -> np.ndarray:
    """Vectorised version. Inputs are 1-D arrays of equal length.

    Returns int64 action labels in [0, 20).
    """
    stick_x  = np.asarray(stick_x, dtype=np.float32)
    mButtons = np.asarray(mButtons, dtype=np.int64)

    cuts = np.asarray(STICK_THRESHOLDS, dtype=stick_x.dtype)
    stick_idx = np.searchsorted(cuts, stick_x, side="right").astype(np.int64)

    r_idx = ((mButtons & BUTTON_DRIFT) > 0).astype(np.int64)
    l_idx = ((mButtons & BUTTON_ITEM)  > 0).astype(np.int64)

    return stick_idx * STICK_STRIDE + r_idx * N_L_BINS + l_idx


def action_histogram(actions: np.ndarray) -> dict:
    """Quick diagnostic — count occurrences of each action label."""
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=N_ACTIONS)
    return {int(a): int(c) for a, c in enumerate(counts) if c > 0}
