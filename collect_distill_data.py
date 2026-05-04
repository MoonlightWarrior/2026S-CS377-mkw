"""Collect (per-kart obs, CPU AI action) pairs for behaviour cloning.

The slave runs in PROBE_INPUT mode (read-only `apply_actions`) with all NOP /
override env vars disabled, so the natural MKW pipeline drives every kart:
real-player slots via their controller path, CPU slots via
Enemy::AIEngine → KPadAIController → KPad::mRaceInputState. The slave reads
each kart's mRaceInputState every frame and forwards mStick.x / mStick.y /
mButtons to the master in the info dict; this script discretizes those into
20-action labels (cpu_action_inference.encode_actions_batch) and writes one
row per (step, env, kart) to an NPZ archive.

Output schema (NPZ):
    obs:        float32 (N, 78)        per-kart observation slice
    race_info:  float32 (N, 5)         (race_stage, FrameCount, PlayerCount, CourseID, EngineClass)
    action:     int64   (N,)           discrete action label in [0, 20)
    stickX:     float32 (N,)           raw mStick.x (post-discretisation reference)
    stickY:     float32 (N,)           raw mStick.y
    buttons:    int32   (N,)           raw mButtons mask
    slot:       int8    (N,)           kart slot 0..11
    step:       int32   (N,)           master step counter
    env_id:     int8    (N,)           which DolphinEnv slave (always 0 for num_envs=1)
    rc:         float32 (N,)           RaceCompletion at this step

By default we record only slots 4..11 (the CPU-driven ones in the standard
RMCP01 1-player + 11-CPU savestate) — slot 0 is the player and slots 1..3 are
TYPE_NONE in that savestate. Use --slots to override.

Usage (default config writes to instance_info/distill_dataset.npz):

    HEADLESS=1 NUM_ENVS=1 docker compose run --rm wii-rl \
        python collect_distill_data.py --steps 1000

The script sets MKW_PROBE_INPUT=1 + un-NOPs the AI pipeline before constructing
DolphinEnv so the env vars propagate into the spawned slaves.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# Configure the slave's environment BEFORE importing DolphinEnv (the subprocess
# inherits os.environ at Popen time, but setting these as early as possible
# also matches the project's MKW_* env-var convention).
_DISTILL_ENV = {
    # Read-only mode: apply_actions reads InputState but never writes.
    "MKW_PROBE_INPUT": "1",
    # Keep MKW's natural input pipeline alive — none of these may be NOP'd.
    "MKW_NOP_UPDATEFROMINPUT": "0",
    "MKW_NOP_AI_CALC":         "0",
    "MKW_NOP_INPUTMANAGER":    "0",
    "MKW_NOP_DRIVER_MANAGER":  "0",
    "MKW_NOP_GHOSTPAD":        "0",
    # Don't convert CPU slots to real_local — they must stay PLAYER_CPU so the
    # AI subsystem keeps driving them.
    "MKW_FORCE_REAL_LOCAL":   "0",
    # Disable physics-side overrides (they'd corrupt the CPU's natural action).
    "MKW_FORCE_ANGVEL":             "0",
    "MKW_FORCE_KARTMOVE_SPEED":     "0",
    "MKW_FORCE_KARTMOVE_SPEED_ALL": "0",
    # Don't reload savestate when the race ends; we'll stop master-side.
    "MKW_DISABLE_RESET": "1",
    # Heavy Memory.Addresses tracker has a known hang resolving slot 4 chains
    # under play_num=12 (some intermediate pointer dereferences hang Dolphin's
    # emulator even with the bounds check). Confine the heavy tracker to slot 0
    # — slots 1..11 are populated via the lite `_safe_read_*` path in get_obs,
    # and the CPU-distill labels come from the independent KPad chain in
    # send_transition. So play_num=1 gives a complete obs without the hang.
    "MKW_PLAY_NUM": "1",
}
for _k, _v in _DISTILL_ENV.items():
    os.environ[_k] = _v

from DolphinEnv import DolphinEnv  # noqa: E402  (must come after os.environ setup)
from cpu_action_inference import (  # noqa: E402
    BUTTON_DRIFT,
    BUTTON_ITEM,
    N_ACTIONS,
    action_histogram,
    encode_actions_batch,
)

NUM_KARTS = 12
PER_KART_OBS_DIMS = 78
RACE_INFO_DIMS = 5  # (race_stage, FrameCount, PlayerCount, CourseID, EngineClass)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000,
                   help="Master step cap. One step ≈ frameskip=4 game frames.")
    p.add_argument("--slots", type=int, nargs="+",
                   default=[4, 5, 6, 7, 8, 9, 10, 11],
                   help="Kart slots to record. Default = CPU-driven slots in 1p+11CPU savestates.")
    p.add_argument("--out", type=str, default="instance_info/distill_dataset.npz",
                   help="Output NPZ path (relative to Vlab-WiiRL).")
    p.add_argument("--reset_savestate", type=str, default="MarioKartSaveStates/RMCP01.s08",
                   help="Override the savestate path used for env reset. Default = mid-race "
                        "(s08); race-start savestates produce no AI input during countdown.")
    p.add_argument("--episode_timeout_steps", type=int, default=10_000,
                   help="Slave-side timeout. Set high so we never hit it within --steps.")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--stop_on_race_end", action="store_true", default=True,
                   help="Break when race_stage reaches 4 (race ended).")
    p.add_argument("--no_stop_on_race_end", dest="stop_on_race_end", action="store_false")
    p.add_argument("--warmup_steps", type=int, default=8,
                   help="Skip recording for the first N steps (countdown / mem chains settling).")
    p.add_argument("--stoch", action="store_true",
                   help="Enable stochastic AI perturbation (MKW_STOCH=1). Slave injects "
                        "macro-action chunks (LEFT/RIGHT/BRAKE/BOOST) on KartDynamics.angVel.y "
                        "and KartMove.speed for chunk_size master steps; perturbed rows are "
                        "dropped from the dataset so labels remain clean AI-at-deviated-state.")
    p.add_argument("--stoch_chunk", type=int, default=10,
                   help="Master steps per stochastic macro chunk (default 10 ≈ 0.66s at frameskip=4).")
    p.add_argument("--target_rows", type=int, default=0,
                   help="If >0, run until this many CLEAN rows are recorded (perturbed rows "
                        "don't count). --steps becomes a hard upper bound on the loop.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    out_path = (project_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.stoch:
        os.environ["MKW_STOCH"] = "1"
        os.environ["MKW_STOCH_CHUNK"] = str(args.stoch_chunk)
        os.environ["MKW_STOCH_SLOTS"] = ",".join(str(s) for s in args.slots)
        print(f"[collect] stochastic ON: chunk={args.stoch_chunk} slots={args.slots}")
    print(f"[collect] env vars set: {sorted(list(_DISTILL_ENV.keys()) + ['MKW_STOCH'] if args.stoch else _DISTILL_ENV.keys())}")
    print(f"[collect] steps={args.steps} target_rows={args.target_rows or 'unlimited'} slots={args.slots} → {out_path}")

    env = DolphinEnv(
        num_envs=1,
        reset_mode="savestate",
        reset_savestate=args.reset_savestate,
        episode_timeout_steps=args.episode_timeout_steps,
    )
    obs, _ = env.reset()
    print(f"[collect] env up. obs.shape={obs.shape}  (expected (1, 4, {RACE_INFO_DIMS + PER_KART_OBS_DIMS * NUM_KARTS}))")

    # If --target_rows is set, auto-extend --steps so we have headroom: with
    # stochastic on, ~30% of slot-steps are clean (NEUTRAL macro) and the rest
    # are dropped. Multiply by ~4x as a safety margin.
    if args.target_rows > 0:
        needed_steps = max(args.steps, int(args.target_rows * 4 / max(len(args.slots), 1)))
        if needed_steps > args.steps:
            print(f"[collect] auto-extending --steps {args.steps} → {needed_steps} for target_rows={args.target_rows}")
            args.steps = needed_steps

    # Pre-allocate generously; trim at the end. Worst-case rows per env step =
    # len(slots), so steps * len(slots) is a tight upper bound.
    max_rows = args.steps * len(args.slots)
    obs_buf      = np.zeros((max_rows, PER_KART_OBS_DIMS), dtype=np.float32)
    race_buf     = np.zeros((max_rows, RACE_INFO_DIMS),    dtype=np.float32)
    action_buf   = np.zeros((max_rows,),                   dtype=np.int64)
    stickX_buf   = np.zeros((max_rows,),                   dtype=np.float32)
    stickY_buf   = np.zeros((max_rows,),                   dtype=np.float32)
    buttons_buf  = np.zeros((max_rows,),                   dtype=np.int32)
    slot_buf     = np.zeros((max_rows,),                   dtype=np.int8)
    step_buf     = np.zeros((max_rows,),                   dtype=np.int32)
    env_buf      = np.zeros((max_rows,),                   dtype=np.int8)
    rc_buf       = np.zeros((max_rows,),                   dtype=np.float32)

    write_idx = 0
    t0 = time.time()
    last_logged_step = -1
    skipped_warmup = 0

    # Dummy actions — slave is in MKW_PROBE_INPUT=1, so it reads InputState but
    # never writes. The action ints don't matter; pass action 16 (stickX=0,
    # accel-only) for clarity in case probe gets disabled later.
    dummy_actions = np.full((1, NUM_KARTS), 16, dtype=np.int64)

    try:
        for step in range(args.steps):
            env.step_async(dummy_actions)
            obs, rewards, dones, truns, infos = env.step_wait()

            stage_arr = infos.get("race_stage", np.zeros(1, dtype=np.int32))
            stage = int(stage_arr[0])

            if step < args.warmup_steps:
                skipped_warmup += 1
                continue

            # Skip rows during countdown (stage<2): KPad::mRaceInputState is not
            # being published yet, so all stickX/buttons would be zero — empty labels
            # poison the BC dataset. Only race-active frames carry real AI signal.
            if stage < 2:
                continue

            sx_all  = infos.get("cpu_stickX_all",
                                np.zeros((1, NUM_KARTS), dtype=np.float32))[0]
            sy_all  = infos.get("cpu_stickY_all",
                                np.zeros((1, NUM_KARTS), dtype=np.float32))[0]
            btn_all = infos.get("cpu_buttons_all",
                                np.zeros((1, NUM_KARTS), dtype=np.int32))[0]
            rc_all  = infos.get("RaceCompletion_all",
                                np.zeros((1, NUM_KARTS), dtype=np.float32))[0]
            perturb_all = infos.get("cpu_perturb_all",
                                np.zeros((1, NUM_KARTS), dtype=np.int8))[0]

            # obs shape: (num_envs, framestack, 5 + 78*12). Use the latest frame.
            latest = obs[0, -1, :]
            race_info = latest[:RACE_INFO_DIMS]

            for slot in args.slots:
                # Drop perturbed rows: the kart's physics is being externally
                # forced this step, so the trajectory diverges from where the
                # AI would naturally be. We still want the AI's *intent* but
                # not while it's being yanked around.
                if perturb_all[slot]:
                    continue

                start = RACE_INFO_DIMS + PER_KART_OBS_DIMS * slot
                end   = start + PER_KART_OBS_DIMS
                kart_slice = latest[start:end]

                action_label = int(encode_actions_batch(
                    np.array([sx_all[slot]], dtype=np.float32),
                    np.array([btn_all[slot]], dtype=np.int64),
                )[0])

                obs_buf[write_idx]      = kart_slice
                race_buf[write_idx]     = race_info
                action_buf[write_idx]   = action_label
                stickX_buf[write_idx]   = float(sx_all[slot])
                stickY_buf[write_idx]   = float(sy_all[slot])
                buttons_buf[write_idx]  = int(btn_all[slot])
                slot_buf[write_idx]     = slot
                step_buf[write_idx]     = step
                env_buf[write_idx]      = 0
                rc_buf[write_idx]       = float(rc_all[slot])
                write_idx += 1

            if args.target_rows > 0 and write_idx >= args.target_rows:
                print(f"[collect] target_rows={args.target_rows} reached at step {step}; breaking.")
                break

            if args.stop_on_race_end and stage >= 4:
                print(f"[collect] race ended at step {step} (stage={stage}); breaking.")
                break

            if (step % max(1, args.log_every) == 0) and step != last_logged_step:
                last_logged_step = step
                elapsed = time.time() - t0
                sps = (step + 1) / max(elapsed, 1e-6)
                # Diagnostic: action distribution so far.
                hist = action_histogram(action_buf[:write_idx])
                print(f"[collect] step={step:5d} sps={sps:5.1f} stage={stage} "
                      f"rows={write_idx} action_hist={hist}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Trim and save.
    n = write_idx
    print(f"[collect] recorded {n} rows; saving to {out_path}")
    if n == 0:
        print("[collect] WARNING — zero rows recorded. Check savestate / env vars.")
    np.savez_compressed(
        out_path,
        obs=obs_buf[:n],
        race_info=race_buf[:n],
        action=action_buf[:n],
        stickX=stickX_buf[:n],
        stickY=stickY_buf[:n],
        buttons=buttons_buf[:n],
        slot=slot_buf[:n],
        step=step_buf[:n],
        env_id=env_buf[:n],
        rc=rc_buf[:n],
        meta_n_actions=np.array([N_ACTIONS], dtype=np.int32),
        meta_button_drift_mask=np.array([BUTTON_DRIFT], dtype=np.int32),
        meta_button_item_mask=np.array([BUTTON_ITEM],  dtype=np.int32),
    )

    if n > 0:
        hist = action_histogram(action_buf[:n])
        print(f"[collect] final action distribution (label: count): {hist}")
        nonzero_stick = int((np.abs(stickX_buf[:n]) > 0.05).sum())
        print(f"[collect] frames with |stickX|>0.05: {nonzero_stick}/{n} "
              f"({100*nonzero_stick/max(n,1):.1f}%)")


if __name__ == "__main__":
    main()
