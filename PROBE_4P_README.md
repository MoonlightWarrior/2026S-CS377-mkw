# 4-player controllability probe (`probe_4p_control.py`)

Standalone Dolphin probe that verifies whether the **4-player, no-CPU
"same-character glitch" savestate** (4× Funky Kong on the Flame Runner /
Bowser bike) can be controlled **independently per player** from a Dolphin
Python script.

## TL;DR — what it proved

Driving exactly **one GameCube controller port** per run (full accelerate +
hard steer) and reading every kart's position/speed from memory each frame:

| Driven GC port | slot 0 | slot 1 | slot 2 | slot 3 |
|---|---|---|---|---|
| port 0 | **3758** | 77 | 77 | 77 |
| port 1 | 77 | **3682** | 286 | 77 |
| port 2 | 77 | 77 | **2124** | 77 |
| port 3 | 77 | 77 | 77 | **3054** |

(xz displacement over a 240-frame window; the ~77-unit off-diagonal is the
harmless start-line creep.)

**Conclusion:** each of the 4 real-local players is independently controllable
via `controller.set_gc_buttons(port, {...})`, with a clean **GC port _i_ → kart
slot _i_** mapping. Confirmed on both `RMCP01_4p_funky.sav` and
`allsameafter.sav` (they are the same pre-race grid state).

## ⚠️ The non-obvious gotcha: enable 4 SI controllers

Both savestates are captured at the **race-start intro** (RaceManager
`frameCount` ~60–90). Under Dolphin's default headless config only `SIDevice0`
is a controller (ports 1–3 = *None*), so a **4-player race-init freezes at the
intro** — Dolphin keeps emulating frames but the game's race state machine never
advances (frameCount stuck, karts static). More warmup frames do **not** help.

The fix (already baked into `run_probe.sh`):

```
-C Dolphin.Core.SIDevice0=6 -C Dolphin.Core.SIDevice1=6 \
-C Dolphin.Core.SIDevice2=6 -C Dolphin.Core.SIDevice3=6     # 6 = standard GC controller
```

With that, the countdown elapses (race goes active around frame ~270–280) and
inputs take effect.

## Files

- `probe_4p_control.py` — standalone Dolphin embedded script. **Does not** use
  the master socket / shared memory / PPO machinery. Loads the savestate, holds
  A through the countdown until a kart moves, then drives one controller and
  logs all 12 karts to CSV. Exits via `os._exit(0)`.
- `run_probe.sh` — in-container launcher (holds the `-C SIDevice` flags and the
  headless Dolphin args).

## Prerequisites — savestates (not in git)

`MarioKartSaveStates/` is gitignored and these `.sav` files are **not** in the
docker image. Get `RMCP01_4p_funky.sav` and `allsameafter.sav` from Dongha and
drop them into `Vlab-WiiRL/MarioKartSaveStates/` before running.

## How to run

One env at a time (the binary loads one Dolphin instance). Pick the controller
to drive with `PROBE_PORT`:

```bash
HEADLESS=1 NUM_ENVS=1 docker compose run --rm \
  -e PROBE_SAV=allsameafter.sav \
  -e PROBE_CHANNEL=gc \
  -e PROBE_PORT=0 \
  -e PROBE_WAIT_ALLPRESS=1 \
  wii-rl bash /src/Vlab-WiiRL/run_probe.sh
```

Env vars:

| var | default | meaning |
|---|---|---|
| `PROBE_SAV` | `RMCP01_4p_funky.sav` | savestate filename under `MarioKartSaveStates/` |
| `PROBE_CHANNEL` | `gc` | `gc` (set_gc_buttons) or `wii` (set_wiimote_buttons, best-effort) |
| `PROBE_PORT` | `0` | which controller (0–3) to drive |
| `PROBE_STICK` | `1.0` | steer; +1 = hard right |
| `PROBE_FRAMES` | `240` | measured drive frames |
| `PROBE_WARMUP` | `180` | boot frames before savestate load |
| `PROBE_MAXWAIT` | `1800` | max frames to wait for the race to start |
| `PROBE_WAIT_ALLPRESS` | `0` | hold A on all 4 ports during the countdown (use `1` for these intro savestates) |

## Output

Written to `instance_info/` (gitignored):

- `probe4p_<sav>_<channel>_p<port>.summary.txt` — config dump (per-slot
  type/controller/kart/char), `PHASE A` race-start result (`emu_advanced`,
  `race_frame`), and per-slot displacement + peak speed.
- `probe4p_<sav>_<channel>_p<port>.csv` — per-frame pos/speed for all 12 karts.

## Caveats

- Verification is **frameCount-based, not visual** (headless). `emu_advanced`
  compares RaceManager frameCount (`*0x809BD730 → +0x20`) before/after.
- The `stage` byte (`*0x809BD730 → +0x28`) reads **0 even mid-race** in this
  build — unreliable. Use frameCount or kart speed (`KartMove+0x20`) as the
  live signal.
- Runs are slower under CPU contention (e.g. a training job on the box); give
  the `docker compose run` a generous timeout.
