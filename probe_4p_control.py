# Standalone Dolphin embedded probe for the 4p-no-CPU "funky glitch" savestate.
#
# Does NOT connect to the master socket / shared memory. It loads
# RMCP01_4p_funky.sav, waits out the start-of-race countdown while holding
# accelerate, then drives exactly ONE controller channel (one GC port or one
# Wii remote) at full accelerate + hard steer, leaving the other three neutral,
# and logs every kart's position + speed each frame to a CSV. From that memory
# trace we can tell which kart slot (if any) that one controller actually
# drives.
#
# Parameterised by env vars so each test is its own `docker compose run`:
#   PROBE_CHANNEL  gc | wii        (default gc)
#   PROBE_PORT     0..3            controller to drive (default 0)
#   PROBE_STICK    -1.0..1.0       steer; +1 = hard right (default 1.0)
#   PROBE_FRAMES   measure frames  (default 240)
#   PROBE_WARMUP   boot frames before savestate load (default 180)
#   PROBE_SETTLE   frames after load before counting (default 20)
#   PROBE_MAXWAIT  max frames to wait for race to start (default 1800)
import os
import sys
from pathlib import Path

from dolphin import event, memory, savestate, controller

SCRIPT_DIR = Path("/src/Vlab-WiiRL")
SAV = SCRIPT_DIR / "MarioKartSaveStates" / os.environ.get("PROBE_SAV", "RMCP01_4p_funky.sav")
OUT_DIR = SCRIPT_DIR / "instance_info"

CHANNEL = os.environ.get("PROBE_CHANNEL", "gc").lower()
PORT = int(os.environ.get("PROBE_PORT", "0"))
STICK = float(os.environ.get("PROBE_STICK", "1.0"))
FRAMES = int(os.environ.get("PROBE_FRAMES", "240"))
WARMUP = int(os.environ.get("PROBE_WARMUP", "180"))
SETTLE = int(os.environ.get("PROBE_SETTLE", "20"))
MAXWAIT = int(os.environ.get("PROBE_MAXWAIT", "1800"))

NUM = 12
SAV_TAG = SAV.stem
TAG = f"{SAV_TAG}_{CHANNEL}_p{PORT}"
CSV_PATH = OUT_DIR / f"probe4p_{TAG}.csv"
SUM_PATH = OUT_DIR / f"probe4p_{TAG}.summary.txt"


def _ok(a):
    return a and (0x80000000 <= a < 0x81800000 or 0x90000000 <= a < 0x94000000)


def r_u32(a):
    try:
        return memory.read_u32(a) if _ok(a) else 0
    except Exception:
        return 0


def r_u16(a):
    try:
        return memory.read_u16(a) if _ok(a) else 0
    except Exception:
        return 0


def r_u8(a):
    try:
        return memory.read_u8(a) if _ok(a) else 0
    except Exception:
        return 0


def r_f32(a):
    try:
        return float(memory.read_f32(a)) if _ok(a) else 0.0
    except Exception:
        return 0.0


def kart_ptrs(i):
    mgr = r_u32(0x809C18F8)
    arr = r_u32(mgr + 0x20) if mgr else 0
    kp = r_u32(arr + 0x4 * i) if arr else 0
    ko = r_u32(kp) if kp else 0
    return kp, ko


def read_kart(i):
    """Return (live, posx, posy, posz, speed) for kart slot i."""
    kp, ko = kart_ptrs(i)
    if not ko:
        return 0, 0.0, 0.0, 0.0, 0.0
    kdc = r_u32(ko + 0x8)
    kdyn = r_u32(kdc + 0x90) if kdc else 0
    px = py = pz = 0.0
    if kdyn:
        px = r_f32(kdyn + 0x18)
        py = r_f32(kdyn + 0x1C)
        pz = r_f32(kdyn + 0x20)
    kmove = r_u32(ko + 0x28)
    spd = r_f32(kmove + 0x20) if kmove else 0.0
    return 1, px, py, pz, spd


def race_info():
    """Return (stage, countdownTimer, frameCount) from RaceManager @ 0x809BD730."""
    rm = r_u32(0x809BD730)
    if not rm:
        return -1, -1, -1
    fc = r_u32(rm + 0x20)
    cd = r_u16(rm + 0x22)
    stage = r_u8(rm + 0x28)
    return stage, cd, fc


def dump_config():
    lines = []
    rd = r_u32(0x809BD728)
    st, cd, fc = race_info()
    lines.append(f"race_data=0x{rd:08x}  stage={st} countdown={cd} frameCount={fc}")
    if rd:
        for i in range(NUM):
            base = rd + 0x28 + 0xF0 * i
            localnum = r_u8(base + 0x05)
            ctrlid = r_u8(base + 0x06)
            ctrltype = r_u8(base + 0x07)
            kart = r_u32(base + 0x08)
            char = r_u32(base + 0x0C)
            ptype = r_u32(base + 0x10)
            live, px, py, pz, spd = read_kart(i)
            lines.append(
                f"  slot {i:2d}: type={ptype} localNum={localnum} ctrlID={ctrlid} "
                f"ctrlType={ctrltype} kart={kart} char={char} live={live} "
                f"pos=({px:.1f},{py:.1f},{pz:.1f}) speed={spd:.2f}"
            )
    return "\n".join(lines)


GC_NEUTRAL = {"StickX": 128, "StickY": 128, "CStickX": 128, "CStickY": 128}


def drive_gc(active_port, stick, steer=True):
    sx = max(0, min(255, int(round(128 + stick * 127)))) if steer else 128
    for p in range(4):
        if p == active_port:
            controller.set_gc_buttons(
                p, {"A": True, "StickX": sx, "StickY": 128, "CStickX": 128, "CStickY": 128}
            )
        else:
            controller.set_gc_buttons(p, dict(GC_NEUTRAL))


def drive_wii(active_port, stick, steer=True):
    for p in range(4):
        if p == active_port:
            try:
                controller.set_wiimote_buttons(p, {"Two": True})
            except Exception:
                pass
            if steer:
                try:
                    controller.set_wiimote_tilt(p, {"X": 0.0, "Y": 0.0, "Z": float(stick)})
                except Exception:
                    pass
        else:
            try:
                controller.set_wiimote_buttons(p, {})
            except Exception:
                pass


def drive(active_port, stick, steer):
    if CHANNEL == "wii":
        drive_wii(active_port, stick, steer)
    else:
        drive_gc(active_port, stick, steer)


def finish(summary):
    try:
        SUM_PATH.write_text(summary)
    except Exception as e:
        print(f"summary write failed: {e}")
    print(summary)
    print(f"PROBE_DONE {TAG}")
    sys.stdout.flush()
    os._exit(0)


# ----------------------------------------------------------------------------
print(f"probe_4p_control start: channel={CHANNEL} port={PORT} stick={STICK} "
      f"frames={FRAMES} warmup={WARMUP} maxwait={MAXWAIT}")
sys.stdout.flush()

for _ in range(WARMUP):
    await event.frameadvance()

loaded = False
for attempt in range(5):
    try:
        savestate.load_from_file(str(SAV))
        loaded = True
        break
    except Exception as e:
        print(f"load attempt {attempt} failed: {e}")
        for _ in range(30):
            await event.frameadvance()
if not loaded:
    finish(f"FAILED to load savestate {SAV}")

for _ in range(SETTLE):
    await event.frameadvance()

cfg = dump_config()
print("=== CONFIG AFTER LOAD ===\n" + cfg)
sys.stdout.flush()

# --- PHASE A: hold accelerate (no steer) until the race goes active (stage==2),
#     or until any kart starts moving, or MAXWAIT frames elapse. --------------
st0, cd0, fc0 = race_info()
stage_seen = set()
race_frame = None
wait_log = []
WAIT_ALLPRESS = os.environ.get("PROBE_WAIT_ALLPRESS", "0") == "1"
fc_seen = set()
for w in range(MAXWAIT):
    if WAIT_ALLPRESS and CHANNEL != "wii":
        for p in range(4):
            controller.set_gc_buttons(p, {"A": True, "StickX": 128, "StickY": 128,
                                          "CStickX": 128, "CStickY": 128})
    else:
        drive(PORT, 0.0, steer=False)
    await event.frameadvance()
    st, cd, fc = race_info()
    stage_seen.add(st)
    fc_seen.add(fc)
    if w % 60 == 0:
        line = f"  wait f={w}: stage={st} countdown={cd} frameCount={fc}"
        wait_log.append(line)
        print(line)
        sys.stdout.flush()
    # race considered active once stage hits 2 (RACE) and a few frames pass
    if st == 2:
        race_frame = w
        break
    # fallback: any kart got real speed -> race is moving
    moving = any(read_kart(i)[4] > 1.0 for i in range(4))
    if moving:
        race_frame = w
        break

st_now, cd_now, fc_now = race_info()
emu_advanced = (fc_now != fc0) or (len(stage_seen - {st0}) > 0)
wait_summary = (
    f"PHASE A: start stage={st0} fc={fc0}; after wait stage={st_now} fc={fc_now}; "
    f"stages_seen={sorted(stage_seen)}; race_frame={race_frame}; "
    f"emu_advanced={emu_advanced}"
)
print(wait_summary)
for l in wait_log:
    print(l)
sys.stdout.flush()

# Baseline positions just before the measured drive.
base_pos = []
for i in range(NUM):
    live, px, py, pz, spd = read_kart(i)
    base_pos.append((px, py, pz))

# --- PHASE B: drive ONE controller at full accel + hard steer; log everything.
f = open(CSV_PATH, "w")
f.write("frame,stage," + ",".join(
    f"s{i}_x,s{i}_y,s{i}_z,s{i}_spd" for i in range(NUM)) + "\n")

for frame in range(FRAMES):
    drive(PORT, STICK, steer=True)
    await event.frameadvance()
    st, cd, fc = race_info()
    cols = [str(frame), str(st)]
    for i in range(NUM):
        live, px, py, pz, spd = read_kart(i)
        cols += [f"{px:.3f}", f"{py:.3f}", f"{pz:.3f}", f"{spd:.3f}"]
    f.write(",".join(cols) + "\n")
    if frame % 60 == 0:
        f.flush()
f.flush()
os.fsync(f.fileno())
f.close()

# Per-slot displacement + peak speed over the measured window.
moved = []
peakspd = [0.0] * NUM
for i in range(NUM):
    live, px, py, pz, spd = read_kart(i)
    bx, by, bz = base_pos[i]
    moved.append(((px - bx) ** 2 + (pz - bz) ** 2) ** 0.5)
# peak speed scanned from CSV
import csv as _csv
with open(CSV_PATH) as fh:
    rd = _csv.reader(fh)
    next(rd)
    for row in rd:
        for i in range(NUM):
            sp = abs(float(row[2 + i * 4 + 3]))
            if sp > peakspd[i]:
                peakspd[i] = sp

lines = [
    f"PROBE channel={CHANNEL} driven_port={PORT} stick={STICK} frames={FRAMES}",
    wait_summary,
    "",
    cfg,
    "",
    "per-slot over measured drive window (displacement xz | peak speed):",
]
for i in range(NUM):
    mark = "  <== MOVED" if (moved[i] > 5.0 or peakspd[i] > 1.0) else ""
    lines.append(f"  slot {i:2d}: moved {moved[i]:9.2f}  peakspd {peakspd[i]:7.2f}{mark}")
finish("\n".join(lines))
