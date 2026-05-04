# slave.py

import sys
import os
import inspect
from pathlib import Path

script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
script_directory = Path(script_directory)

shared_site_path = script_directory / "shared_site.txt"

# add libraries from your python install (needs to match dolphin version (currently 3.12))
if shared_site_path.exists() and shared_site_path.is_file():
    with open(shared_site_path, 'r', encoding='utf-8') as file:
        site_path = file.read()

    sys.path.append(site_path)

# Dolphin 내장 Python에서는 sys.executable이 제대로 안 잡히므로 .venv의 python을 지정
# sys.executable = str(script_directory / ".venv" / "Scripts" / "python.exe")

from dolphin import event, gui, savestate, memory, controller

# Now we can import other libraries safely
try:
    import time
    import traceback
    import random
    import math
    import numpy as np
    from collections import deque
    from multiprocessing.connection import Client
    from PIL import Image, ImageEnhance
    from multiprocessing import shared_memory
    from copy import deepcopy
except Exception as e:
    print(e)
    raise Exception("stop")

def increment_alive(path='alive.txt'):
    path = script_directory / Path(path)
    alive_num = int(path.read_text().strip()) if path.exists() else 0
    path.write_text(str(alive_num + 1))
    return alive_num

save_states_path = script_directory / "MarioKartSaveStates"

instance_info_folder = script_directory / Path('instance_info')

# Read pid from pid_num.txt
pid = int((instance_info_folder / 'pid_num.txt').read_text().strip())

# Read our specific ID
id = int((instance_info_folder / f'instance_id{pid}.txt').read_text().strip())

# Debuging code
# import debugpy

# debugpy.listen(("localhost", 5678+id))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

# print("Script Started!")

# Write our own PID into script_pid{id}.txt
(instance_info_folder / f'script_pid{id}.txt').write_text(str(os.getpid()))

alive_num = increment_alive()

num_envs = int((instance_info_folder / 'num_envs.txt').read_text().strip())
reset_mode_path = instance_info_folder / "reset_mode.txt"
reset_savestate_path = instance_info_folder / "reset_savestate.txt"
reset_mode = (
    reset_mode_path.read_text().strip()
    if reset_mode_path.exists()
    else "savestate"
)
reset_savestate = (
    reset_savestate_path.read_text().strip()
    if reset_savestate_path.exists()
    else ""
)
episode_timeout_steps_path = instance_info_folder / "episode_timeout_steps.txt"
episode_timeout_steps_text = (
    episode_timeout_steps_path.read_text().strip()
    if episode_timeout_steps_path.exists()
    else ""
)
episode_timeout_steps = (
    int(episode_timeout_steps_text)
    if episode_timeout_steps_text
    else (None if reset_mode == "race_start" else 700)
)

log_path = instance_info_folder / f'slave_{id}.log'
def log_exc(exc: BaseException):
    with open(log_path, 'a') as f:
        f.write('— Exception occurred —\n')
        traceback.print_exc(file=f)
        f.write('\n\n')

def log_diag(msg: str):
    """File-only diagnostic log (Dolphin embedded Python's stdout doesn't reach master)."""
    try:
        with open(log_path, 'a') as f:
            f.write(f"[diag {time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


FILE_PATH = script_directory / "shared_value.txt"

if not FILE_PATH.exists():
    FILE_PATH = script_directory.parent / "shared_value.txt"

print(f"FILE_PATH: {FILE_PATH}")

def get_value() -> float:
    """
    Read a float from shared_value.txt in the current directory.
    If it doesn’t exist yet, create it with INITIAL_VALUE.
    """
    if not FILE_PATH.exists():
        raise Exception("File doesn't exist!")
    return float(FILE_PATH.read_text().strip())

def set_value(new_val: float):
    """
    Overwrite shared_value.txt in the current directory with the given float.
    """
    FILE_PATH.write_text(str(float(new_val)))


# ============= [InputData layout - PAL/RMCP01] =============
# Source: https://github.com/SeekyCt/mkw-structures/blob/master/inputdata.h
#   InputData *sInstance @ 0x809BD70C   (read_u32 to get base)
#   InputData {
#     RealControllerHolder    realControllerHolders[4];     // @ +0x000, each 0xEC
#     VirtualControllerHolder virtualControllerHolders[12]; // @ +0x3B0, each 0x180
#   }
#   ControllerHolder { ...; InputState inputStates[2] @ +0x28 }
#   InputState (size 0x18) {  vtable @ +0x00 (0x808B2F2C); members start at +0x04:
#     u32 vtable        @ +0x00  (DO NOT WRITE)
#     u16 buttonActions @ +0x04  (bit0=accel, bit1=brake, bit2=item, bit3=drift, bit5=rear-view)
#     u16 buttonRaw     @ +0x06
#     f32 stickX        @ +0x08   (-1.0..1.0)
#     f32 stickY        @ +0x0C
#     u8  qStickX       @ +0x10   (0..14)
#     u8  qStickY       @ +0x11
#     u8  motionFlick   @ +0x12   (1=up, 2=down, 3=left, 4=right)
#     u8  motionFlick2  @ +0x13
#     u8  unknown       @ +0x14   (suffix in mkw-structures confirms offset)
#   }
INPUT_DATA_PTR    = 0x809BD70C
# InputData layout: vtable u32 @ +0x00; then realControllerHolders[4] (each 0xEC) at +0x04;
# then virtualControllerHolders[12] (each 0x180) at +0x3B4.
REAL_HOLDERS      = 0x004
REAL_STRIDE       = 0x0EC
VIRTUAL_HOLDERS   = 0x3B4
VIRTUAL_STRIDE    = 0x180
INPUT_STATE_OFF   = 0x28          # inputStates[0] within ControllerHolder
NUM_KARTS         = 12
NUM_REAL          = 4             # only first 4 slots have a realControllerHolder

# Phase-1 probe gate. When set, no input writes happen and per-frame InputState reads
# are dumped to instance_info/input_probe_*.csv for offline correlation analysis.
PROBE_INPUT = os.environ.get("MKW_PROBE_INPUT", "0") == "1"

# Hard kill-switch for episode resets. When MKW_DISABLE_RESET=1, get_reward_terminal_trun
# swallows every terminal/trun event and the savestate reload never fires. Useful for
# GUI inspection ("why does the env die after a few seconds?") and long override tests.
DISABLE_RESET = os.environ.get("MKW_DISABLE_RESET", "0") == "1"


# PlayerSub1c per-kart fields — see SeekyCt/mkw-structures · player.h
#   PlayerSub1c is the DOWNSTREAM input target (PadProxy → updateFromInput →
#   PlayerSub1c → kart physics). Writing to PlayerSub1c[i] bypasses the
#   InputData/PadProxy chain that PadProxy::calc() rewrites every frame.
#
# Layout (size 0xC0):
#   +0x00  vtable u32
#   +0x04  bitfield0 u32  — bit numbering is PowerPC big-endian "MSB-first".
#                           "bit N" in mkw-structures means (1 << (31 - N)).
#                           Verified empirically: writing 0x80000000 (bit 0 MSB) to
#                           bf0 makes kart accelerate; 0x00000001 (LSB) does nothing.
#                           Bits we own:
#                             bit  0 (=0x80000000)  accelerate
#                             bit  1 (=0x40000000)  brake
#                             bit  2 (=0x20000000)  drift input
#                             bit 13 (=0x00040000)  stick left
#                             bit 24 (=0x00000080)  stick right
#   +0x14  bitfield4 u32   (bit0=cpu, bit1=real_local, ...)
#   +0x88  stickX f32
#   +0x8C  stickY f32
#
# Reach via: 0x809C18F8 (PlayerHolder::sInstance) → [0x20, 0x4*i, 0x0, 0x4]
PLAYERSUB1C_BITFIELD0 = 0x04
PLAYERSUB1C_STICKX    = 0x88
PLAYERSUB1C_STICKY    = 0x8C

# PowerPC MSB-from-0 bit values (1 << (31 - N))
PSUB1C_ACCEL_BIT       = 0x80000000  # bit 0
PSUB1C_BRAKE_BIT       = 0x40000000  # bit 1
PSUB1C_DRIFT_BIT       = 0x20000000  # bit 2
PSUB1C_STICKLEFT_BIT   = 0x00040000  # bit 13
PSUB1C_STICKRIGHT_BIT  = 0x00000080  # bit 24
PSUB1C_INPUT_BITS_MASK = (PSUB1C_ACCEL_BIT | PSUB1C_BRAKE_BIT | PSUB1C_DRIFT_BIT
                          | PSUB1C_STICKLEFT_BIT | PSUB1C_STICKRIGHT_BIT)  # 0xE0040080


def resolve_playersub1c(slot):
    """Return the PlayerSub1c* for kart `slot`, or 0 if not resolvable."""
    holder = _safe_read_u32(0x809C18F8)
    players = _safe_read_u32(holder + 0x20) if holder else 0
    player_i = _safe_read_u32(players + 0x4 * slot) if players else 0
    pointers = _safe_read_u32(player_i + 0x0) if player_i else 0
    sub1c = _safe_read_u32(pointers + 0x4) if pointers else 0
    return sub1c


class KartInputWriter:
    """Writes per-kart input to MKW's InputData controller holders.

    Slot mapping discovered empirically:
      * virtualControllerHolders[i] (i ∈ 0..11) drives slots 4..11 (CPU karts).
        For slots 0..3 the game reads from realControllerHolders[i] instead, so
        writing only to virtualControllerHolders has no effect for those slots.
      * For uniform 12-kart override we write to BOTH holders for slots 0..3.

    The InputData base pointer can change after savestate loads; call
    refresh_base() after each savestate.load_from_file().
    """

    def __init__(self):
        self.base = 0
        # Per-slot list of InputState addresses we write each frame. Slots 0..3 get
        # both real and virtual holders; slots 4..11 get only their virtual holder.
        self.state_addrs = [[] for _ in range(NUM_KARTS)]
        self.refresh_base()

    def refresh_base(self):
        self.base = memory.read_u32(INPUT_DATA_PTR)
        for i in range(NUM_KARTS):
            addrs = []
            if i < NUM_REAL:
                addrs.append(self.base + REAL_HOLDERS + i * REAL_STRIDE + INPUT_STATE_OFF)
            addrs.append(self.base + VIRTUAL_HOLDERS + i * VIRTUAL_STRIDE + INPUT_STATE_OFF)
            self.state_addrs[i] = addrs

    def write(self, slot, stickX, buttons, motion_flick, stickY=0.0):
        q = max(0, min(14, int(round(float(stickX) * 7)) + 7))
        b = int(buttons) & 0xFFFF
        sx = float(stickX)
        sy = float(stickY)
        mf = int(motion_flick) & 0xFF
        for s in self.state_addrs[slot]:  # InputState base (vtable @ +0x0)
            memory.write_u16(s + 0x04, b)    # buttonActions
            memory.write_f32(s + 0x08, sx)   # stickX
            memory.write_f32(s + 0x0C, sy)   # stickY
            memory.write_u8(s + 0x10, q)     # quantisedStickX
            memory.write_u8(s + 0x12, mf)    # motionControlFlick

    def read_state(self, slot, which="virtual"):
        """Read InputState at slot for probe / verification.

        which: 'virtual' (default; always present) or 'real' (only for slots 0..3).
        """
        if which == "real":
            if slot >= NUM_REAL:
                return None
            s = self.base + REAL_HOLDERS + slot * REAL_STRIDE + INPUT_STATE_OFF
        else:
            s = self.base + VIRTUAL_HOLDERS + slot * VIRTUAL_STRIDE + INPUT_STATE_OFF
        return {
            "vtable":        memory.read_u32(s + 0x00),
            "buttonActions": memory.read_u16(s + 0x04),
            "buttonRaw":     memory.read_u16(s + 0x06),
            "stickX":        memory.read_f32(s + 0x08),
            "stickY":        memory.read_f32(s + 0x0C),
            "qStickX":       memory.read_u8(s + 0x10),
            "qStickY":       memory.read_u8(s + 0x11),
            "motionFlick":   memory.read_u8(s + 0x12),
            "motionFlick2":  memory.read_u8(s + 0x13),
        }


def _safe_read_u32(addr):
    if not addr or addr < 0x80000000 or addr >= 0x81800000:
        return 0
    try:
        return memory.read_u32(addr)
    except Exception:
        return 0


def _safe_read_f32(addr):
    if not addr or addr < 0x80000000 or addr >= 0x81800000:
        return 0.0
    try:
        return float(memory.read_f32(addr))
    except Exception:
        return 0.0


def _safe_read_u8(addr):
    if not addr or addr < 0x80000000 or addr >= 0x81800000:
        return 0
    try:
        return int(memory.read_u8(addr))
    except Exception:
        return 0


def _safe_read_u16(addr):
    if not addr or addr < 0x80000000 or addr >= 0x81800000:
        return 0
    try:
        return int(memory.read_u16(addr))
    except Exception:
        return 0


def _safe_read_u32_any(addr):
    """Like _safe_read_u32 but accepts MEM2 too (for race-manager pointers)."""
    if not addr or not (
        (0x80000000 <= addr < 0x81800000) or (0x90000000 <= addr < 0x94000000)
    ):
        return 0
    try:
        return memory.read_u32(addr)
    except Exception:
        return 0


def _lite_per_kart_obs(n, karr, plr_arr, race_mgr_for_player):
    """Return the 78-element per-kart observation slice for slot n via direct
    chain reads (no Memory.Addresses cache).

    Field layout matches Memory.get_obs for tracked slots so the 941-wide obs
    stays aligned. Each field falls back to 0 if its chain can't be resolved.
    Per-frame cost is ~30 chain reads; safe in practice because each frame
    yields back to the emulator via `await event.frameadvance()` between calls.

    Args:
      karr     -- PlayerHolder players[] array base (read once per get_obs)
      plr_arr  -- RaceManagerPlayer per-player array base (read once per get_obs)
      race_mgr_for_player -- RaceManager (0x809BD728 → +0x28) per-player struct base
    """
    out = [0.0] * 78
    out[0] = float(n)  # PlayerID

    # ----- RaceManager player slot (race-config: kart, character, lap counts) -----
    # 0x809BD728 → +(0x28 + 0xF0*n) is start of RacedataPlayer (per Vlab-WiiRL chains).
    if race_mgr_for_player:
        slot_base = race_mgr_for_player + 0xF0 * n
        out[1] = float(_safe_read_u8(slot_base + 0x05))   # LocalPlayerNum
        out[2] = float(_safe_read_u8(slot_base + 0x06))   # RealControllerID
        out[3] = float(_safe_read_u32_any(slot_base + 0x08))  # KartID
        out[4] = float(_safe_read_u32_any(slot_base + 0x0C))  # CharacterID

    # ----- RaceManagerPlayer per-player block (race-time progress) -----
    if plr_arr:
        plr = _safe_read_u32(plr_arr + 0x4 * n)
        if plr:
            out[5]  = _safe_read_f32(plr + 0xC)            # RaceCompletion
            out[6]  = _safe_read_f32(plr + 0x10)           # MaxRaceCompletion
            out[7]  = _safe_read_f32(plr + 0x14)           # FirstKcpLapCompletion
            out[8]  = _safe_read_f32(plr + 0x18)           # NextCheckpointLapCompletion
            out[9]  = _safe_read_f32(plr + 0x1C)           # NextCheckpointLapCompletionMax
            out[10] = float(_safe_read_u16(plr + 0x24))    # currentLap
            out[11] = float(_safe_read_u8(plr + 0x26))     # MaxLap
            out[12] = float(_safe_read_u8(plr + 0x27))     # currentKCP
            out[13] = float(_safe_read_u8(plr + 0x28))     # maxKCP
            out[75] = float(_safe_read_u8(plr + 0x3B))     # StateBit

    # ----- KartObject chain (physics) -----
    if karr:
        kp = _safe_read_u32(karr + 0x4 * n)
        ko = _safe_read_u32(kp) if kp else 0
        if ko:
            # KartDynamics container @ +0x8 → +0x90 → +0x4 = inner KartDynamics
            kdc = _safe_read_u32(ko + 0x8)
            kdyn = _safe_read_u32(kdc + 0x90) if kdc else 0
            if kdyn:
                inner = _safe_read_u32(kdyn + 0x4)
                # position f32×3 @ kdyn+0x18 (NOT inner+0x18 — Vlab chains differ)
                out[16] = _safe_read_f32(kdyn + 0x18 + 0)
                out[17] = _safe_read_f32(kdyn + 0x18 + 4)
                out[18] = _safe_read_f32(kdyn + 0x18 + 8)
                if inner:
                    # velocity, internalVelocity, externalVelocity, angularVelocity,
                    # acceleration, mainRotation all live inside `inner`.
                    out[19] = _safe_read_f32(inner + 0xD4 + 0)   # velocity x
                    out[20] = _safe_read_f32(inner + 0xD4 + 4)
                    out[21] = _safe_read_f32(inner + 0xD4 + 8)
                    out[22] = _safe_read_f32(inner + 0x14C + 0)  # internalVelocity
                    out[23] = _safe_read_f32(inner + 0x14C + 4)
                    out[24] = _safe_read_f32(inner + 0x14C + 8)
                    out[25] = _safe_read_f32(inner + 0x74 + 0)   # externalVelocity
                    out[26] = _safe_read_f32(inner + 0x74 + 4)
                    out[27] = _safe_read_f32(inner + 0x74 + 8)
                    out[28] = _safe_read_f32(inner + 0xA4 + 0)   # angularVelocity
                    out[29] = _safe_read_f32(inner + 0xA4 + 4)
                    out[30] = _safe_read_f32(inner + 0xA4 + 8)
                    out[31] = _safe_read_f32(inner + 0x80 + 0)   # acceleration
                    out[32] = _safe_read_f32(inner + 0x80 + 4)
                    out[33] = _safe_read_f32(inner + 0x80 + 8)
                    out[34] = _safe_read_f32(inner + 0xF0 + 0)   # mainRotation x
                    out[35] = _safe_read_f32(inner + 0xF0 + 4)
                    out[36] = _safe_read_f32(inner + 0xF0 + 8)
                    out[37] = _safe_read_f32(inner + 0xF0 + 12)
            # KartState @ ko + 0x4 — bitfields, airtime, start-boost
            kstate = _safe_read_u32(ko + 0x4)
            if kstate:
                out[47] = float(_safe_read_u32(kstate + 0x4))    # BitField0
                out[48] = float(_safe_read_u32(kstate + 0x8))    # BitField1
                out[49] = float(_safe_read_u32(kstate + 0xC))    # bitfield2
                out[50] = float(_safe_read_u32(kstate + 0x10))   # BitField3
                out[58] = float(_safe_read_u16(kstate + 0xA6))   # trickableTimer
                out[60] = float(_safe_read_u32(kstate + 0x1C))   # airtime
                out[76] = _safe_read_f32(kstate + 0x9C)          # startBoostCharge
                out[77] = float(_safe_read_u32(kstate + 0xA0))   # startBoostIdx
            # KartCollide @ ko + 0x18 — surface, collision, race_pos, respawn
            kcoll = _safe_read_u32(ko + 0x18)
            if kcoll:
                inner_coll = _safe_read_u32(kcoll + 0x18)
                if inner_coll:
                    out[51] = float(_safe_read_u32(inner_coll + 0x2C))   # surfaceFlags
                    out[63] = float(_safe_read_u16(inner_coll + 0x48))   # respawn_timer
                out[61] = float(_safe_read_u8(kcoll + 0x3C))     # race_position
                out[62] = float(_safe_read_u16(kcoll + 0x40))    # floor_collision_count
                out[64] = float(_safe_read_u32(kcoll + 0x18 + 0x8) if _safe_read_u32(kcoll + 0x18) else 0)  # wall_collide
            # KartMove @ ko + 0x28 — speed, drift, wheelie, hop, status timers
            kmove = _safe_read_u32(ko + 0x28)
            if kmove:
                out[14] = _safe_read_f32(kmove + 0x18)           # SoftSpeedLimit
                out[15] = _safe_read_f32(kmove + 0x2C)           # HardSpeedLimit
                out[38] = _safe_read_f32(kmove + 0x20)           # speed
                out[39] = _safe_read_f32(kmove + 0x30)           # acceleration_KartMove
                out[43] = float(_safe_read_u16(kmove + 0x148))   # offroadInvincibility
                out[44] = float(_safe_read_u32(kmove + 0x2A8))   # wheelieFrames
                out[45] = float(_safe_read_u16(kmove + 0x2B6))   # wheelieCooldown
                out[46] = _safe_read_f32(kmove + 0x294)          # leanRot
                out[52] = _safe_read_f32(kmove + 0x228 + 0)      # HopVector
                out[53] = _safe_read_f32(kmove + 0x228 + 4)
                out[54] = _safe_read_f32(kmove + 0x228 + 8)
                out[55] = float(_safe_read_u16(kmove + 0x102))   # mt_boost_timer
                out[56] = float(_safe_read_u16(kmove + 0x10C))   # allmt
                out[57] = float(_safe_read_u16(kmove + 0x110))   # mush_and_boost
                out[69] = float(_safe_read_u16(kmove + 0x18A))   # StarTimer
                out[70] = float(_safe_read_u16(kmove + 0x18C))   # ShockTimer
                out[71] = float(_safe_read_u16(kmove + 0x18E))   # BlooperInkTimer
                out[72] = float(_safe_read_u8 (kmove + 0x190))   # BlooperStateFlag
                out[73] = float(_safe_read_u16(kmove + 0x192))   # CrushTimer
                out[74] = float(_safe_read_u16(kmove + 0x194))   # MegaTimer
                # trick_cooldown via KartJump @ kmove+0x258
                kjump = _safe_read_u32(kmove + 0x258)
                if kjump:
                    out[59] = float(_safe_read_u16(kjump + 0x38))  # trick_cooldown
            # Misc @ kp + 0x44 — drift state, miniturbo. Heavy chain
            # `[0x20, 0x4*i, 0x44, 0xFC]` skips the `0x0` step (kart_obj deref)
            # because KartMisc is hung off `kart_ptr`, not `kart_obj`.
            # Reading `ko + 0x44` (the previous code) lands at a sibling field
            # inside KartObject — DriftState came back as ~17000-18000 garbage.
            kmisc = _safe_read_u32(kp + 0x44) if kp else 0
            if kmisc:
                out[40] = float(_safe_read_u16(kmisc + 0xFC))    # DriftState
                out[41] = float(_safe_read_u16(kmisc + 0xFE))    # miniturboCharge
                out[42] = float(_safe_read_u16(kmisc + 0x100))   # SMiniturboCharge

    # ----- ItemManager per-kart KartItem @ 0x809C3618 -----
    # Authoritative layout (mkw-sp symbols.txt + KartItem.hh + ItemInventory.hh):
    #   *(0x809C3618) = Item::ItemManager::s_instance (size 0x430)
    #   ItemManager + 0x14 = KartItem* (CONTIGUOUS struct array, stride 0x248)
    #   KartItem[i] is at base + 0x248 * i  (NOT base + 0x4*i — that bug made
    #   slots 1..11 read garbage from inside KartItem[0])
    #   KartItem[i] + 0x88 = ItemInventory; +0x04 m_currentItemID, +0x08 m_currentItemCount
    #   ⇒ Item ID at KartItem[i] + 0x8C (s32, -1=None, 0..18=items, 19/20=Unused/NoItem)
    #   ⇒ Item count at KartItem[i] + 0x90 (s32)
    try:
        item_mgr = _safe_read_u32(0x809C3618)
        kitem_base = _safe_read_u32(item_mgr + 0x14) if item_mgr else 0
        if kitem_base:
            kitem = kitem_base + 0x248 * n
            out[65] = float(_safe_read_u32(kitem + 0x8C))   # Item ID (-1..20)
            out[66] = float(_safe_read_u32(kitem + 0x90))   # Item count
    except Exception:
        pass

    return out


class Memory:
    class Addresses:
        def __init__(self, num_players):
            # ================= [RACE INFO Addresses] =================
            self.stage = self.resolve_address(0x809BD730, [0x28])
            self.countdownTimer = self.resolve_address(0x809BD730, [0x22])
            self.FrameCount = self.resolve_address(0x809BD730, [0x20])
            self.PlayerCount_addr = 0x809C38B8
            self.CourseID = self.resolve_address(0x809BD728, [0xB68])
            self.EngineClass = self.resolve_address(0x809BD728, [0xB6C])

            # ================= [PLAYER INFO Addresses (리스트로 확장)] =================
            self.RaceCompletion = []
            self.currentLap = []
            self.position = []
            self.acceleration_KartDynamics = []
            self.mainRotation = []
            self.internalVelocity = []
            self.externalVelocity = []
            self.angularVelocity = []
            self.velocity = []
            self.speed = []
            self.acceleration_KartMove = []
            self.miniturboCharge = []
            self.offroadInvincibility = []
            self.wheelieFrames = []
            self.wheelieCooldown = []
            self.leanRot = []
            self.bitfield2 = []
            self.surfaceFlags = []
            self.mushroomCount = []
            self.hopPos = []
            self.mt_boost_timer = []
            self.airtime = []
            self.allmt = []
            self.mush_and_boost = []
            self.floor_collision_count = []
            self.race_position = []
            self.respawn_timer = []
            self.wall_collide = []
            self.soft_speed_limit = []
            self.trickableTimer = []
            self.trick_cooldown = []
            self.LocalPlayerNum = []
            self.RealControllerID = []
            self.KartID = []
            self.CharacterID = []
            self.MaxRaceCompletion = []
            self.FirstKcpLapCompletion = []
            self.NextCheckpointLapCompletion = []
            self.NextCheckpointLapCompletionMax = []
            self.MaxLap = []
            self.currentKCP = []
            self.maxKCP = []
            self.StateBit = []
            self.HardSpeedLimit = []
            self.DriftState = []
            self.SMiniturboCharge = []
            self.BitField0 = []
            self.BitField1 = []
            self.BitField3 = []
            self.HopVector = []
            self.Item = []
            self.ItemNum = []
            self.PassiveItem = []
            self.PassiveItemNum = []
            self.StarTimer = []
            self.ShockTimer = []
            self.BlooperInkTimer = []
            self.BlooperStateFlag = []
            self.CrushTimer = []
            self.MegaTimer = []
            self.startBoostCharge = []
            self.startBoostIdx = []

            for i in range(num_players):
                try:
                    log_diag(f"Memory.Addresses: resolving slot {i}")
                except Exception:
                    pass
                # RaceManagerPlayer
                self.RaceCompletion.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0xC]))
                self.currentLap.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x24]))
                self.MaxRaceCompletion.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x10]))
                self.FirstKcpLapCompletion.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x14]))
                self.NextCheckpointLapCompletion.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x18]))
                self.NextCheckpointLapCompletionMax.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x1C]))
                self.MaxLap.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x26]))
                self.currentKCP.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x27]))
                self.maxKCP.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x28]))
                self.StateBit.append(self.resolve_address(0x809BD730, [0xC, 0x4 * i, 0x3B]))

                # RaceManager
                self.LocalPlayerNum.append(self.resolve_address(0x809BD728, [0x2D + 0xF0 * i]))
                self.RealControllerID.append(self.resolve_address(0x809BD728, [0x2E + 0xF0 * i]))
                self.KartID.append(self.resolve_address(0x809BD728, [0x30 + 0xF0 * i]))
                self.CharacterID.append(self.resolve_address(0x809BD728, [0x34 + 0xF0 * i]))

                # KartDynamics
                self.position.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x18]))
                self.acceleration_KartDynamics.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x4, 0x80]))
                self.mainRotation.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x4, 0xF0]))
                self.internalVelocity.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x4, 0x14C]))
                self.externalVelocity.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x4, 0x74]))
                self.angularVelocity.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x4, 0xA4]))
                self.velocity.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x4, 0xD4]))
                self.wall_collide.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x8, 0x90, 0x8, 0x8]))

                # KartMove
                self.speed.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x20]))
                self.acceleration_KartMove.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x30]))
                self.offroadInvincibility.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x148]))
                self.wheelieFrames.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x2A8]))
                self.wheelieCooldown.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x2B6]))
                self.leanRot.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x294]))
                self.mt_boost_timer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x102]))
                self.allmt.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x10C]))
                self.mush_and_boost.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x110]))
                self.soft_speed_limit.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x18]))
                self.HardSpeedLimit.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x2C]))
                self.trick_cooldown.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x258, 0x38]))
                self.HopVector.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x228]))
                self.StarTimer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x18A]))
                self.ShockTimer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x18C]))
                self.BlooperInkTimer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x18E]))
                self.BlooperStateFlag.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x190]))
                self.CrushTimer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x192]))
                self.MegaTimer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x28, 0x194]))

                # KartState
                self.bitfield2.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0xC]))
                self.airtime.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0x1C]))
                self.trickableTimer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0xA6]))
                self.BitField0.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0x4]))
                self.BitField1.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0x8]))
                self.BitField3.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0x10]))
                self.startBoostCharge.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0x9C]))
                self.startBoostIdx.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x4, 0xA0]))

                # KartCollide
                self.surfaceFlags.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x18, 0x18, 0x2C]))
                self.floor_collision_count.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x18, 0x40]))
                self.race_position.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x18, 0x3C]))
                self.respawn_timer.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x0, 0x18, 0x18, 0x48]))

                # Misc
                self.miniturboCharge.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x44, 0xFE]))
                self.SMiniturboCharge.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x44, 0x100]))
                self.DriftState.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x44, 0xFC]))
                self.hopPos.append(self.resolve_address(0x809C18F8, [0x20, 0x4 * i, 0x44, 0x22C]))
                # Item chain: ItemManager.kartItems is a CONTIGUOUS KartItem array
                # (stride 0x248), not a pointer array. resolve_address can't express
                # "no inner deref", so hand-resolve. PassiveItem* fields at +0xCC/+0x104
                # were unverified (inside KartItem opaque tail) — dropped.
                _item_mgr = memory.read_u32(0x809C3618)
                _kitem_base = memory.read_u32(_item_mgr + 0x14) if _item_mgr else 0
                _kitem_i = (_kitem_base + 0x248 * i) if _kitem_base else 0
                self.mushroomCount.append(_kitem_i + 0x90 if _kitem_i else 0)  # alias for ItemNum
                self.Item.append(_kitem_i + 0x8C if _kitem_i else 0)
                self.ItemNum.append(_kitem_i + 0x90 if _kitem_i else 0)
                self.PassiveItem.append(0)
                self.PassiveItemNum.append(0)

        def resolve_address(self, base_address, offsets):
            # Defensive: bounds-check every dereference. PowerPC valid memory is
            # MEM1 [0x80000000, 0x81800000) or MEM2 [0x90000000, 0x94000000).
            # Without the check, dereferencing an invalid intermediate pointer can
            # hang Dolphin's emulator (observed at slot 4 with play_num=12).
            current_address = memory.read_u32(base_address)
            for offset in offsets:
                addr = current_address + offset
                value_address = addr
                in_mem = (0x80000000 <= addr < 0x81800000) or (0x90000000 <= addr < 0x94000000)
                if not in_mem:
                    return 0
                try:
                    current_address = memory.read_u32(addr)
                except Exception:
                    return 0
            return value_address

    def __init__(self, num_players=1):
        self.num_players = num_players
        self.addresses = self.Addresses(num_players)

        # ================= [RACE INFO] =================
        self.stage: int = 0
        self.countdownTimer: int = 0
        self.FrameCount: int = 0
        self.PlayerCount: int = 0
        self.CourseID: int = 0
        self.EngineClass: int = 0

        # ================= [PLAYER INFO (Lists)] =================
        self.RaceCompletion = [0.0] * num_players
        self.currentLap = [0] * num_players
        
        self.position = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.acceleration_KartDynamics = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.mainRotation = [np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.mainRotationEuler = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.internalVelocity = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.externalVelocity = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.angularVelocity = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.velocity = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        self.HopVector = [np.array([0.0, 0.0, 0.0]) for _ in range(num_players)]
        
        self.speed = [0.0] * num_players
        self.acceleration_KartMove = [0.0] * num_players
        self.miniturboCharge = [0] * num_players
        self.offroadInvincibility = [0] * num_players
        self.wheelieFrames = [0] * num_players
        self.wheelieCooldown = [0] * num_players
        self.leanRot = [0.0] * num_players

        self.bitfield2 = [0] * num_players
        self.isWheelie = [False] * num_players

        self.surfaceFlags = [0] * num_players
        self.isAboveOffroad = [False] * num_players
        self.isTouchingOffroad = [False] * num_players

        self.mushroomCount = [0] * num_players
        self.hopPos = [0.0] * num_players
        # 기존 코드 유지용 추가 변수들
        self.oilSpillRadians = [0.0] * num_players
        self.oilSpillDistance = [0.0] * num_players
        self.boostPanelRadians = [0.0] * num_players
        self.boostPanelDistance = [0.0] * num_players

        self.mt_boost_timer = [0] * num_players
        self.airtime = [0] * num_players
        self.allmt = [0] * num_players
        self.mush_and_boost = [0] * num_players
        self.floor_collision_count = [2] * num_players
        self.race_position = [12] * num_players
        self.respawn_timer = [0] * num_players
        self.wall_collide = [0] * num_players
        self.speed_limit = [100.0] * num_players
        self.trickableTimer = [0] * num_players
        self.trick_cooldown = [0] * num_players

        self.LocalPlayerNum = [0] * num_players
        self.RealControllerID = [0] * num_players
        self.KartID = [0] * num_players
        self.CharacterID = [0] * num_players
        
        self.MaxRaceCompletion = [0.0] * num_players
        self.FirstKcpLapCompletion = [0.0] * num_players
        self.NextCheckpointLapCompletion = [0.0] * num_players
        self.NextCheckpointLapCompletionMax = [0.0] * num_players
        
        self.MaxLap = [0] * num_players
        self.currentKCP = [0] * num_players
        self.maxKCP = [0] * num_players
        self.StateBit = [0] * num_players
        self.HardSpeedLimit = [0.0] * num_players
        
        self.DriftState = [0] * num_players
        self.SMiniturboCharge = [0] * num_players
        
        self.BitField0 = [0] * num_players
        self.BitField1 = [0] * num_players
        self.BitField3 = [0] * num_players
        
        self.Item = [0] * num_players
        self.ItemNum = [0] * num_players
        self.PassiveItem = [0] * num_players
        self.PassiveItemNum = [0] * num_players
        
        self.StarTimer = [0] * num_players
        self.ShockTimer = [0] * num_players
        self.BlooperInkTimer = [0] * num_players
        self.BlooperStateFlag = [0] * num_players
        self.CrushTimer = [0] * num_players
        self.MegaTimer = [0] * num_players
        
        self.startBoostCharge = [0.0] * num_players
        self.startBoostIdx = [0] * num_players

    def update(self):
        # RACE INFO
        self.stage = memory.read_u32(self.addresses.stage)
        self.countdownTimer = memory.read_u16(self.addresses.countdownTimer)
        self.FrameCount = memory.read_u32(self.addresses.FrameCount)
        self.PlayerCount = memory.read_u8(self.addresses.PlayerCount_addr)
        self.CourseID = memory.read_u32(self.addresses.CourseID)
        self.EngineClass = memory.read_u32(self.addresses.EngineClass)
        
        # PLAYER INFO
        for i in range(self.num_players):
            self.RaceCompletion[i] = memory.read_f32(self.addresses.RaceCompletion[i])
            self.currentLap[i] = memory.read_u16(self.addresses.currentLap[i])
            self.MaxRaceCompletion[i] = memory.read_f32(self.addresses.MaxRaceCompletion[i])
            self.FirstKcpLapCompletion[i] = memory.read_f32(self.addresses.FirstKcpLapCompletion[i])
            self.NextCheckpointLapCompletion[i] = memory.read_f32(self.addresses.NextCheckpointLapCompletion[i])
            self.NextCheckpointLapCompletionMax[i] = memory.read_f32(self.addresses.NextCheckpointLapCompletionMax[i])
            
            self.MaxLap[i] = memory.read_u8(self.addresses.MaxLap[i])
            self.currentKCP[i] = memory.read_u8(self.addresses.currentKCP[i])
            self.maxKCP[i] = memory.read_u8(self.addresses.maxKCP[i])
            self.StateBit[i] = memory.read_u8(self.addresses.StateBit[i])

            self.LocalPlayerNum[i] = memory.read_u8(self.addresses.LocalPlayerNum[i])
            self.RealControllerID[i] = memory.read_u8(self.addresses.RealControllerID[i])
            self.KartID[i] = memory.read_u32(self.addresses.KartID[i])
            self.CharacterID[i] = memory.read_u32(self.addresses.CharacterID[i])

            # 3D Vectors
            for j in range(3):
                self.position[i][j] = memory.read_f32(self.addresses.position[i] + j*4)
                self.acceleration_KartDynamics[i][j] = memory.read_f32(self.addresses.acceleration_KartDynamics[i] + j*4)
                self.internalVelocity[i][j] = memory.read_f32(self.addresses.internalVelocity[i] + j*4)
                self.externalVelocity[i][j] = memory.read_f32(self.addresses.externalVelocity[i] + j*4)
                self.angularVelocity[i][j] = memory.read_f32(self.addresses.angularVelocity[i] + j*4)
                self.velocity[i][j] = memory.read_f32(self.addresses.velocity[i] + j*4)
                self.HopVector[i][j] = memory.read_f32(self.addresses.HopVector[i] + j*4)
            
            # Quaternion & Euler
            for j in range(4):
                self.mainRotation[i][j] = memory.read_f32(self.addresses.mainRotation[i] + j*4)
            self.mainRotationEuler[i] = self.Quat2Euler(self.mainRotation[i])

            self.wall_collide[i] = memory.read_u32(self.addresses.wall_collide[i])
            
            self.speed[i] = memory.read_f32(self.addresses.speed[i])
            self.acceleration_KartMove[i] = memory.read_f32(self.addresses.acceleration_KartMove[i])
            self.offroadInvincibility[i] = memory.read_u16(self.addresses.offroadInvincibility[i])
            self.wheelieFrames[i] = memory.read_u32(self.addresses.wheelieFrames[i])
            self.wheelieCooldown[i] = memory.read_u16(self.addresses.wheelieCooldown[i])
            self.leanRot[i] = memory.read_f32(self.addresses.leanRot[i])
            
            self.mt_boost_timer[i] = memory.read_u16(self.addresses.mt_boost_timer[i])
            self.allmt[i] = memory.read_u16(self.addresses.allmt[i])
            self.mush_and_boost[i] = memory.read_u16(self.addresses.mush_and_boost[i])
            self.speed_limit[i] = memory.read_f32(self.addresses.soft_speed_limit[i])
            self.HardSpeedLimit[i] = memory.read_f32(self.addresses.HardSpeedLimit[i])
            self.trick_cooldown[i] = memory.read_u16(self.addresses.trick_cooldown[i])

            self.bitfield2[i] = memory.read_u32(self.addresses.bitfield2[i])
            self.isWheelie[i] = (self.bitfield2[i] & (1 << 31)) != 0
            
            self.airtime[i] = memory.read_u32(self.addresses.airtime[i])
            self.trickableTimer[i] = memory.read_u16(self.addresses.trickableTimer[i])
            self.BitField0[i] = memory.read_u32(self.addresses.BitField0[i])
            self.BitField1[i] = memory.read_u32(self.addresses.BitField1[i])
            self.BitField3[i] = memory.read_u32(self.addresses.BitField3[i])
            
            self.startBoostCharge[i] = memory.read_f32(self.addresses.startBoostCharge[i])
            self.startBoostIdx[i] = memory.read_u32(self.addresses.startBoostIdx[i])

            self.surfaceFlags[i] = memory.read_u32(self.addresses.surfaceFlags[i])
            self.isTouchingOffroad[i] = (self.surfaceFlags[i] & (1 << (7 - 1))) != 0
            
            self.floor_collision_count[i] = memory.read_u16(self.addresses.floor_collision_count[i])
            self.race_position[i] = memory.read_u8(self.addresses.race_position[i])
            self.respawn_timer[i] = memory.read_u16(self.addresses.respawn_timer[i])

            self.miniturboCharge[i] = memory.read_u16(self.addresses.miniturboCharge[i])
            self.SMiniturboCharge[i] = memory.read_u16(self.addresses.SMiniturboCharge[i])
            self.DriftState[i] = memory.read_u16(self.addresses.DriftState[i])
            self.hopPos[i] = memory.read_f32(self.addresses.hopPos[i])
            
            # Item-related addresses may be 0 if ItemManager isn't fully constructed
            # yet (e.g., during early countdown frames). _safe_read_u32 returns 0 for
            # null/out-of-range — safer than a raw memory.read_u32 which can throw.
            self.mushroomCount[i] = _safe_read_u32(self.addresses.mushroomCount[i])
            self.Item[i] = _safe_read_u32(self.addresses.Item[i])
            self.ItemNum[i] = _safe_read_u32(self.addresses.ItemNum[i])
            self.PassiveItem[i] = _safe_read_u32(self.addresses.PassiveItem[i])
            self.PassiveItemNum[i] = _safe_read_u32(self.addresses.PassiveItemNum[i])

            self.StarTimer[i] = memory.read_u16(self.addresses.StarTimer[i])
            self.ShockTimer[i] = memory.read_u16(self.addresses.ShockTimer[i])
            self.BlooperInkTimer[i] = memory.read_u16(self.addresses.BlooperInkTimer[i])
            self.BlooperStateFlag[i] = memory.read_u8(self.addresses.BlooperStateFlag[i])
            self.CrushTimer[i] = memory.read_u16(self.addresses.CrushTimer[i])
            self.MegaTimer[i] = memory.read_u16(self.addresses.MegaTimer[i])

    def get_obs(self):
        """Return a fixed-width observation vector of size 5 + 78 * NUM_KARTS = 941.

        Slots within Memory(play_num=N) (i.e., n < num_players) carry the full 78-dim
        per-kart slice from the heavyweight tracker. Slots num_players..NUM_KARTS-1
        are populated on-the-fly via `_lite_per_kart_obs` (RaceCompletion, position,
        velocity, race_position, currentLap; rest zero-padded). This keeps the obs
        width aligned with the master's shared-memory shape regardless of MKW_PLAY_NUM.
        """
        obs = []

        # 1. RACE_INFO
        race_info = (
            self.stage,
            self.FrameCount,
            self.PlayerCount,
            self.CourseID,
            self.EngineClass
        )
        obs.extend(race_info)

        # Pre-resolve once per get_obs() so the lite reader doesn't redo them per slot.
        try:
            mgr = memory.read_u32(0x809C18F8)
            karr = memory.read_u32(mgr + 0x20) if mgr else 0
        except Exception:
            karr = 0
        try:
            rmp = memory.read_u32(0x809BD730)
            plr_arr = memory.read_u32(rmp + 0xC) if rmp else 0
        except Exception:
            plr_arr = 0
        try:
            # RaceManager → +0x28 = per-player block start (matches Vlab-WiiRL chains).
            race_data = memory.read_u32(0x809BD728)
            race_mgr_player = race_data + 0x28 if race_data else 0
        except Exception:
            race_mgr_player = 0

        # 2. PER-KART INFO — iterate ALL 12 slots; tracked vs lite per slot.
        for n in range(NUM_KARTS):
            if n >= self.num_players:
                obs.extend(_lite_per_kart_obs(n, karr, plr_arr, race_mgr_player))
                continue
            p_info = (
                n,                          # PlayerID
                self.LocalPlayerNum[n],
                self.RealControllerID[n],
                self.KartID[n],
                self.CharacterID[n],
                
                self.RaceCompletion[n],
                self.MaxRaceCompletion[n],
                self.FirstKcpLapCompletion[n],
                self.NextCheckpointLapCompletion[n],
                self.NextCheckpointLapCompletionMax[n],
                
                self.currentLap[n],
                self.MaxLap[n],
                
                self.currentKCP[n],
                self.maxKCP[n],
                
                self.speed_limit[n],        # SoftSpeedLimit
                self.HardSpeedLimit[n],
                
                *self.position[n],
                *self.velocity[n],
                *self.internalVelocity[n],
                *self.externalVelocity[n],
                *self.angularVelocity[n],
                *self.acceleration_KartDynamics[n],
                *self.mainRotation[n],
                
                self.speed[n],
                self.acceleration_KartMove[n],
                
                self.DriftState[n],
                self.miniturboCharge[n],
                self.SMiniturboCharge[n],
                
                self.offroadInvincibility[n],
                
                self.wheelieFrames[n],
                self.wheelieCooldown[n],
                self.leanRot[n],
                
                self.BitField0[n],
                self.BitField1[n],
                self.bitfield2[n],          # BitField2
                self.BitField3[n],
                
                self.surfaceFlags[n],
                
                *self.HopVector[n],         # HopVelY, HopPosY, HopGravity
                
                self.mt_boost_timer[n],
                self.allmt[n],
                self.mush_and_boost[n],
                
                self.trickableTimer[n],
                self.trick_cooldown[n],
                self.airtime[n],
                
                self.race_position[n],
                self.floor_collision_count[n],
                self.respawn_timer[n],
                
                self.wall_collide[n],
                
                self.Item[n],
                self.ItemNum[n],
                self.PassiveItem[n],
                self.PassiveItemNum[n],
                
                self.StarTimer[n],
                self.ShockTimer[n],
                self.BlooperInkTimer[n],
                self.BlooperStateFlag[n],
                self.CrushTimer[n],
                self.MegaTimer[n],
                
                self.StateBit[n],
                
                self.startBoostCharge[n],
                self.startBoostIdx[n]
            )
            obs.extend(p_info)
            
        return obs

    @staticmethod
    def Quat2Euler(quaternion):

        x, y, w, z = quaternion

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])

class DolphinInstance:
    def __init__(self, id, play_num):

        address = ('localhost', 26330 + id)
        print(f"Connecting to master at {address}...")
        self.conn = Client(address, authkey=b'secret password')
        print("Connected to master!")

        self.play_num = play_num
        # obs_shape is always 5 + 78 * NUM_KARTS so master/slave shared-memory
        # strides match. Memory(play_num=N) only tracks N slots; get_obs() pads
        # the rest via _lite_per_kart_obs(...) to keep the width fixed.
        self.obs_shape = 5 + 78 * NUM_KARTS

        self.bestL1 = 999999
        self.bestL2 = 999999
        self.best_time = get_value()
        self.save_idx = 10

        self.reset_frame_buffer = False
        self.prev_wall_collide = 0

        self.env_id = id

        self.framestack = 4
        self.frameskip = 4
        self.frames = deque([], maxlen=self.framestack) #frame buffer for framestacking
        self.num_envs = num_envs
        print(f"Num envs{self.num_envs}")

        try:
            # setup shared memory
            if(sys.version_info[1] < 13):
                self.shm = shared_memory.SharedMemory(name="states_shm",size=self.num_envs*self.framestack*self.obs_shape)
            else:
                # Make sure that the shared memory doesn't get deleted on upon exiting script by setting track=False
                self.shm = shared_memory.SharedMemory(name="states_shm", track=False, size=self.num_envs*self.framestack*self.obs_shape)
            self.states = np.ndarray(
                (self.num_envs, self.framestack, self.obs_shape),
                dtype=np.float32,
                buffer=self.shm.buf
            )
        except Exception as e:
            print(e)
            print("Error when creating shared memory")

        self.define_action_space()
        # current per-kart action list, updated by recieve_actions(); driven into the
        # input struct each frame inside event.on_frameadvance.
        self.applied_actions = [0] * NUM_KARTS
        # Probe-mode CSV writer for Phase-1 verification (gated by MKW_PROBE_INPUT=1).
        self._probe_csv_path = (
            instance_info_folder / f"input_probe_env{id}.csv" if PROBE_INPUT else None
        )
        self._probe_frame = 0
        self.input_writer = None  # initialized in reset() once the savestate is loaded
        self.reset()

    def _list_save_states(self):
        return sorted(
            [
                file
                for file in Path(save_states_path).rglob("*")
                if file.is_file() and ".s" in file.name
            ]
        )

    def _select_reset_savestate(self) -> Path:
        save_states = self._list_save_states()
        if not save_states:
            raise FileNotFoundError(f"No savestates found in {save_states_path}")

        if reset_savestate:
            requested_path = Path(reset_savestate)
            if not requested_path.is_absolute():
                requested_path = script_directory / requested_path
            if not requested_path.exists():
                raise FileNotFoundError(
                    f"Configured reset savestate does not exist: {requested_path}"
                )
            return requested_path

        if reset_mode == "race_start":
            return save_states[0]

        return random.choice(save_states)

    def define_action_space(self):

        self.wii_dic = {
            "Left": False, "Right": False, "Down": False,
            "Up": False, "Z": False, "R": False, "L": False,
            "A": True, "B": False, "X": False, "Y": False,
            "Start": False, "StickX": 0, "StickY": 0, "CStickX": 0,
            "CStickY": 0, "TriggerLeft": 0, "TriggerRight": 0,
            "AnalogA": 0, "AnalogB": 0, "Connected": True
        }

        # Define discrete action values
        # Aligned with kinoko-env/Vlab-WiiRL: acceleration ALWAYS held (A=True),
        # Up "trick" ALWAYS held too (up_values=[True] → n_actions = 20).
        # This matches the trained PPO checkpoint's action_dim=20.
        self.stickX_values = [-1, -0.4, 0, 0.4, 1]
        self.r_values = [False, True]
        self.up_values = [True]            # always trigger trick when airborne
        self.l_values = [False, True]
        # n_actions = 5 × 2 × 1 × 2 = 20
        self.n_actions = (len(self.stickX_values) *
                          len(self.r_values) *
                          len(self.up_values) *
                          len(self.l_values))

    def send_init_state(self, status):
        self.states[self.env_id] = status
        self.conn.send("Sent initial states")

    def recieve_actions(self):
        """Receive a list of NUM_KARTS action ints from the master.

        Backwards-compat: if the master sends a single int (legacy slot-0-only path),
        we broadcast it to slot 0 only and leave the other slots' last values intact.
        """
        a = self.conn.recv()
        if isinstance(a, (list, tuple)):
            self.applied_actions = list(a)
        else:
            self.applied_actions[0] = int(a)

    def send_transition(self, reward, terminal, trun, new_status):
        # write into shared memory

        if self.reset_frame_buffer:
            # Overwrite the entire frame stack with the new frame
            self.states[self.env_id, ...] = new_status
            self.reset_frame_buffer = False
        else:
            # Shift frames left: frames 1..end → 0..end-1
            self.states[self.env_id, :-1] = self.states[self.env_id, 1:]
            # Add new frame at the end (index -1)
            self.states[self.env_id, -1] = new_status

        # Per-kart arrays for the 12p logging path. Memory(play_num=N) only tracks
        # slots 0..N-1, so we always do a lightweight direct chain read for slots N..11
        # so the runner CSV captures all 12 trajectories.
        rc_all = [0.0] * NUM_KARTS
        kx_all = [0.0] * NUM_KARTS
        kz_all = [0.0] * NUM_KARTS
        rp_all = [0]   * NUM_KARTS
        # CPU-distillation labels: per-kart KPad::mRaceInputState read.
        # Chain (riidefi/mkw KartObjectProxy::getInput + RaceManager.hpp):
        #   *(0x809BD730) = RaceManager*
        #   RaceManager + 0x0C = RaceManagerPlayer**
        #   *(players + 4*i) = RaceManagerPlayer*
        #   RaceManagerPlayer + 0x48 = KPad* (kpadPlayer)
        #   KPad + 0x28 = KPadRaceInputState
        #     +0x04 u16 mButtons (bit0 = accel)
        #     +0x08 f32 mStick.x  ← what KPadAIController publishes from mCpuStick each frame
        #     +0x0C f32 mStick.y
        # Same address for real and CPU karts; KPadAIController::calcInner copies
        # mCpuStick → mRaceInputState before KartMove::calc reads. Reading post-frame
        # (we're inside send_transition, after frameadvance) gives the AI's actual
        # stick output as a clean BC label.
        cpu_sx_all = [0.0] * NUM_KARTS
        cpu_sy_all = [0.0] * NUM_KARTS
        cpu_btn_all = [0]  * NUM_KARTS
        try:
            mgr = memory.read_u32(0x809C18F8)
            kart_array = memory.read_u32(mgr + 0x20) if mgr else 0
        except Exception:
            kart_array = 0
        try:
            rmp_root = memory.read_u32(0x809BD730)
            players_arr = memory.read_u32(rmp_root + 0xC) if rmp_root else 0
        except Exception:
            players_arr = 0
        for i in range(NUM_KARTS):
            if i < self.memory_tracker.num_players:
                rc_all[i] = float(self.memory_tracker.RaceCompletion[i])
                kx_all[i] = float(self.memory_tracker.position[i][0])
                kz_all[i] = float(self.memory_tracker.position[i][2])
                rp_all[i] = int(self.memory_tracker.race_position[i])
            else:
                try:
                    # RaceCompletion via 0x809BD730 → [0xC, 0x4*i, 0xC]
                    pl = memory.read_u32(players_arr + 0x4 * i) if players_arr else 0
                    if pl >= 0x80000000 and pl < 0x81800000:
                        rc_all[i] = memory.read_f32(pl + 0xC)
                    # position via kart_array → kart_ptr → kart_obj → +0x8 → +0x90 → +0x18
                    if kart_array:
                        kp = memory.read_u32(kart_array + 0x4 * i)
                        if kp >= 0x80000000 and kp < 0x81800000:
                            ko = memory.read_u32(kp)
                            if ko and ko < 0x81800000:
                                kdc = memory.read_u32(ko + 0x8)
                                if kdc and kdc < 0x81800000:
                                    kdyn = memory.read_u32(kdc + 0x90)
                                    if kdyn and kdyn < 0x81800000:
                                        pos_addr = kdyn + 0x18
                                        kx_all[i] = memory.read_f32(pos_addr + 0)
                                        # +0x4 = y, +0x8 = z
                                        kz_all[i] = memory.read_f32(pos_addr + 8)
                    # race_position via kart_obj+0x18 → +0x3C
                    if kart_array:
                        kp = memory.read_u32(kart_array + 0x4 * i)
                        if kp >= 0x80000000 and kp < 0x81800000:
                            ko = memory.read_u32(kp)
                            if ko and ko < 0x81800000:
                                kc = memory.read_u32(ko + 0x18)
                                if kc and kc < 0x81800000:
                                    rp_all[i] = memory.read_u8(kc + 0x3C)
                except Exception:
                    pass
            # KPad mRaceInputState read — same chain regardless of play_num. Slots
            # without a kpadPlayer (TYPE_NONE) leave the defaults at 0.
            try:
                pl = memory.read_u32(players_arr + 0x4 * i) if players_arr else 0
                _kpad_dbg = 0
                if pl >= 0x80000000 and pl < 0x81800000:
                    kpad = memory.read_u32(pl + 0x48)
                    _kpad_dbg = kpad
                    # Allow MEM2 too — kpadPlayer may live in either heap.
                    if (0x80000000 <= kpad < 0x81800000) or (0x90000000 <= kpad < 0x94000000):
                        cpu_btn_all[i] = int(memory.read_u16(kpad + 0x28 + 0x04))
                        cpu_sx_all[i]  = float(memory.read_f32(kpad + 0x28 + 0x08))
                        cpu_sy_all[i]  = float(memory.read_f32(kpad + 0x28 + 0x0C))
                # Diagnostic dump (only when MKW_KPAD_DIAG=1): probe kpad layout +
                # the +0x04 sub-pointer at race-start to verify mRaceInputState offset.
                if i == NUM_KARTS - 1 and os.environ.get("MKW_KPAD_DIAG") == "1":
                    _diag_step_count = getattr(self, "_kpad_diag_step", 0) + 1
                    self._kpad_diag_step = _diag_step_count
                    _stage_now = int(self.mem_race_stage) if hasattr(self, "mem_race_stage") else 0
                    _race_seen = getattr(self, "_kpad_race_first_step", None)
                    if _race_seen is None and _stage_now >= 2:
                        self._kpad_race_first_step = _diag_step_count
                        _race_seen = _diag_step_count
                    fire = (
                        _diag_step_count == 1 or
                        (_race_seen is not None and (_diag_step_count - _race_seen) in (0, 5, 30))
                    )
                    if fire:
                        diag_lines = [f"KPad probe @ step {_diag_step_count} stage={_stage_now}:"]
                        for j in (0, 4, 5, 8):
                            try:
                                pj = memory.read_u32(players_arr + 0x4 * j) if players_arr else 0
                            except Exception:
                                pj = 0
                            kj = 0
                            if (0x80000000 <= pj < 0x81800000) or (0x90000000 <= pj < 0x94000000):
                                try:
                                    kj = memory.read_u32(pj + 0x48)
                                except Exception:
                                    kj = 0
                            diag_lines.append(f"  --- slot {j}  pl=0x{pj:08x}  kpad=0x{kj:08x} ---")
                            valid_kj = (0x80000000 <= kj < 0x81800000) or (0x90000000 <= kj < 0x94000000)
                            if valid_kj:
                                # Dump 0x80 bytes (32 u32s) from kpad.
                                row = []
                                for k in range(32):
                                    try:
                                        v = memory.read_u32(kj + 0x4 * k)
                                    except Exception:
                                        v = 0xDEADBEEF
                                    row.append(f"+0x{0x4*k:02x}={v:08x}")
                                    if (k + 1) % 4 == 0:
                                        diag_lines.append("    " + "  ".join(row))
                                        row = []
                                # Also dump 0x40 bytes from the kpad+0x04 sub-pointer.
                                try:
                                    sub = memory.read_u32(kj + 0x04)
                                except Exception:
                                    sub = 0
                                if (0x80000000 <= sub < 0x81800000) or (0x90000000 <= sub < 0x94000000):
                                    diag_lines.append(f"    [sub @ 0x{sub:08x} (=*(kpad+0x04)):]")
                                    row = []
                                    for k in range(16):
                                        try:
                                            v = memory.read_u32(sub + 0x4 * k)
                                        except Exception:
                                            v = 0xDEADBEEF
                                        row.append(f"+0x{0x4*k:02x}={v:08x}")
                                        if (k + 1) % 4 == 0:
                                            diag_lines.append("    " + "  ".join(row))
                                            row = []
                        log_diag("\n".join(diag_lines))
            except Exception:
                pass

        # Kart-layout diagnostic (MKW_KART_LAYOUT_DIAG=1): one-shot dump of
        # KartMisc (DriftState) + ItemSlot (Item ID) chain pointers and bytes
        # for slots 0, 4, 5. Use this to find the correct offset within
        # ItemSlot for current Item ID — the +0x8C offset returns garbage for
        # CPU slots in PAL/RMCP01.
        if os.environ.get("MKW_KART_LAYOUT_DIAG") == "1" and not getattr(self, "_layout_diag_done", False):
            _stage_now = int(self.mem_race_stage) if hasattr(self, "mem_race_stage") else 0
            if _stage_now >= 2:
                self._layout_diag_done = True
                try:
                    karr_dbg = kart_array
                    item_mgr_dbg = memory.read_u32(0x809C3618)
                    item_arr_dbg = memory.read_u32(item_mgr_dbg + 0x14) if item_mgr_dbg else 0
                except Exception:
                    karr_dbg = item_mgr_dbg = item_arr_dbg = 0
                ll = [f"Kart-layout probe stage={_stage_now}:",
                      f"  karr=0x{karr_dbg:08x}  item_mgr=0x{item_mgr_dbg:08x}  item_arr=0x{item_arr_dbg:08x}"]
                for j in (0, 4, 5):
                    try:
                        kp_j = memory.read_u32(karr_dbg + 0x4 * j) if karr_dbg else 0
                    except Exception:
                        kp_j = 0
                    ko_j = 0
                    if 0x80000000 <= kp_j < 0x81800000:
                        try:
                            ko_j = memory.read_u32(kp_j)
                        except Exception:
                            ko_j = 0
                    # DriftState chain comparison: kp+0x44 (heavy) vs ko+0x44 (lite-old)
                    kmisc_kp = kmisc_ko = 0
                    drift_via_kp = drift_via_ko = "n/a"
                    if 0x80000000 <= kp_j < 0x81800000:
                        try:
                            kmisc_kp = memory.read_u32(kp_j + 0x44)
                            if 0x80000000 <= kmisc_kp < 0x81800000:
                                drift_via_kp = f"{memory.read_u16(kmisc_kp + 0xFC)}"
                        except Exception:
                            pass
                    if 0x80000000 <= ko_j < 0x81800000:
                        try:
                            kmisc_ko = memory.read_u32(ko_j + 0x44)
                            if 0x80000000 <= kmisc_ko < 0x81800000:
                                drift_via_ko = f"{memory.read_u16(kmisc_ko + 0xFC)}"
                        except Exception:
                            pass
                    # ItemSlot + bytes
                    islot_j = 0
                    if item_arr_dbg:
                        try:
                            islot_j = memory.read_u32(item_arr_dbg + 0x4 * j)
                        except Exception:
                            islot_j = 0
                    ll.append(
                        f"  --- slot {j}  kp=0x{kp_j:08x}  ko=0x{ko_j:08x} ---"
                    )
                    ll.append(
                        f"      kmisc(kp+0x44)=0x{kmisc_kp:08x}  drift={drift_via_kp}    "
                        f"kmisc(ko+0x44)=0x{kmisc_ko:08x}  drift={drift_via_ko}"
                    )
                    ll.append(f"      islot=0x{islot_j:08x}")
                    if 0x80000000 <= islot_j < 0x81800000:
                        # Dump u32s from islot+0x80 to islot+0x110, looking for item ID 0..18.
                        row = []
                        for k in range(0x80 // 4, 0x114 // 4):
                            off = 0x4 * k
                            try:
                                v = memory.read_u32(islot_j + off)
                            except Exception:
                                v = 0xDEADBEEF
                            row.append(f"+0x{off:03x}={v:08x}")
                            if len(row) == 4:
                                ll.append("        " + "  ".join(row))
                                row = []
                        if row:
                            ll.append("        " + "  ".join(row))
                log_diag("\n".join(ll))

        info = {
            "RaceCompletion": float(self.mem_race_com),
            "RaceCompletion_all": rc_all,
            "kart_x_all": kx_all,
            "kart_z_all": kz_all,
            "race_pos_all": rp_all,
            "cpu_stickX_all": cpu_sx_all,
            "cpu_stickY_all": cpu_sy_all,
            "cpu_buttons_all": cpu_btn_all,
            "race_stage": int(self.mem_race_stage) if hasattr(self, "mem_race_stage") else int(self.memory_tracker.stage),
        }
        self.conn.send((reward, terminal, trun, info))

    def get_mem_values(self):
        self.memory_tracker.update()

        self.mem_speed = self.memory_tracker.speed[0]

        self.mem_race_pos = self.memory_tracker.race_position[0]

        # max race completion
        self.mem_race_com = self.memory_tracker.RaceCompletion[0]

        self.mem_offroad_invin = self.memory_tracker.offroadInvincibility[0]
        self.mem_touching_offroad = self.memory_tracker.isTouchingOffroad[0]

        self.mem_race_stage = self.memory_tracker.stage

    def reset(self):

        self.ep_length = 0

        # per-kart actions applied each frame; default 0 = stickX 0, no R, no L.
        self.applied_actions = [0] * NUM_KARTS

        self.frames_since_chkpt = 0

        self.num_checkpoints_per_lap = 10  # 3 laps total
        self.checkpoints = []
        self.current_checkpoint = 0

        num_checkpoints_per_lap = 10
        num_laps = 3
        start = 1.0
        end = 4.0
        lap_length = (end - start) / num_laps  # 1.0

        for lap in range(num_laps):
            lap_start = start + lap * lap_length
            step = lap_length / num_checkpoints_per_lap
            # Exclude the lap_start itself (since your tracker starts at 1.0, not 0.0)
            for i in range(1, num_checkpoints_per_lap + 1):
                self.checkpoints.append(round(lap_start + i * step, 10))  # rounding for floating point issues

        # just make sure we don't list index out of range
        self.checkpoints.append(9999.)

        savestate_path = self._select_reset_savestate()
        log_diag(f"reset: loading savestate {savestate_path}")
        savestate.load_from_file(str(savestate_path))
        log_diag("reset: savestate loaded; probing kart-array layout before Memory(...)")
        # Diagnostic: walk the kart-pointer array at 0x809C18F8 → [0x20, 0x4*i] for i in 0..15
        # and log each slot's pointer + KartObject base + KartState/Move/Collide pointers.
        # This tells us which slots are actually live in this savestate before we let
        # Memory.Addresses iterate (which currently hangs/crashes on bad slots).
        try:
            mgr = memory.read_u32(0x809C18F8)
            arr = memory.read_u32(mgr + 0x20) if mgr else 0
            log_diag(f"  KartObjectManager @ 0x809C18F8 -> mgr=0x{mgr:08x}; kart_array=0x{arr:08x}")
            for i in range(16):
                if arr == 0:
                    break
                kart_ptr_addr = arr + 0x4 * i
                if kart_ptr_addr < 0x80000000 or kart_ptr_addr >= 0x81800000:
                    log_diag(f"  slot {i}: array slot @0x{kart_ptr_addr:08x} OUT OF RANGE")
                    break
                kp = memory.read_u32(kart_ptr_addr)
                ko = memory.read_u32(kp) if (kp >= 0x80000000 and kp < 0x81800000) else 0
                log_diag(f"  slot {i}: ptr@0x{kart_ptr_addr:08x} -> kart_ptr=0x{kp:08x} kart_obj=0x{ko:08x}")
        except Exception as e:
            log_exc(e)

        log_diag("reset: building Memory(play_num={})".format(self.play_num))
        self.memory_tracker = Memory(self.play_num)
        log_diag("reset: Memory built; resolving InputData base")

        # InputData base pointer can change after savestate load. Resolve it now.
        if self.input_writer is None:
            self.input_writer = KartInputWriter()
        else:
            self.input_writer.refresh_base()
        log_diag(f"reset: input_writer base=0x{self.input_writer.base:08x}")

        # NOP-patch PlayerSub1c::updateFromInput (PAL/RMCP01 @ 0x8059487c) to a `blr`
        # so it returns immediately. This stops MKW from rewriting PlayerSub1c each
        # frame from InputData/AI sources, letting our direct PlayerSub1c writes
        # persist into kart physics. After this patch, ALL 12 karts must be driven
        # by our writes — including slot 0 (no more controller.set_gc_buttons fallback).
        # Gated by env var so we can compare A/B without rebuilding.
        # NOP these per-frame functions to silence the game's input pipeline so
        # our direct PlayerSub1c writes survive into kart physics. Both addresses
        # are PAL/RMCP01 from the mkw-sp symbol map (https://github.com/mkw-sp/mkw-sp/blob/main/symbols.txt)
        # and SeekyCt/mkw-structures · player.h.
        BLR = 0x4E800020
        if os.environ.get("MKW_NOP_UPDATEFROMINPUT", "1") == "1":
            try:
                UPDATEFROMINPUT_ADDR = 0x8059487C  # PlayerSub1c::updateFromInput
                before = memory.read_u32(UPDATEFROMINPUT_ADDR)
                memory.write_u32(UPDATEFROMINPUT_ADDR, BLR)
                after  = memory.read_u32(UPDATEFROMINPUT_ADDR)
                log_diag(f"reset: NOP updateFromInput @ 0x{UPDATEFROMINPUT_ADDR:08x}: before=0x{before:08x} after=0x{after:08x}")
            except Exception as e:
                log_exc(e)
        if os.environ.get("MKW_NOP_AI_CALC", "1") == "1":
            try:
                # Enemy::EnemyManager::calc — per-frame AI for slots 4-11.
                # Empirically only stops AI for those slots; slots 1-3 keep racing.
                AI_CALC_ADDR = 0x8073942C
                before = memory.read_u32(AI_CALC_ADDR)
                memory.write_u32(AI_CALC_ADDR, BLR)
                log_diag(f"reset: NOP EnemyManager::calc @ 0x{AI_CALC_ADDR:08x}: before=0x{before:08x}")
            except Exception as e:
                log_exc(e)
        if os.environ.get("MKW_NOP_DRIVER_MANAGER", "0") == "1":  # off by default — didn't help
            try:
                DM_CALC_ADDR = 0x8078D824
                memory.write_u32(DM_CALC_ADDR, BLR)
                log_diag(f"reset: NOP DriverManager::calc @ 0x{DM_CALC_ADDR:08x}")
            except Exception as e:
                log_exc(e)
        if os.environ.get("MKW_NOP_GHOSTPAD", "0") == "1":  # off — didn't isolate slots 1-3
            try:
                GHOSTPAD_ADDR = 0x80520B9C
                memory.write_u32(GHOSTPAD_ADDR, BLR)
                log_diag(f"reset: NOP GhostPad::process @ 0x{GHOSTPAD_ADDR:08x}")
            except Exception as e:
                log_exc(e)
        if os.environ.get("MKW_NOP_INPUTMANAGER", "1") == "1":
            try:
                # System::InputManager::calc — top-level pad processing dispatch.
                # Stops all pad I/O for all 16 holders. Slot 0's GC controller path
                # is already irrelevant here (we drive slot 0 via PlayerSub1c writes
                # and updateFromInput is NOPed). For slots 1-3 (which seem to use
                # some Pad chain we haven't identified), this should also silence them.
                IM_CALC_ADDR = 0x805238F0
                before = memory.read_u32(IM_CALC_ADDR)
                memory.write_u32(IM_CALC_ADDR, BLR)
                log_diag(f"reset: NOP InputManager::calc @ 0x{IM_CALC_ADDR:08x}: before=0x{before:08x}")
            except Exception as e:
                log_exc(e)

        # Change PlayerType for slots 4-11 from PLAYER_CPU (=1) to PLAYER_REAL_LOCAL (=0)
        # so MKW's AI subsystem stops driving them. Reach via:
        #   0x809BD728 → read_u32 → + (0x38 + 0xF0*i)
        # (per SeekyCt/mkw-structures · racedata.h: RacedataPlayer is 0xF0 bytes,
        #  playerType is at struct +0x10 = chain +0x38, where 0x28 is the array
        #  offset within the RaceData struct that 0x809BD728 → resolves to.)
        if os.environ.get("MKW_FORCE_REAL_LOCAL", "1") == "1":
            try:
                race_data = memory.read_u32(0x809BD728)
                # accept both MEM1 (0x80000000..0x817FFFFF) and MEM2 (0x90000000..0x93FFFFFF)
                rd_ok = (race_data and (
                    (0x80000000 <= race_data < 0x81800000) or
                    (0x90000000 <= race_data < 0x94000000)
                ))
                if rd_ok:
                    PLAYER_REAL_LOCAL = 0
                    converted = []
                    for slot in range(NUM_KARTS):
                        addr = race_data + 0x38 + 0xF0 * slot
                        try:
                            cur = memory.read_u32(addr)
                            if slot >= 4:
                                memory.write_u32(addr, PLAYER_REAL_LOCAL)
                            converted.append(f"slot{slot}:{cur}{'→0' if slot>=4 else ''}")
                        except Exception:
                            pass
                    log_diag(f"reset: PlayerType (race_data=0x{race_data:08x}): {' '.join(converted)}")
                else:
                    log_diag(f"reset: PlayerType conversion SKIPPED — race_data=0x{race_data:08x} out of MEM1/MEM2")
            except Exception as e:
                log_exc(e)

        # Probe-mode CSV header (write once, on the first reset that creates the file).
        if PROBE_INPUT and self._probe_csv_path is not None:
            try:
                if not self._probe_csv_path.exists():
                    header = "frame,slot,vtable,buttonActions,buttonRaw,stickX,stickY,qStickX,qStickY,motionFlick,motionFlick2,yaw_deg\n"
                    self._probe_csv_path.write_text(header)
            except Exception as e:
                log_exc(e)

        self.get_mem_values()
        self.prev_race_completion = float(self.mem_race_com)
        self.prev_wall_collide = int(self.memory_tracker.wall_collide[0])

        # move our current checkpoint to where we are based on spawn location
        while self.mem_race_com > self.checkpoints[self.current_checkpoint]:
            self.current_checkpoint += 1


    def _decode_action(self, action):
        """Decode a single Discrete action int into (stickX, R, Up, L) bools/values."""
        action = int(action) % self.n_actions
        stride = len(self.r_values) * len(self.up_values) * len(self.l_values)
        stick_idx = action // stride
        rem = action % stride
        r_idx = rem // (len(self.up_values) * len(self.l_values))
        rem = rem % (len(self.up_values) * len(self.l_values))
        up_idx = rem // len(self.l_values)
        l_idx = rem % len(self.l_values)
        return (
            self.stickX_values[stick_idx],
            bool(self.r_values[r_idx]),
            bool(self.up_values[up_idx]),
            bool(self.l_values[l_idx]),
        )

    def apply_actions(self, actions):
        """Write per-kart input for all 12 karts.

        Slot 0 is the local player and is driven via Dolphin's GC port 0 (the proven
        path that has been working all along). Slots 1..11 are driven by direct memory
        writes into MKW's virtualControllerHolders[i].inputStates[0] (and also the
        matching realControllerHolders[1..3] for slots 1..3, since those map to
        physical GC ports the game also reads from).

        In probe mode (MKW_PROBE_INPUT=1), no writes happen; instead, we read every
        kart's InputState + yaw and dump a CSV row.
        """
        self.get_mem_values()

        if PROBE_INPUT:
            try:
                rows = []
                for i in range(NUM_KARTS):
                    st = self.input_writer.read_state(i)
                    yaw_deg = float(self.memory_tracker.mainRotationEuler[i][2]) if i < self.memory_tracker.num_players else 0.0
                    rows.append(
                        f"{self._probe_frame},{i},0x{st['vtable']:08x},{st['buttonActions']},{st['buttonRaw']},"
                        f"{st['stickX']:.6f},{st['stickY']:.6f},{st['qStickX']},{st['qStickY']},"
                        f"{st['motionFlick']},{st['motionFlick2']},{yaw_deg:.4f}\n"
                    )
                if self._probe_csv_path is not None:
                    with open(self._probe_csv_path, "a") as f:
                        f.writelines(rows)
                self._probe_frame += 1
            except Exception as e:
                log_exc(e)
            return

        # On the very first call where race_stage transitions out of countdown,
        # dump every kart's PlayerSub1c state so we can verify our writes land.
        if not getattr(self, "_dumped_active", False) and self.mem_race_stage == 2:
            try:
                for ki in range(NUM_KARTS):
                    sub1c = resolve_playersub1c(ki)
                    if sub1c >= 0x80000000 and sub1c < 0x81800000:
                        bf0 = memory.read_u32(sub1c + 0x04)
                        bf4 = memory.read_u32(sub1c + 0x14)
                        sx = memory.read_f32(sub1c + 0x88)
                        log_diag(f"  kart {ki}: sub1c=0x{sub1c:08x} bf0=0x{bf0:08x} bf4=0x{bf4:08x} stickX={sx:.3f}")
                self._dumped_active = True
            except Exception as e:
                log_exc(e)

        # With updateFromInput NOP-patched, the only path the game has for getting
        # input into PlayerSub1c is OUR direct writes. Drive every kart uniformly
        # via PlayerSub1c writes; no more set_gc_buttons fallback for slot 0.
        for i in range(NUM_KARTS):
            try:
                stickX, R, Up, L = self._decode_action(actions[i])
            except Exception:
                stickX, R, Up, L = 0.0, False, False, False

            # bitfield0 — PowerPC MSB-from-0 bit numbering (verified empirically).
            buttons_bf = PSUB1C_ACCEL_BIT  # accelerate always held
            if R:
                buttons_bf |= PSUB1C_DRIFT_BIT
            if stickX < -0.05:
                buttons_bf |= PSUB1C_STICKLEFT_BIT
            elif stickX > 0.05:
                buttons_bf |= PSUB1C_STICKRIGHT_BIT

            try:
                sub1c = resolve_playersub1c(i)
                if sub1c >= 0x80000000 and sub1c < 0x81800000:
                    # R-M-W bitfield0: preserve game-state bits (collision/ground/etc.),
                    # overwrite only our 5 input bits.
                    cur = memory.read_u32(sub1c + PLAYERSUB1C_BITFIELD0)
                    new = (cur & ~PSUB1C_INPUT_BITS_MASK) | (buttons_bf & PSUB1C_INPUT_BITS_MASK)
                    memory.write_u32(sub1c + PLAYERSUB1C_BITFIELD0, new)
                    memory.write_f32(sub1c + PLAYERSUB1C_STICKX, float(stickX))
                    memory.write_f32(sub1c + PLAYERSUB1C_STICKY, 0.0)

                    # Force PlayerSub1c.bitfield4 to "real local" for slots 4-11 each
                    # frame. (Doesn't actually disable AI — AI subsystem is wired at
                    # race-init — but harmless. Keeping for consistency.)
                    if i >= 4:
                        memory.write_u32(sub1c + 0x14, 0x2)

                    # Slots 1-3 always need their KartMove.speed forced (mystery
                    # drive mechanism). Other slots optionally — set
                    # MKW_FORCE_KARTMOVE_SPEED_ALL=1 to force ALL slots' speed
                    # too, useful in mid-race savestates where pre-existing
                    # momentum keeps the karts moving forward despite our NOPs.
                    force_speed_all = os.environ.get("MKW_FORCE_KARTMOVE_SPEED_ALL", "0") == "1"
                    if (1 <= i <= 3) or force_speed_all:
                        try:
                            mgr = memory.read_u32(0x809C18F8)
                            karr = memory.read_u32(mgr + 0x20) if mgr else 0
                            kp = memory.read_u32(karr + 0x4 * i) if karr else 0
                            ko = memory.read_u32(kp) if kp else 0
                            kmove = memory.read_u32(ko + 0x28) if ko else 0
                            if kmove and 0x80000000 <= kmove < 0x94000000:
                                # Speed mode: 0 = freeze (default OFF), forward speed
                                # = drive forward at a constant rate when accel bit set.
                                if os.environ.get("MKW_FORCE_KARTMOVE_SPEED", "0") == "1":
                                    memory.write_f32(kmove + 0x20, 0.0)
                                else:
                                    # Accel always held in our input → push forward speed.
                                    # Desired ~50 units/sec which is a moderate cruising speed
                                    # for MKW karts on Luigi Circuit.
                                    target_speed = 50.0
                                    memory.write_f32(kmove + 0x20, target_speed)
                                memory.write_f32(kmove + 0x18, 120.0)  # softSpeedLimit
                                memory.write_f32(kmove + 0x2C, 120.0)  # hardSpeedLimit
                        except Exception:
                            pass

                    # PlayerSub1c.stickX writes don't actually steer the kart — the
                    # physics layer reads stick from somewhere else (likely KartMove
                    # internal state). Brute-force fix: set the kart's angular
                    # velocity Y (yaw rate) directly each frame to stickX*turn_rate.
                    # AngularVelocity is at KartDynamics +0xA4 (Vec3 f32, 12 bytes).
                    # Chain: 0x809C18F8 → [0x20, 0x4*i, 0x0, 0x8, 0x90, 0x4, 0xA4]
                    # Y component = +0xA8.
                    if os.environ.get("MKW_FORCE_ANGVEL", "1") == "1":
                        try:
                            mgr = memory.read_u32(0x809C18F8)
                            karr = memory.read_u32(mgr + 0x20) if mgr else 0
                            kp = memory.read_u32(karr + 0x4 * i) if karr else 0
                            ko = memory.read_u32(kp) if kp else 0
                            kdc = memory.read_u32(ko + 0x8) if ko else 0
                            kdyn = memory.read_u32(kdc + 0x90) if kdc else 0
                            inner = memory.read_u32(kdyn + 0x4) if kdyn else 0
                            if inner and 0x80000000 <= inner < 0x94000000:
                                # Turn rate at full stick deflection. Default 3.0 rad/s
                                # works for low-speed savestates (s01); for high-momentum
                                # mid-race savestates (s08) the physics integrator can
                                # absorb a 3.0 yaw — bump MKW_YAW_RATE to overcome.
                                yaw_rate = float(os.environ.get("MKW_YAW_RATE", "3.0"))
                                yaw = float(stickX) * yaw_rate
                                memory.write_f32(inner + 0xA8, yaw)
                        except Exception:
                            pass
            except Exception as e:
                if i == 0 and self._probe_frame == 0:
                    log_exc(e)

    def get_reward_terminal_trun(self):
        reward = 0.
        terminal = False
        trun = False

        # refresh memory values
        self.get_mem_values()
        progress_delta = float(self.mem_race_com - self.prev_race_completion)
        self.prev_race_completion = float(self.mem_race_com)
        wall_hit = self.mem_wall_collide_changed()
        touching_offroad = bool(self.mem_touching_offroad)

        self.ep_length += 1

        # Dense progress reward makes PPO much easier to optimize than sparse checkpoints alone.
        reward += 20.0 * min(max(progress_delta, 0.0), 0.05)

        # Encourage faster forward driving, but only when actual progress is happening.
        if progress_delta > 0.0:
            speed_bonus = min(max(float(self.mem_speed), 0.0), 120.0) / 120.0
            reward += 0.02 * speed_bonus

        if touching_offroad:
            reward -= 0.01

        if wall_hit:
            reward -= 0.05

        # checkpoint bonus
        if self.mem_race_com > self.checkpoints[self.current_checkpoint]:
            reward += 1.
            self.current_checkpoint += 1
            self.frames_since_chkpt = 0

        # reward for finishing race and set terminal
        if self.mem_race_com >= 4.0:
            reward = (13 - self.mem_race_pos) / 2
            terminal = True
            log_diag(f"terminal: FINISH ep_len={self.ep_length} rc={self.mem_race_com:.3f} race_pos={self.mem_race_pos}")
        # race has ended, reset
        elif self.mem_race_stage == 4:
            reward = -1
            terminal = True
            log_diag(f"terminal: race_stage=4 ep_len={self.ep_length} rc={self.mem_race_com:.3f} frames_since_chkpt={self.frames_since_chkpt}")
        # reset condition.
        elif (
            episode_timeout_steps is not None
            and self.frames_since_chkpt > episode_timeout_steps
        ):
            reward = -1.
            terminal = True
            log_diag(f"terminal: TIMEOUT ep_len={self.ep_length} rc={self.mem_race_com:.3f} frames_since_chkpt={self.frames_since_chkpt} timeout={episode_timeout_steps}")

        # Hard kill-switch: when MKW_DISABLE_RESET=1, swallow ALL terminal events
        # so no savestate reload ever fires. The env runs forever (until docker
        # timeout / Ctrl-C). Useful for GUI inspection and long override tests.
        if DISABLE_RESET and (terminal or trun):
            if not getattr(self, "_logged_disable_reset", False):
                log_diag(f"DISABLE_RESET active — swallowed terminal at ep_len={self.ep_length} rc={self.mem_race_com:.3f}")
                self._logged_disable_reset = True
            terminal = False
            trun = False
            reward = 0.0

        self.frames_since_chkpt += 1

        return reward, terminal, trun

    def mem_wall_collide_changed(self):
        current = int(self.memory_tracker.wall_collide[0])
        changed = current != 0 and current != self.prev_wall_collide
        self.prev_wall_collide = current
        return changed

log_diag(f"slave bootstrap: PROBE_INPUT={PROBE_INPUT} DISABLE_RESET={DISABLE_RESET} NUM_KARTS={NUM_KARTS}")
log_diag(f"dolphin api: dir(memory) has write_u32? {'write_u32' in dir(memory)}; dir(event)={[x for x in dir(event) if 'frame' in x.lower() or 'memory' in x.lower()]}")

for i in range(4):
    await event.frameadvance()
log_diag("4 warm-up frames advanced; constructing DolphinInstance(play_num=NUM_KARTS)")

# Allow narrowing the heavyweight Memory tracker for debugging via env var.
# The obs vector is ALWAYS 941 wide regardless — slots beyond play_num are
# populated by _lite_per_kart_obs in Memory.get_obs().
play_num = int(os.environ.get("MKW_PLAY_NUM", str(NUM_KARTS)))
play_num = max(1, min(NUM_KARTS, play_num))
obs_shape = 5 + 78 * NUM_KARTS  # always 941
log_diag(f"play_num={play_num} (MKW_PLAY_NUM env override applied if set); obs_shape={obs_shape}")
env = DolphinInstance(id, play_num)
log_diag(f"DolphinInstance constructed; play_num={play_num} obs_shape={obs_shape}")

for i in range(8):
    await event.frameadvance()
log_diag("8 post-init frames advanced; updating memory_tracker for init state")

await event.frameadvance()

env.memory_tracker.update()
log_diag("memory_tracker.update OK; calling get_obs()")
current_vector = env.memory_tracker.get_obs()
log_diag(f"get_obs OK; len={len(current_vector)}")
init_vector = np.array([current_vector for _ in range(env.frameskip)])

env.send_init_state(init_vector)
log_diag("send_init_state sent")

print(f"Sent init state | play_num={play_num} obs_shape={obs_shape} probe={PROBE_INPUT}")

def my_callback():
    env.apply_actions(env.applied_actions)

# MKW_INPUT_HOOK: 'frameadvance' (default) or 'framedrawn' — try framedrawn to see
# if it fires later in the frame (after PadProxy::calc has filled m_currentRaceInputState
# but before next-frame's calc clobbers our writes).
_input_hook = os.environ.get("MKW_INPUT_HOOK", "frameadvance")
if _input_hook == "framedrawn":
    def _drawn_cb(width, height, data):
        env.apply_actions(env.applied_actions)
    event.on_framedrawn(_drawn_cb)
    log_diag("on_framedrawn registered (MKW_INPUT_HOOK=framedrawn); entering main loop")
else:
    event.on_frameadvance(my_callback)
    log_diag("on_frameadvance(my_callback) registered; entering main loop")
# make sure we apply the action every single frame. Otherwise this can lead to some weird stuttering
# behaviour

reward = 0
terminal = False
trun = False

print("Starting Main Loop...")
# atari pools the most recent two frames, don't blame me why its so confusing
frame_data = np.zeros((obs_shape), dtype=np.float32)
# TODO: data range check
while True:

    # get action from main Dolphin Script
    env.recieve_actions()

    for i in range(env.frameskip):
        if i >= env.frameskip-1:
            await event.frameadvance()
            env.memory_tracker.update()
            current_vector = env.memory_tracker.get_obs()
            frame_data = current_vector
        else:
            # no frame data, just skip frame
            await event.frameadvance()

        rewardN, terminalN, trunN = env.get_reward_terminal_trun()

        if not terminal and not trun:
            terminal = terminal or terminalN
            trun = trun or trunN
            reward += rewardN

        if terminal or trun:
            # send transition so we can carry going on while resetting

            env.send_transition(reward, terminal, trun, np.array(frame_data).copy())

            # add some time here or dolphin seems to freeze up sometimes
            for _ in range(2):
                await event.frameadvance()

            env.reset()

            for _ in range(1):
                await event.frameadvance()

            # reset frame_buffer
            env.reset_frame_buffer = True
            break

    if not (terminal or trun):
        env.send_transition(reward, terminal, trun, np.array(frame_data).copy())

    reward = 0
    terminal = False
    trun = False
