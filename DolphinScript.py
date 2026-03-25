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
sys.executable = str(script_directory / ".venv" / "Scripts" / "python.exe")

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

log_path = instance_info_folder / f'slave_{id}.log'
def log_exc(exc: BaseException):
    with open(log_path, 'a') as f:
        f.write('— Exception occurred —\n')
        traceback.print_exc(file=f)
        f.write('\n\n')


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
                self.mushroomCount.append(self.resolve_address(0x809C3618, [0x14, 0x4 * i, 0x90]))
                self.Item.append(self.resolve_address(0x809C3618, [0x14, 0x4 * i, 0x8C]))
                self.ItemNum.append(self.resolve_address(0x809C3618, [0x14, 0x4 * i, 0x90]))
                self.PassiveItem.append(self.resolve_address(0x809C3618, [0x14, 0x4 * i, 0xCC]))
                self.PassiveItemNum.append(self.resolve_address(0x809C3618, [0x14, 0x4 * i, 0x104]))

        def resolve_address(self, base_address, offsets):
            current_address = memory.read_u32(base_address)
            for offset in offsets:
                value_address = current_address + offset
                current_address = memory.read_u32(current_address + offset)
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
            
            self.mushroomCount[i] = memory.read_u32(self.addresses.mushroomCount[i])
            self.Item[i] = memory.read_u32(self.addresses.Item[i])
            self.ItemNum[i] = memory.read_u32(self.addresses.ItemNum[i])
            self.PassiveItem[i] = memory.read_u32(self.addresses.PassiveItem[i])
            self.PassiveItemNum[i] = memory.read_u32(self.addresses.PassiveItemNum[i])

            self.StarTimer[i] = memory.read_u16(self.addresses.StarTimer[i])
            self.ShockTimer[i] = memory.read_u16(self.addresses.ShockTimer[i])
            self.BlooperInkTimer[i] = memory.read_u16(self.addresses.BlooperInkTimer[i])
            self.BlooperStateFlag[i] = memory.read_u8(self.addresses.BlooperStateFlag[i])
            self.CrushTimer[i] = memory.read_u16(self.addresses.CrushTimer[i])
            self.MegaTimer[i] = memory.read_u16(self.addresses.MegaTimer[i])

    def get_obs(self):
        obs = []
        
        # 1. RACE_INFO 구성
        race_info = (
            self.stage,
            self.FrameCount,
            self.PlayerCount,
            self.CourseID,
            self.EngineClass
        )
        obs.extend(race_info)

        # 2. PLAYER_INFO 구성
        for n in range(self.num_players):
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
        self.obs_shape = 5 + 78 * play_num

        self.bestL1 = 999999
        self.bestL2 = 999999
        self.best_time = get_value()
        self.save_idx = 10

        self.reset_frame_buffer = False

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
        self.reset()

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
        self.stickX_values = [-1, -0.4, 0, 0.4, 1]
        self.r_values = [False, True]
        self.up_values = [False, True]
        self.l_values = [False, True]
        # Compute total number of discrete actions
        self.n_actions = (len(self.stickX_values) *
                          len(self.r_values) *
                          len(self.up_values) *
                          len(self.l_values))

    def send_init_state(self, status):
        self.states[self.env_id] = status
        self.conn.send("Sent initial states")

    def recieve_action(self):
        self.applied_action = self.conn.recv()

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

        # send the rest over the socket
        info = {
            "RaceCompletion": float(self.mem_race_com),
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

        # this action will be applied directly before the frame is drawn
        self.applied_action = 0

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

        # pick a random state to reset to
        save_states = [file for file in Path(save_states_path).rglob('*') if file.is_file() and ".s" in file.name]
        savestate.load_from_file(str(random.choice(save_states)))

        self.memory_tracker = Memory(self.play_num)

        self.get_mem_values()

        # move our current checkpoint to where we are based on spawn location
        while self.mem_race_com > self.checkpoints[self.current_checkpoint]:
            self.current_checkpoint += 1


    def apply_action(self, action):
        assert 0 <= action < self.n_actions, f"Action must be in 0..{self.n_actions-1}"

        # reset dictionary to default state (A is always held down)
        self.wii_dic = {
            "Left": False, "Right": False, "Down": False,
            "Up": False, "Z": False, "R": False, "L": False,
            "A": True, "B": False, "X": False, "Y": False,
            "Start": False, "StickX": 0, "StickY": 0, "CStickX": 0,
            "CStickY": 0, "TriggerLeft": 0, "TriggerRight": 0,
            "AnalogA": 0, "AnalogB": 0, "Connected": True
        }

        self.get_mem_values()

        # Decode indices. Can't lie ChatGPT did this, idn wtf is going on here
        stick_idx = action // (2 * 2 * 2)
        rem = action % (2 * 2 * 2)
        r_idx = rem // (2 * 2)
        rem = rem % (2 * 2)
        up_idx = rem // 2
        l_idx = rem % 2

        # Set relevant fields
        self.wii_dic["StickX"] = self.stickX_values[stick_idx]
        self.wii_dic["R"] = self.r_values[r_idx]
        self.wii_dic["Up"] = self.up_values[up_idx]
        self.wii_dic["L"] = self.l_values[l_idx]

        self.applied_action = action
        controller.set_gc_buttons(0, self.wii_dic)

    def get_reward_terminal_trun(self):
        reward = 0.
        terminal = False
        trun = False

        # refresh memory values
        self.get_mem_values()

        self.ep_length += 1

        # checkpoint bonus
        if self.mem_race_com > self.checkpoints[self.current_checkpoint]:
            reward += 1.
            self.current_checkpoint += 1
            self.frames_since_chkpt = 0

        # reward for finishing race and set terminal
        if self.mem_race_com >= 4.0:
            # reward based on position
            reward = (13 - self.mem_race_pos) / 2
            terminal = True
        # race has ended, reset
        elif self.mem_race_stage == 4:
            reward = -1
            terminal = True
        # reset condition.
        elif self.frames_since_chkpt > 700:
            reward = -1.
            terminal = True

        self.frames_since_chkpt += 1

        return reward, terminal, trun

for i in range(4):
    await event.frameadvance()

play_num = 1
obs_shape = 5 + 78 * play_num
env = DolphinInstance(id,play_num)

for i in range(8):
    await event.frameadvance()

await event.frameadvance()

env.memory_tracker.update()
current_vector = env.memory_tracker.get_obs()
init_vector = np.array([current_vector for _ in range(env.frameskip)])

env.send_init_state(init_vector)

print("Sent init state")

def my_callback():
    env.apply_action(env.applied_action)

event.on_frameadvance(my_callback)
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
    env.recieve_action()

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
