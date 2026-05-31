import mmap
import os
import struct
from typing import List


class DolphinMem:
    def __init__(self, pid: int):
        self.pid = pid

        fd_dir = f"/proc/{self.pid}/fd"
        target_fd = -1
        for fd_num in os.listdir(fd_dir):
            fd_path = os.path.join(fd_dir, fd_num)
            if "/dev/shm/dolphin-emu" in os.readlink(fd_path):
                with open(fd_path, "rb") as f:
                    target_fd = f.fileno()
                    # fmt: off
                    self.mm = mmap.mmap(target_fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
                    # fmt: on
                    self.mv = memoryview(self.mm)
                    return

        raise ValueError()

    def get_offset(self, gcn_addr: int) -> int:
        if 0x80000000 <= gcn_addr < 0x81800000:
            return gcn_addr - 0x80000000
        elif 0x90000000 <= gcn_addr < 0x94000000:
            return (gcn_addr - 0x90000000) + 0x02040000

        raise ValueError()

    def read_u8(self, addr: int) -> int:
        return self.mv[self.get_offset(addr)]

    def read_u16(self, addr: int) -> int:
        off = self.get_offset(addr)
        return struct.unpack(">H", self.mv[off : off + 2])[0]

    def read_u32(self, addr: int) -> int:
        off = self.get_offset(addr)
        return struct.unpack(">I", self.mv[off : off + 4])[0]

    def read_f32(self, addr: int) -> float:
        off = self.get_offset(addr)
        return struct.unpack(">f", self.mv[off : off + 4])[0]

    def read_fff(self, addr: int) -> List[float]:
        off = self.get_offset(addr)
        return list(struct.unpack(">fff", self.mv[off : off + 12]))

    def read_ffff(self, addr: int) -> List[float]:
        off = self.get_offset(addr)
        return list(struct.unpack(">ffff", self.mv[off : off + 16]))

    def read_ptr(self, addr: int) -> int:
        return self.read_u32(addr)

    def resolve_chain(self, base_addr: int, offsets: List[int]) -> int:
        assert len(offsets) > 0
        curr = self.read_ptr(base_addr)
        for offset in offsets[:-1]:
            curr += offset
            curr = self.read_ptr(curr)
        return curr + offsets[-1]

    def read_obs(self, agent_n: int) -> dict:
        obs = {"RACE_INFO": {}, "PLAYER_INFO": []}

        obs["RACE_INFO"] = {
            "StageID": self.read_u32(self.resolve_chain(0x809BD730, [0x28])),
            "FrameCount": self.read_u32(self.resolve_chain(0x809BD730, [0x20])),
            "PlayerCount": self.read_u8(0x809C38B8),
            "CourseID": self.read_u32(self.resolve_chain(0x809BD728, [0xB68])),
            "EngineClass": self.read_u32(self.resolve_chain(0x809BD728, [0xB6C])),
        }

        for n in range(agent_n):
            p_info = {}
            p_info["PlayerID"] = n

            p_info["LocalPlayerNum"] = self.read_u8(
                self.resolve_chain(0x809BD728, [0x2D + 0xF0 * n])
            )
            p_info["RealControllerID"] = self.read_u8(
                self.resolve_chain(0x809BD728, [0x2E + 0xF0 * n])
            )
            p_info["KartID"] = self.read_u32(
                self.resolve_chain(0x809BD728, [0x30 + 0xF0 * n])
            )
            p_info["CharacterID"] = self.read_u32(
                self.resolve_chain(0x809BD728, [0x34 + 0xF0 * n])
            )

            p_info["CurrentRaceCompletion"] = self.read_f32(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0xC])
            )
            p_info["MaxRaceCompletion"] = self.read_f32(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x10])
            )
            p_info["FirstKcpLapCompletion"] = self.read_f32(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x14])
            )
            p_info["NextCheckpointLapCompletion"] = self.read_f32(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x18])
            )
            p_info["NextCheckpointLapCompletionMax"] = self.read_f32(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x1C])
            )

            p_info["CurrentLap"] = self.read_u16(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x24])
            )
            p_info["MaxLap"] = self.read_u8(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x26])
            )
            p_info["currentKCP"] = self.read_u8(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x27])
            )
            p_info["maxKCP"] = self.read_u8(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x28])
            )
            p_info["StateBit"] = self.read_u8(
                self.resolve_chain(0x809BD730, [0xC, 0x4 * n, 0x3B])
            )

            p_info["SoftSpeedLimit"] = self.read_f32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x18])
            )
            p_info["HardSpeedLimit"] = self.read_f32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x2C])
            )

            p_info["Position"] = tuple(
                self.read_fff(
                    self.resolve_chain(
                        0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x18]
                    )
                )
            )
            p_info["Velocity"] = tuple(
                self.read_fff(
                    self.resolve_chain(
                        0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x4, 0xD4]
                    )
                )
            )
            p_info["InternalVelocity"] = tuple(
                self.read_fff(
                    self.resolve_chain(
                        0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x4, 0x14C]
                    )
                )
            )
            p_info["ExternalVelocity"] = tuple(
                self.read_fff(
                    self.resolve_chain(
                        0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x4, 0x74]
                    )
                )
            )
            p_info["AngularVelocity"] = tuple(
                self.read_fff(
                    self.resolve_chain(
                        0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x4, 0xA4]
                    )
                )
            )
            p_info["Acceleration"] = tuple(
                self.read_fff(
                    self.resolve_chain(
                        0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x4, 0x80]
                    )
                )
            )

            p_info["MainRotation"] = tuple(
                self.read_ffff(
                    self.resolve_chain(
                        0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x4, 0xF0]
                    )
                )
            )

            p_info["Speed"] = self.read_f32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x20])
            )
            p_info["AccelerationKartMove"] = self.read_f32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x30])
            )

            p_info["DriftState"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x44, 0xFC])
            )
            p_info["MiniturboCharge"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x44, 0xFE])
            )
            p_info["SMiniturboCharge"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x44, 0x100])
            )

            p_info["OffroadInvincibilityTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x148])
            )

            p_info["WheelieFrameCount"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x2A8])
            )
            p_info["WheelieCooldownCount"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x2B6])
            )
            p_info["LeanRot"] = self.read_f32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x294])
            )

            p_info["BitField0"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0x4])
            )
            p_info["BitField1"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0x8])
            )
            p_info["BitField2"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0xC])
            )
            p_info["BitField3"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0x10])
            )
            p_info["SurfaceFlag"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x18, 0x18, 0x2C])
            )

            p_info["Hop"] = tuple(
                self.read_fff(
                    self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x228])
                )
            )

            p_info["MTBoostTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x102])
            )
            p_info["AllMTCharge"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x10C])
            )
            p_info["MushroomBoostTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x110])
            )
            p_info["TrickableTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0xA6])
            )
            p_info["TrickCooldown"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x258, 0x38])
            )
            p_info["AirtimeCount"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0x1C])
            )

            p_info["RacePosition"] = self.read_u8(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x18, 0x3C])
            )
            p_info["FloorCollisionCount"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x18, 0x40])
            )
            p_info["RespawnTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x18, 0x18, 0x48])
            )
            p_info["WallCollideFlag"] = self.read_u32(
                self.resolve_chain(
                    0x809C18F8, [0x20, 0x4 * n, 0x0, 0x8, 0x90, 0x8, 0x8]
                )
            )

            p_info["Item"] = self.read_u32(
                self.resolve_chain(0x809C3618, [0x14] + [0xBC] * n + [0x8C])
            )
            p_info["ItemNum"] = self.read_u32(
                self.resolve_chain(0x809C3618, [0x14] + [0xBC] * n + [0x90])
            )
            p_info["PassiveItem"] = self.read_u32(
                self.resolve_chain(0x809C3618, [0x14] + [0xBC] * n + [0xCC])
            )
            p_info["PassiveItemNum"] = self.read_u32(
                self.resolve_chain(0x809C3618, [0x14] + [0xBC] * n + [0x104])
            )

            p_info["StarTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x18A])
            )
            p_info["ShockTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x18C])
            )
            p_info["BlooperInkTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x18E])
            )
            p_info["BlooperStateFlag"] = self.read_u8(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x190])
            )
            p_info["CrushTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x192])
            )
            p_info["MegaTimer"] = self.read_u16(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x28, 0x194])
            )

            p_info["startBoostCharge"] = self.read_f32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0x9C])
            )
            p_info["startBoostIdx"] = self.read_u32(
                self.resolve_chain(0x809C18F8, [0x20, 0x4 * n, 0x0, 0x4, 0xA0])
            )
            obs["PLAYER_INFO"].append(p_info)

        return obs
