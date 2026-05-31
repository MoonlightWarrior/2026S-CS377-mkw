import numpy as np
from multiprocessing import shared_memory
import struct
from typing import List
from PIL import Image


class KartGraphicObs:
    def __init__(self, instance_id: int):
        self.shm_name = f"DolphinSharedFrameBuffer_{instance_id}"
        self.NUM_SHM_FRAMES = 7
        self.HEADER_METADATA_SIZE = 40
        self.FRAME_INFO_SIZE = 32
        self.TOTAL_HEADER_SIZE = self.HEADER_METADATA_SIZE + (
            self.FRAME_INFO_SIZE * self.NUM_SHM_FRAMES
        )
        self.EXPECTED_MAGIC = 0x444C464E
        self.VERSION = 2

        self.shm = shared_memory.SharedMemory(name=self.shm_name)

    def get(self) -> List[np.ndarray]:
        assert self.shm.buf is not None
        buf = self.shm.buf

        metadata = struct.unpack("IIIIQQII", buf[: self.HEADER_METADATA_SIZE])
        magic, version, num_frames, ready, frame_number, timestamp, _, _ = metadata
        assert magic == self.EXPECTED_MAGIC and version == self.VERSION

        while ready == 0:
            ready = struct.unpack("I", buf[12:16])[0]

        images = []
        for i in range(self.NUM_SHM_FRAMES):
            offset = self.HEADER_METADATA_SIZE + (i * self.FRAME_INFO_SIZE)
            f_info = struct.unpack(
                "IIIIIIII", buf[offset : offset + self.FRAME_INFO_SIZE]
            )
            width, height, stride, fmt, data_offset, data_size, _, _ = f_info

            img_view = np.ndarray(
                shape=(height, width, 4),
                dtype=np.uint8,
                buffer=buf[data_offset : data_offset + data_size],
                strides=(stride, 4, 1),
            )
            images.append(img_view)
            # Maybe we need to copy the image, idk

        return images

    def close(self):
        self.shm.close()

    def __del__(self):
        self.close()


def save_graphic_obs(observation: List[np.ndarray]):
    for i, obs_array in enumerate(observation, start=1):
        img = Image.fromarray(obs_array)
        filename = f"image_{i}.png"
        img.save(filename)
