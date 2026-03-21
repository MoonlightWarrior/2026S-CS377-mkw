import os
from pathlib import Path
import platform

from common import download_file, extract_zip, extract_tar

DOLPHIN_WIN_URL = "https://github.com/VIPTankz/Wii-RL/releases/download/dolphin/dolphin0.zip"
DOLPHIN_MAC_ARM_URL = "https://github.com/unexploredtest/dolphin/releases/download/dolphin-wii-rl/DolphinMacArm.tar.gz"
DOLPHIN_MAC_X86_URL = "https://github.com/unexploredtest/dolphin/releases/download/dolphin-wii-rl/DolphinMacx86.tar.gz"
ZIP_NAME = "Dolphin.zip"
TAR_NAME = "Dolphin.tar.gz"

def main():
    current_directory = Path.cwd()

    if(platform.system() == "Windows"):
        download_file(DOLPHIN_WIN_URL, ZIP_NAME)
        extract_zip(ZIP_NAME, current_directory)
        os.remove(ZIP_NAME)
    elif(platform.system() == "Darwin" and platform.machine() == "arm64"):
        download_file(DOLPHIN_MAC_ARM_URL, TAR_NAME)
        extract_tar(TAR_NAME, current_directory)
        os.remove(TAR_NAME)
    elif(platform.system() == "Darwin" and platform.machine() == "x86_64"):
        download_file(DOLPHIN_MAC_X86_URL, TAR_NAME)
        extract_tar(TAR_NAME, current_directory)
        os.remove(TAR_NAME)
    else:
        raise RuntimeError(f"The operating system '{platform.system()}' is not supported.")

    print("Done!")


if __name__ == "__main__":
    main()