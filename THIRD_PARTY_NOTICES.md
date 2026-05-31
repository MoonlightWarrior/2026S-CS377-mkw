# Third-party components

This repository vendors and builds third-party software so it can run standalone.

## `kart_env/` — vendored environment engine
The `kart_env/` package (4-player Mario Kart Wii environment: `KartEnvironment`,
memory reading, menu macros, action/obs/reward registration) is **vendored from
the Vlab-Kart-env project (Vlab-KAIST)**. It is included here so this repository
is self-contained for the course submission.

No upstream license file was present in the source at vendoring time. Team 3 is
affiliated with the Vlab group; if this repository is to be published, confirm
the intended license/permission with the Vlab-KAIST maintainers first.

## Dolphin (built from source in the Docker image)
The `Dockerfile` clones and builds the **Vlab-dolphin** scripting fork
(`https://github.com/vlab-kaist/Vlab-dolphin`), which derives from Felk's
Dolphin Python-scripting fork and ultimately from the **Dolphin emulator**
(GPL-2.0-or-later). Dolphin source is not redistributed in this repository; it is
fetched and compiled at image-build time. Dolphin's license applies to the built
binary.

## Beyond the Rainbow (BTR)
The single-agent baseline files (`BTR.py`, `DolphinEnv.py`, `DolphinScript.py`)
originate from the AI-Tango / VIPTankz "Wii-RL" (Beyond the Rainbow) repository,
which this repository was forked from.

## Game ROM
The Mario Kart Wii ISO is **not** included or redistributed (it is git-ignored).
You must supply your own legally-obtained copy at `game/MarioKartWii.iso`.
