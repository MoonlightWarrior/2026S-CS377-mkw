# AI-Tango's Official Wii Reinforcement Learning Repository

[https://www.youtube.com/@aitango](https://www.youtube.com/@aitango)

---

**Please watch the video which explains how to use this repository in detail!**  
[(Video)](https://www.youtube.com/watch?v=zPQTg1L1M8U)

---

## Overview

This repository contains code to allow Reinforcement Learning agents to play Wii games.  
In this repo, we provide an example of:

- A Mario Kart Wii environment, using Luigi Circuit against Hard CPUs on 150cc.
- A Reinforcement Learning algorithm (Beyond The Rainbow), which is setup and ready to interact with the environment.

This algorithm is able to get first place in approximately one day of training using an RTX4090.  
The algorithm can still be run on lighter hardware, but may take slightly longer.

**Beyond The Rainbow (BTR) algorithm, accepted at ICML 2025 (Poster):**  
[Paper!](https://openreview.net/pdf?id=V3KXsUFw8D)

---

## How to run — `feature/cpu-distill` (CPU AI behaviour cloning)

This fork hijacks MKW's input pipeline at the `KPad::mRaceInputState` layer
to capture the CPU AI's per-frame stick/buttons as supervised labels for
behaviour cloning. Three primary scripts:

| Script | Purpose |
|---|---|
| `collect_distill_data.py` | Runs Dolphin in probe mode; dumps (obs, action) pairs from CPU karts to an NPZ. |
| `viz_cpu_movement.py` | Top-down matplotlib plot of all CPU trajectories on Luigi Circuit + optional per-slot action panel + optional MP4 animation. |
| `inspect_cpu_tui.py` | Curses TUI for scrubbing through an NPZ frame-by-frame, inspecting all 78 obs fields per slot. Yellow highlight = changed since previous frame. |

### Collect a BC dataset

Tier-0 recommended recipe (8000 clean rows on Luigi Circuit mid-race
savestate `RMCP01.s08`, with DAgger-style noisy-teacher perturbation):

```bash
NUM_ENVS=1 docker compose run --rm --no-deps -T \
    -e HEADLESS=1 -e PYTHONUNBUFFERED=1 \
    wii-rl bash -c '
      cd /src/Vlab-WiiRL
      rm -f instance_info/slave_*.log alive.txt
      uv sync --quiet
      timeout 900 uv run python -u collect_distill_data.py \
          --stoch --target_rows 8000 \
          --out instance_info/distill_lc_s08_v1.npz
    '
```

Constraints baked in (don't override): `MKW_PLAY_NUM=1`,
`MKW_PROBE_INPUT=1`, default savestate `RMCP01.s08`. Records slots 4–11
(CPU); the 78-dim per-kart obs is fetched via the verified lite path.

### Visualise + debug

```bash
# Top-down trajectory plot of all CPU karts on Luigi Circuit
.venv/bin/python viz_cpu_movement.py --npz instance_info/distill_lc_s08_v1.npz

# Add action-stream side panel for one slot
.venv/bin/python viz_cpu_movement.py --npz instance_info/distill_lc_s08_v1.npz --slot 4

# MP4 animation with fading trails
.venv/bin/python viz_cpu_movement.py --npz instance_info/distill_lc_s08_v1.npz --animate

# Interactive obs scrubber (needs real terminal — don't pipe stdin)
.venv/bin/python inspect_cpu_tui.py --npz instance_info/distill_lc_s08_v1.npz
```

TUI keys: `←/→` frame, `Shift+←/→` jump 10, `↑/↓` slot, `TAB` toggle
single-slot ↔ all-slots view, `g` goto frame, `q` quit.

### See also

- **`../cmd.txt`** — every Docker recipe, including PPO 12-player runs,
  random-policy demos, and BC collection variants.
- **`../notes_new/`** — dated session logs (latest:
  `2026-05-14-obs-verification-and-bug-fixes.md`).
- **`../analysis/12phack/`** — design docs for both branches
  (`feature/12phack` input override, `feature/cpu-distill` BC pipeline).

Verified key invariants (worth knowing before extending):
- Per-kart 78-dim obs is fully covered by the lite path
  (`_lite_per_kart_obs` in `DolphinScript.py`); same-slot byte-equal
  vs. heavy. See note 2026-05-14.
- `play_num` ≥ 5 hangs Dolphin at slot 4 (RMCP01.s08 1p+11CPU savestate).
  Cap at `MKW_PLAY_NUM=1` for normal runs; can bump to 4 for
  cross-verification only.
- KPad chain `*(0x809BD730) → +0xC → 4*i → +0x48 → +0x28` is a uniform
  read interface for real-player AND CPU input — same address regardless
  of who wrote it.

---

## Installation Instructions

### 1. Prerequisites

#### Windows:

- Download **Python 3.12**  
  _(It needs to be this version as the Dolphin Scripting Fork relies on it)_  
  https://www.python.org/downloads/release/python-3120/

- Download **Visual Studio's C++ build tools package**
  `https://visualstudio.microsoft.com/downloads/`, then install `Desktop development with C++`

#### Linux:

- You need to have Python 3.12 or higher installed system-wide using your package manager. If you're using a virtual environment, the version should match the system Python version.

#### Mac OS:

- You need to have Python 3.12 or higher installed.

- To use the compiled Dolphin provided by the script, you have to have python 3.13.5 installed via Homebrew:
  ```sh
  brew update
  brew install python@3.13.5
  ```
  If you're using a virtual environment, the version should match the one that is installed.

#### Clone this repository:

```sh
git clone https://github.com/VIPTankz/Wii-RL.git
```

---

### 2. Game ROM

Once you've installed the above, you will need to download the game, Mario Kart Wii.  
We cannot legally distribute this ROM, so you will need to acquire a Mario Kart Wii ROM yourself.
When you acquire the ROM, rename it to `mkw.iso` and put it in the directory `game`.

- **We use the European RMCP01 version of the game, `mkw.iso` (4.38GB).**  
  Please install this version to avoid other potential issues.
  (MD5 checksum e7b1ff1fabb0789482ce2cb0661d986e)

---

### 3. Installing Libraries

To install the relevant libraries, please do `pip install -r requirements.txt`.
(We do also include `environment.yml` if you want to install via conda).

---

### 4. Setting Up This Repository

To correctly allow this repo to interact with Dolphin, please follow these steps:

1. Download **Felk's Fork of Dolphin**, which allows programmatic input to the emulator via Python. This can be done by running the `download_dolphin.py` script as shown below (only Windows and macOS are supported):

   ```sh
   python3 scripts/download_dolphin.py
   ```

   For Linux, you need to compile Dolphin from source. You are likely to encounter build errors; open an issue if you do. Use the script `build-dolphin-linux.sh` as shown below:

   ```sh
   bash scripts/build-dolphin-linux.sh
   ```

   To run multiple instances of Dolphin for training, you need to clone the installed Dolphin. You can use the script `clone_dolphins.py` as shown below:

   ```sh
   python3 scripts/clone_dolphins.py
   ```

2. Download the save states (which control the AI's starting position) by running the `download_savestates.py` script as shown below:
   ```sh
   python3 scripts/download_savestates.py
   ```

---

### 5. Download a Pre-Trained Model

While we provide code to train your own models, we also include a model for you to run and test. To use this model, download the pytorch model file from `https://github.com/VIPTankz/Wii-RL/releases/tag/model`, and place this in this directory.

---

### 6. Running The AI with Dolphin

To first test whether everything is set up as intended, we recommend first running `python BTR_test.py --model_path YOUR_MODEL_PATH_HERE`. This will run the pretrained model installed in the last step, in two emulators in parallel.

To test if training on your machine works quickly, you can also run `python BTR.py --testing 1`.

To actually do your own training, simply run `python BTR.py`. This will use 4 instances of dolphin by default. This will put quite some strain on most PCs, so you may want to reduce this to 2 or 1 (You can also do 8 if you have a crazy good machine and don't mind your fans going crazy).

#### Note on Mac OS:

To train using the Mac GPU, run the following command:

```sh
python BTR.py --spectral 0 --device mps
```

---

### 7. What to Expect

1. For the first 200k timesteps, the agent will simply execute a random policy, so don't expect to see any improvements during this time.
2. From 200k timesteps to 2M timesteps, the agent will slowly use fewer random actions, but during this period it may be hard to see any improvement.
3. At 2M-5M timesteps, you should be able to clearly see improvement. If not, something is messed up.
4. When testing this on an i9-13900k and RTX4090, it takes around 12 Hours to get an agent which can consistently finish the race.
5. When you start a training run, a new folder will be created (something like BTR_MarioKart2000M). This will contain a .png file with a graph of reward over time.
6. By default, the agent will run for 2 Billion frames (500 Million Timesteps). This is a LONG time, so don't be waiting around for it to finish.

Best of Luck!

---

### 8. FAQs

1. Can I run this on MacOS/Linux/{my_favourite_os}?
   Currently this has only been tested mostly on Windows, but we have also tested on Linux and MacOS. If you need anything else, considering doing a pull request.

2. Can I play games other than Mario Kart Wii?
   Currently this repository only supports a basic scenario using Mario Kart Wii on Luigi Circuit, however when doing videos I might try to add new content for people to play around with. It may take a little while, but you are welcome to attempt to use this repo to get an AI to play your favourite games.

### 9. Docker

#### 사전 준비

1. prebuilt dolphin 바이너리를 https://www.notion.so/vlab-kaist/Wii-RL-Docker-329b457b891780949854c2aee5c1d5b1 에서 찾아 다운로드하여 `HEREISFILE/dolphin-emu.tar.gz`에 위치

2. game ROM 및 savestates 준비
```bash
cp mkw.iso game/mkw.iso
uv run scripts/download_savestates.py
```

#### Headless (학습 서버, 디스플레이 없음)

```bash
HEADLESS=1 docker compose up -d --build
docker compose logs -f
```

#### GUI (호스트 디스플레이 연결)

`xhost`는 SSH에서 안 됩니다. 물리 모니터가 연결된 로컬 터미널에서 실행하세요.

```bash
# ls /tmp/.X11-unix/ 으로 디스플레이 번호 확인 (X0 → :0, X1 → :1)
export DISPLAY=:0
xhost +local:docker
# "non-network local connections being added to access control list" 한 줄만 나오면 성공
docker compose up -d --build
docker compose logs -f
```

#### 환경 수 변경 (기본 8)

```bash
NUM_ENVS=4 docker compose up -d --build
```

#### 기타

```bash
docker compose exec wii-rl bash   # 컨테이너 쉘 접속
docker compose down                # 컨테이너 중지/제거
docker rm -f wii-rl                # 컨테이너 강제 삭제
```
