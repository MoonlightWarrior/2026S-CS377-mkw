# Running standalone (clone-and-go)

This repository is **self-contained**: it vendors the `kart_env/` environment
engine and its `Dockerfile` builds the Dolphin emulator from source, so you do
**not** need the separate Vlab-Kart-env repo to run the 2v2 MAPPO self-play.

## Prerequisites
- Docker + the **NVIDIA Container Toolkit**, and an NVIDIA GPU.
- A legally-obtained **Mario Kart Wii ISO** (RMCP01). Not included (git-ignored).
- **~40 GB free disk** for the image (Dolphin is compiled into it).

## Steps
```bash
git clone https://github.com/MoonlightWarrior/2026S-CS377-mkw.git
cd 2026S-CS377-mkw

# 1. drop your ISO here
cp /path/to/MarioKartWii.iso game/MarioKartWii.iso

# 2. (optional) Weights & Biases
export WANDB_API_KEY=...        # omit, or pass --no-wandb to train

# 3. build + start (FIRST BUILD ~20-30 min: it compiles Dolphin)
docker compose -f compose.marl.yml up -d --build

# 4. train (2v2 MAPPO self-play, rank reward, clean SIGTERM shutdown)
docker compose -f compose.marl.yml exec marl bash -lc \
  'cd /workspace && python -m marl.train_marl --config marl/configs/mkw_2v2_mappo.yaml'

# 5. watch the emulator live in a browser:  http://localhost:6080
#    or follow the wandb run printed at startup.

# 6. analysis / evaluation
docker compose -f compose.marl.yml exec marl bash -lc \
  'cd /workspace && python marl/analysis/evaluate.py \
     --checkpoint marl/results/checkpoints/<run>/checkpoint_XXXX.pt --episodes 10'
```

## What's verified vs. not
- **Verified here:** the vendored `kart_env` imports correctly and all five
  `marl` unit-test suites pass against it; setuptools discovers every package;
  `pyproject.toml` is valid.
- **Not built here:** the Docker image was **not** compiled in our environment
  (disk was at 95%). The `Dockerfile` mirrors the proven Vlab-Kart-env build
  recipe (same base image, apt deps, and Dolphin cmake flags), with the only
  change being a `git clone` of the Dolphin source instead of a build context.
  The first `--build` may need minor tweaks on a fresh host.

## Known follow-ups
- **4-kart (CPUs-off) race** for clean rank-1-vs-4 analysis is still pending —
  see `marl/results/FOURKART_SETUP.md`.
- `kart_env/` is vendored from Vlab-KAIST and Dolphin is GPL — see
  `THIRD_PARTY_NOTICES.md`.

## Note on the other compose file
`compose.yml` is the **single-agent BTR baseline** (uses a prebuilt image, not
clone-and-go). For the MARL project use `compose.marl.yml` as above.
