# CS377 Team#3 — MARL for Mario Kart Wii

Implementation of the proposal *"MARL for Mario Kart Wii"*: 2v2 self-play with a
single shared policy, **MAPPO** (centralized critic / CTDE), memory-state
observations, and rank-based team rewards — studying rank-differentiated item
behavior and team cooperation under MKW's rubber-banding.

## Requirements

`marl/` imports `kart_env` (the 4-player Dolphin environment). That engine is now
**vendored into this repo** at [`../kart_env/`](../kart_env), so the project is
self-contained — see [`../STANDALONE.md`](../STANDALONE.md) for the clone-and-go
Docker build (`compose.marl.yml`). You still need a Mario Kart Wii ISO and a GPU.
The unit tests under `tests/` import `kart_env`, so run them inside the built
container (or any env where the repo is `pip install -e .`'d).

## Architecture

This package holds **our contributions**. The environment engine (4-player
Dolphin, memory reading, `KartEnvironment` / `KartGameState`, action parser) is
reused from **Vlab-Kart-env** via the installed `kart_env` package — the same
Dolphin/memory lineage as BTR, extended to 4 local players.

```
2026S-CS377-mkw/marl/
  obs.py                     # Phase 1 ✓  asymmetric actor obs + centralized critic state
  diagnostics/check_items.py # Phase 0     verify items appear & fire
  tests/test_obs_dims.py     # Phase 1 ✓  offline dim + asymmetry self-test
  configs/mkw_2v2_mappo.yaml  # full run spec (MAPPO + self-play)
  algo/mappo.py              # Phase 2 (TODO) centralized-critic MAPPO learner
  rewards/rank_reward.py     # Phase 3 (TODO) rank + team-success reward
  selfplay/snapshot_pool.py  # Phase 4 (TODO) frozen-opponent pool
  analysis/metrics.py        # Phase 5 (TODO) rank-conditioned + coordination metrics
  train_marl.py              # Phase 2 (TODO) entrypoint
  results/                   # committed deliverables (plots, CSVs)
```

## How to run (inside the Vlab container, with this repo mounted at /cs377)

```bash
cd ~/Vlab-Kart-env
docker compose up -d --no-build          # /cs377 mount is in compose.override.yml

# Phase 1 — offline obs self-test (no Dolphin needed)
docker compose exec kart_env bash -lc \
  'PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/tests/test_obs_dims.py'

# Phase 0 — verify items end-to-end (boots Dolphin)
docker compose exec kart_env bash -lc \
  'PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/diagnostics/check_items.py --steps 400'
```

## Training and analysis

```bash
# train (full self-play MAPPO; wandb on, clean SIGTERM shutdown)
docker compose exec -d kart_env bash -lc \
  'PYTHONPATH=/cs377:$PYTHONPATH python -u /cs377/marl/train_marl.py \
     --config /cs377/marl/configs/mkw_2v2_mappo.yaml > /workspace/users/train_mappo.log 2>&1'

# stop cleanly (saves a final checkpoint + finishes wandb)
docker compose exec kart_env pkill -TERM -f train_marl.py

# pull the run's curves into a report-ready CSV (+ PNGs)
docker compose exec kart_env bash -lc \
  'PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/analysis/fetch_wandb.py \
     --run <entity>/<project>/<run_id> --window 20'

# behavioral analysis (RQ1/RQ2) from a checkpoint -> results/analysis/<run>/summary.csv + plots
docker compose exec kart_env bash -lc \
  'PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/analysis/evaluate.py \
     --checkpoint results/checkpoints/<run>/checkpoint_XXXX.pt --episodes 10'
```

## Observation design (Phase 1)

| View | Builder | Size | Contents |
|---|---|---|---|
| Actor (execution) | `AsymmetricTeamObs` | 59 | full self (25) + full teammate (25) + 2× coarse opponent (4) + team flag (1) |
| Critic (training) | `CentralizedTeamState` | 102 | full self + teammate + both opponents (4×25) + race meta (2) |

The actor never sees opponent item/velocity/charge — only their relative
position and rank, matching what a human reads off screen/minimap. The critic
sees everything (CTDE).

## Phase status

- [x] **Phase 0** — item verification ✅ see [results/PHASE0_FINDINGS.md](results/PHASE0_FINDINGS.md).
      Items work; **item button is L not X** (fixed in `action.py`); field is 12 karts
      (CPUs on) — needs a 4-kart save state for clean rank-1..4 analysis.
- [x] **Phase 1** — asymmetric obs + centralized state (`obs.py`, self-test passing)
- [x] **Phase 2** — MAPPO centralized-critic learner (`algo/mappo.py` + `train_marl.py`).
      Offline test passes; ran end-to-end against the real env (iters 1–3);
      SIGTERM clean-shutdown + checkpoint-to-repo verified. Reward is an INTERIM
      progress+rank stub — replaced in Phase 3.
- [x] **Phase 3** — rank-based team reward (`rewards/rank_reward.py`, wired into
      `train_marl.py`, offline test passing). Potential-based individual `RankReward`
      + shared `TeamRankReward` + `FinishReward`/`LapReward` + light speed/MT shaping.
- [x] **Phase 4** — self-play snapshot pool (`selfplay/snapshot_pool.py`, wired into
      `train_marl.py`, offline test passing). Learner team `[0,1]` vs frozen snapshot
      `[2,3]`; trains only learner trajectories; logs `win_vs_snapshot`.
- [x] **Phase 5** — analysis & evaluation harness (`analysis/metrics.py` +
      `analysis/evaluate.py`, offline test passing). RQ1 rank-conditioned item
      holding/use-rate, overtakes, mini-turbo, and an RQ2 opponent-hit/teamkill
      proxy; `evaluate.py` loads a checkpoint and writes `summary.csv` + plots.

### Open env-config decision (blocks clean rank analysis)
The proposal reasons about "rank 1 vs rank 4" → needs a **4-kart 2v2** race (CPUs off,
ranks 1–4). The current save state is a 12-kart field. Either author a CPUs-off
save state, or define rank reward/analysis over the 12-kart field.
