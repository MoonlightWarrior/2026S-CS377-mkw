# Phase 0 — Environment verification findings

Verified with `marl/diagnostics/check_items.py` against `config/config_2v2.yaml`
(Luigi Circuit, 100cc) inside the Vlab-Kart-env container.

## Results

| Check | Result | Evidence |
|---|---|---|
| Karts race / progress | ✅ YES | `max_race_completion` 0.95 → 1.19 over 300 steps driving forward |
| Items enabled (boxes award items) | ✅ YES | agents picked up real item slots (ids 5, 9, 10) — slot leaves the empty sentinel |
| Items can be fired | ✅ YES — **with L only** | `--item-button L`: `confirmed_fires ≥ 1`; `--item-button X` (the default lookup table): `confirmed_fires = 0` |

## Key findings → required fixes

1. **Item button is `L`, not `X`.** MKW on a GameCube controller uses **L = use item**
   (X = look-behind). The Vlab `DiscreteLookupAction` presses `X`, so karts pick up
   items but never use them. → Fixed in **`marl/action.py` (`MKWTeamAction`)**, which
   maps the item column to `L`. Use this parser everywhere instead of `DiscreteLookupAction`.

2. **Item slot empty sentinel = `20`.** `PlayerState.item == 20` means "no item".
   Real items are ids `0–19`. Reward/analysis code must treat 20 as empty.

3. **Field is 12 karts, not 4.** Observed `race_position` values span 8–12 → the race
   has **8 CPUs + our 4 agents**. The proposal's "rank 1 vs rank 4" analysis assumes a
   clean **4-kart 2v2 (ranks 1–4, CPUs off)**. → ENV CONFIG DECISION NEEDED: produce a
   VS/Team-Race save state with CPUs disabled (4 karts total), or adapt the rank
   reward/analysis to a 12-kart field.

4. **Intermittent boot flakiness.** ~1 in 3 boots the memory reader raises `ValueError`
   (game pointers not initialized; race didn't finish loading). → The training loop must
   wrap `env.reset()` in a retry + warmup (pattern demonstrated in `check_items.py`).
