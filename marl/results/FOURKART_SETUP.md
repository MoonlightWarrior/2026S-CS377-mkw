# 4-kart (CPUs-off) race setup — handoff

The proposal's RQ1 ("rank 1 vs rank 4") needs a clean **4-kart 2v2** race so ranks
are 1–4. The current environment produces a **12-kart field** (4 agents + 8 CPUs,
ranks 8–12).

## Root cause (located)

The race is configured by a menu-navigation macro, not a saved file. In the
environment's `src/kart_env/utils/helper.py`, function `_launch_game_4p`, right
after selecting **VS Race** there is:

```python
env.click({0: {"A": 1}}, num_frame=100)  # select VS Race in ["VS Race", "Battle"]
# TODO: setting rules (CC, CPU, etc.) in the future
```

The macro accepts MKW's **default** VS rules, which include CPUs → 12 karts. No
`OptionType` field exists for CPU count, so this cannot be set from config today.

## What needs to happen

Extend `_launch_game_4p` (and add a `cpus_off`/`num_karts` field to `OptionType`)
to navigate the VS **settings screen** and set **CPU: None** before confirming.
In MKW's settings screen the CPU row is selected by D-pad and toggled
left/right; the exact number of presses must be confirmed visually.

## Why this needs a visual pass

Blind menu navigation is unreliable (no other team on this server has automated
it). Verify interactively via the already-exposed noVNC:

1. With a Dolphin running, open **http://localhost:7080** in a browser.
2. Watch the VS settings screen; note the D-pad path to set **CPU → None**.
3. Encode that path as `env.click(...)` steps in `_launch_game_4p`.
4. Re-run `marl/diagnostics/check_items.py`; success = the
   `field ranks observed` line reads `[1, 2, 3, 4]` (not `[8..12]`).

Once ranks are 1–4, all of training (`train_marl.py`) and analysis
(`analysis/evaluate.py`) work unchanged — they already normalize by the live
field size.

## Interim

Until then, training runs on the 12-kart field. Learning, self-play win-rate, and
the analysis harness all function; only the *cleanliness* of the rank-1-vs-4
contrast is affected (ranks are 8–12 rather than 1–4).
