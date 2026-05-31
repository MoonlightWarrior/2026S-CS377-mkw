"""
Phase 0 — Verify the items premise end-to-end.

The entire proposal hinges on item dynamics (rubber-banding, rank-conditioned
item use). Before building anything else we must confirm, in THIS environment:

  1. Karts actually RECEIVE items during the race  (Item slot becomes non-zero)
  2. The item-use action actually FIRES them        (Item slot drops to 0 right
     after we press the item button)

Strategy: drive forward; whenever an agent holds an item, issue an item-use
action (lookup indices 2/7/12 set the "X" item button). Log per agent whether
items ever appeared and whether item-actions coincided with the slot emptying.

Run inside the Vlab container:
  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/diagnostics/check_items.py \
      --config /workspace/config/config_2v2.yaml --steps 400
"""
from __future__ import annotations

import argparse
import time

from omegaconf import OmegaConf

from kart_env import KartEnvironment, OptionType
from kart_env.utils.macro_helper import (
    CCChoice, CharacterChoice, CupChoice, CourseChoice, DriftModeChoice,
    VehicleChoice, coerce_choice,
)
from kart_env.rl import KartGameState
from kart_env.rl.registration import register_components, get_action_parser

# Lookup-table indices whose controller dict presses the item button ("X").
ITEM_USE_ACTIONS = {2, 7, 12}
FORWARD_ACTION = 0     # straight + gas
USE_ITEM_ACTION = 2    # straight + gas + item (keep racing while firing)

# MKW "no item / empty slot" sentinel (inferred empirically; item ids are small).
EMPTY_ITEM = 20


def has_real_item(slot: int) -> bool:
    return 0 <= slot < EMPTY_ITEM


def build_env_options(env_cfg) -> OptionType:
    return OptionType(
        num_agents=env_cfg.num_agents,
        character=[coerce_choice(c, CharacterChoice) for c in env_cfg.character],
        vehicle=[coerce_choice(v, VehicleChoice) for v in env_cfg.vehicle],
        drift_modes=[coerce_choice(d, DriftModeChoice) for d in env_cfg.drift_modes],
        cup=coerce_choice(env_cfg.cup, CupChoice),
        course=coerce_choice(env_cfg.course, CourseChoice),
        cc=coerce_choice(env_cfg.cc, CCChoice),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/workspace/config/config_2v2.yaml")
    ap.add_argument("--steps", type=int, default=400, help="policy steps to probe")
    ap.add_argument("--item-button", default="L",
                    help="controller button to test for item use (L is the MKW GCN item button; "
                         "X is look-behind)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    register_components()
    action_parser = get_action_parser(cfg.model.action_parser)
    repeats = int(cfg.env_setting.action_repeats)

    env = KartEnvironment(env_id=cfg.env_setting.env_id,
                          options=build_env_options(cfg.env_setting))

    # The memory reader can raise transiently while the race is still loading.
    # Retry reset, then take a few neutral warmup steps until reads are stable.
    neutral = action_parser.parse_action(FORWARD_ACTION)
    obs_dict = None
    for attempt in range(5):
        try:
            obs_dict, _ = env.reset()
            agents = list(env.agents)
            for _ in range(10):  # settle: let the race finish loading
                for _ in range(repeats):
                    obs_dict, *_ = env.step({a: neutral for a in agents})
            state = KartGameState.from_obs_dict(obs_dict)
            break
        except ValueError:
            print(f"[warmup] memory not ready (attempt {attempt+1}/5), retrying...", flush=True)
            time.sleep(2.0)
    else:
        raise RuntimeError("env memory never stabilized after reset — race did not load")
    agents = list(env.agents)

    ever_had_item = {a: False for a in agents}
    item_frames = {a: 0 for a in agents}          # frames holding a REAL item
    box_pickups = {a: 0 for a in agents}          # empty -> real transitions
    use_attempts = {a: 0 for a in agents}         # pressed X while holding real item
    confirmed_fires = {a: 0 for a in agents}      # real -> empty right after pressing X
    rank_seen = {a: set() for a in agents}
    items_seen = {a: set() for a in agents}       # distinct raw item values
    max_completion = {a: 0.0 for a in agents}

    prev_item = {a: int(state.players[a].item) for a in agents}
    pressed_use = {a: False for a in agents}

    for step in range(args.steps):
        # Drive FORWARD to race & reach item boxes; fire only when truly holding one,
        # by pressing the configured item button on top of a forward input.
        kart_actions = {}
        for a in agents:
            held = int(state.players[a].item)
            d = action_parser.parse_action(FORWARD_ACTION)
            if has_real_item(held):
                d[args.item_button] = 1          # press the candidate item button
                pressed_use[a] = True
                use_attempts[a] += 1
            else:
                pressed_use[a] = False
            kart_actions[a] = d
        try:
            for _ in range(repeats):
                obs_dict, _, terminations, truncations, _ = env.step(kart_actions)
            state = KartGameState.from_obs_dict(obs_dict)
        except ValueError:
            # transient memory read during a load/reset boundary — skip this step
            continue

        for a in agents:
            held = int(state.players[a].item)
            items_seen[a].add(held)
            rank_seen[a].add(int(state.players[a].race_position))
            max_completion[a] = max(max_completion[a], state.players[a].max_race_completion)
            if has_real_item(held):
                ever_had_item[a] = True
                item_frames[a] += 1
            # box pickup: empty -> real
            if not has_real_item(prev_item[a]) and has_real_item(held):
                box_pickups[a] += 1
            # fire: held real, pressed X, now empty
            if pressed_use[a] and has_real_item(prev_item[a]) and not has_real_item(held):
                confirmed_fires[a] += 1
            prev_item[a] = held

        if step % 50 == 0:
            snap = {a: int(state.players[a].item) for a in agents}
            comp = {a: round(state.players[a].max_race_completion, 2) for a in agents}
            print(f"[step {step:4d}] items={snap}  completion={comp}  ranks="
                  f"{ {a: int(state.players[a].race_position) for a in agents} }",
                  flush=True)

        if any(terminations.values()) or any(truncations.values()):
            print(f"[info] episode ended at step {step}")
            break

    print("\n================ ITEM VERIFICATION SUMMARY ================")
    any_items = any(ever_had_item.values())
    any_fire = any(confirmed_fires[a] > 0 for a in agents)
    progressed = any(max_completion[a] > 0.05 for a in agents)
    for a in agents:
        print(f"  agent {a}: ever_item={ever_had_item[a]}  box_pickups={box_pickups[a]}  "
              f"item_frames={item_frames[a]}  use_attempts={use_attempts[a]}  "
              f"confirmed_fires={confirmed_fires[a]}  completion={max_completion[a]:.2f}")
        print(f"            items_seen={sorted(items_seen[a])}  ranks_seen={sorted(rank_seen[a])}")
    print("-----------------------------------------------------------")
    print(f"  KARTS RACE    : {'YES' if progressed else 'NO  (not progressing — driving/menu issue)'}")
    print(f"  ITEMS APPEAR  : {'YES' if any_items else 'NO  (items DISABLED, or never reached a box)'}")
    print(f"  ITEMS FIRE    : {'YES' if any_fire else 'NO  (button mapping wrong, or no items held)'}")
    print(f"  field ranks observed across agents: "
          f"{sorted(set().union(*rank_seen.values()))}  (1-4 => clean 2v2, 1-12 => CPUs present)")
    print("===========================================================")

    env.close()


if __name__ == "__main__":
    main()
