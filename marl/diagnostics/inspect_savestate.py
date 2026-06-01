"""
Inspect an arbitrary Dolphin savestate: boot the env, load the state, and dump
the race setup (player count, each kart's character/vehicle/item/completion).

Usage (emulator must be free):
  PYTHONPATH=/cs377 python /cs377/marl/diagnostics/inspect_savestate.py /tmp/timetrial.bin
"""
from __future__ import annotations

import sys

from omegaconf import OmegaConf

from kart_env import KartEnvironment

from marl.action import MKWTeamAction
from marl.train_marl import build_env_options, reset_with_retry

NAMES = {0: "Mario", 1: "Baby Peach", 2: "Waluigi", 3: "Bowser", 4: "Baby Daisy",
         5: "Dry Bones", 6: "Baby Mario", 7: "Luigi", 8: "Toad", 9: "Donkey Kong",
         10: "Yoshi", 11: "Wario", 12: "Baby Luigi", 13: "Toadette", 14: "Koopa Troopa",
         15: "Daisy", 16: "Peach", 17: "Birdo", 18: "Diddy Kong", 19: "King Boo",
         20: "Bowser Jr", 21: "Dry Bowser", 22: "Funky Kong", 23: "Rosalina"}
# vehicle ids: karts 0-17, bikes 18-35
VEH = {0: "Standard Kart S", 1: "Standard Kart M", 2: "Standard Kart L",
       20: "Standard Bike L", 22: "Mach Bike", 23: "Flame Runner (Bowser Bike)"}
# item ids (subset)
ITEM = {0: "Green Shell", 1: "Red Shell", 2: "Banana", 3: "Mushroom",
        4: "Triple Mushroom", 5: "Bob-omb", 6: "Blue Shell", 20: "none/empty",
        0x14: "none", 0xFFFFFFFF: "none"}


def main():
    sav = sys.argv[1] if len(sys.argv) > 1 else "/tmp/timetrial.bin"
    cfg = OmegaConf.load("/cs377/marl/configs/mkw_2v2_mappo.yaml")
    parser = MKWTeamAction()
    repeats = cfg.env_setting.action_repeats

    env = None
    for attempt in range(5):
        try:
            env = KartEnvironment(env_id=0, options=build_env_options(cfg.env_setting))
            reset_with_retry(env, parser, None, repeats)
            break
        except RuntimeError as e:
            print(f"[boot] attempt {attempt+1}/5 failed: {e}; recreating", flush=True)
            try:
                env.close()
            except Exception:
                pass
            env = None
    if env is None:
        raise SystemExit("env never stabilized")

    print(f"loading {sav} ...", flush=True)
    env.load_file(sav)
    neutral = parser.parse_action(0)
    for _ in range(20):
        try:
            env.step({a: neutral for a in env.agents})
        except Exception:
            pass

    m = env.dolphins_mem[0]
    obs = m.read_obs(4)
    ri = obs["RACE_INFO"]
    pc = ri.get("PlayerCount")
    print("RACE_INFO:", ri, flush=True)
    print(f"==> PlayerCount = {pc}", flush=True)
    for n, p in enumerate(obs["PLAYER_INFO"]):
        c = p["CharacterID"]; k = p["KartID"]; it = p.get("Item"); itn = p.get("ItemNum")
        mc = p.get("MaxRaceCompletion")
        tag = "" if (pc is None or n < pc) else "  (beyond PlayerCount -> likely garbage)"
        print("  player {0}: char={1} ({2})  kart={3} ({4})  item={5} ({6}) num={7}  completion={8}{9}".format(
            n, c, NAMES.get(c, "?"), k, VEH.get(k, "id" + str(k)),
            it, ITEM.get(it, "?"), itn, ("%.3f" % mc) if isinstance(mc, float) else mc, tag), flush=True)
    env.close()


if __name__ == "__main__":
    main()
