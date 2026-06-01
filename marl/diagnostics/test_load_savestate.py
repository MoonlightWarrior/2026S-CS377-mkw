"""
Empirically test whether the uploaded 4xFunky Dolphin savestate (built by a
different Dolphin: scripting-preview3-2361) loads in OUR build (main fc7e655f89).

Boots the env (which menu-navs to the configured chars), records the characters,
then load_file()s the uploaded state and re-reads the characters:
  - all become Funky Kong (22) -> SUCCESS, and we re-save under our build.
  - unchanged / error            -> Dolphin rejected it (build mismatch).

Run with the emulator free (training stopped):
  PYTHONPATH=/cs377 python /cs377/marl/diagnostics/test_load_savestate.py
"""
from __future__ import annotations

from omegaconf import OmegaConf

from kart_env import KartEnvironment

from marl.action import MKWTeamAction
from marl.train_marl import build_env_options, reset_with_retry

NAMES = {0: "Mario", 1: "Baby Peach", 2: "Waluigi", 3: "Bowser", 4: "Baby Daisy",
         5: "Dry Bones", 6: "Baby Mario", 7: "Luigi", 8: "Toad", 9: "Donkey Kong",
         10: "Yoshi", 11: "Wario", 12: "Baby Luigi", 13: "Toadette", 14: "Koopa Troopa",
         15: "Daisy", 16: "Peach", 17: "Birdo", 18: "Diddy Kong", 19: "King Boo",
         20: "Bowser Jr", 21: "Dry Bowser", 22: "Funky Kong", 23: "Rosalina"}

SAV = "/tmp/funky.sav"
OUT = "/cs377/marl/results/startstates/funky_4p_ourbuild.sav"


def chars(env):
    obs = env.dolphins_mem[0].read_obs(env.options.num_agents)
    return [p["CharacterID"] for p in obs["PLAYER_INFO"]]


def fmt(cs):
    return [(c, NAMES.get(c, "?")) for c in cs]


def main():
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
        raise SystemExit("env never stabilized after 5 attempts")

    base = chars(env)
    print("BEFORE load:", fmt(base), flush=True)

    print(f"attempting env.load_file({SAV}) ...", flush=True)
    neutral = parser.parse_action(0)
    try:
        env.load_file(SAV)
        for _ in range(60):
            try:
                env.step({a: neutral for a in env.agents})
            except Exception:
                pass
        after = chars(env)
        print("AFTER load: ", fmt(after), flush=True)

        if after == base:
            print("RESULT: UNCHANGED -> savestate did NOT load (build-rejected).", flush=True)
        elif len(set(after)) == 1 and after[0] == 22:
            print("RESULT: SUCCESS -> all four are Funky Kong; savestate loaded.", flush=True)
            env.save_file(OUT)
            print(f"re-saved under OUR build -> {OUT}", flush=True)
        else:
            print(f"RESULT: changed but not all-Funky -> {after}", flush=True)
    except Exception as e:
        print("LOAD RAISED:", repr(e), flush=True)

    env.close()


if __name__ == "__main__":
    main()
