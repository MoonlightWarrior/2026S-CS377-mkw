"""
CS377 Team#3 — MARL for Mario Kart Wii.

Our contributions on top of the Vlab-Kart-env environment engine:
  - obs.py        : information-asymmetric actor observation + centralized critic state (Phase 1)
  - algo (TODO)   : MAPPO with centralized critic / CTDE (Phase 2)
  - rewards (TODO): rank-based individual + team-success reward (Phase 3)
  - selfplay(TODO): frozen-snapshot opponent pool (Phase 4)
  - analysis(TODO): rank-conditioned behavior + team-coordination metrics (Phase 5)

The environment (KartEnvironment, KartGameState, memory reading, Dolphin) is
imported from the `kart_env` package provided by Vlab-Kart-env.
"""
