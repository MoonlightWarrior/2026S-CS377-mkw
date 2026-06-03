"""
Phase 1 — Information-asymmetric observations for 2v2 MAPPO.

Two builders, matching the proposal's CTDE design:

  AsymmetricTeamObs   (ACTOR  / decentralized execution)
      Per-agent restricted view:
        - FULL self state
        - FULL teammate state          (perfect intra-team communication)
        - COARSE opponents only        (relative position + rank — what a human
                                         reads off screen/minimap)
        - team indicator feature
      Opponent item / velocity / charge are deliberately HIDDEN.

  CentralizedTeamState (CRITIC / centralized training)
      Ego-centric global state: FULL state of all four karts (self, teammate,
      both opponents) + race metadata. Used only by the value function during
      training, never at execution.

Both consume a `kart_env.rl.game_state.KartGameState`. Team layout is the
4-agent 2v2 convention used throughout the env: agents [0,1] vs [2,3].
"""
from __future__ import annotations

from typing import List

import numpy as np

from kart_env.rl.game_state import AgentID, KartGameState, PlayerState
from kart_env.rl.obs_builder import ObsBuilder


# ── normalization constants (shared by actor & critic) ──────────────────────
SPEED_NORM = 120.0
POS_NORM = 10_000.0
ANG_VEL_NORM = 3.0
ITEM_NORM = 32.0      # item slot id is a small enum; crude scale, fine as a feature
CHARGE_NORM = 270.0   # mini-turbo / super mini-turbo charge max
BOOST_NORM = 100.0    # boost-timer frames
FIELD_SIZE = 12.0     # MKW always tracks 12 race positions
FRAME_NORM = 6000.0   # ~ a few minutes of frames, for race-clock feature

# Per-player block sizes
FULL_DIM = 25         # complete physics + item + boost state for one kart
COARSE_DIM = 4        # relative position (3) + rank (1)


def _full_block(p: PlayerState, me: PlayerState, *, relative_pos: bool) -> List[float]:
    """25-dim complete state for one kart. Position is relative to `me` unless
    this *is* `me` (then absolute, normalized)."""
    if relative_pos:
        px = (p.position[0] - me.position[0]) / POS_NORM
        py = (p.position[1] - me.position[1]) / POS_NORM
        pz = (p.position[2] - me.position[2]) / POS_NORM
    else:
        px = p.position[0] / POS_NORM
        py = p.position[1] / POS_NORM
        pz = p.position[2] / POS_NORM
    return [
        p.current_race_completion / 3.0,
        p.max_race_completion / 3.0,
        p.current_lap / 3.0,
        p.speed / SPEED_NORM,
        px, py, pz,
        p.velocity[0] / SPEED_NORM,
        p.velocity[1] / SPEED_NORM,
        p.velocity[2] / SPEED_NORM,
        p.main_rotation[0], p.main_rotation[1], p.main_rotation[2], p.main_rotation[3],
        p.angular_velocity[0] / ANG_VEL_NORM,
        p.angular_velocity[1] / ANG_VEL_NORM,
        p.angular_velocity[2] / ANG_VEL_NORM,
        p.drift_state / 3.0,
        p.race_position / FIELD_SIZE,
        min(p.item / ITEM_NORM, 1.0),
        min(p.miniturbo_charge / CHARGE_NORM, 1.0),
        min(p.smt_charge / CHARGE_NORM, 1.0),
        min(p.mt_boost_timer / BOOST_NORM, 1.0),
        min(p.mushroom_boost_timer / BOOST_NORM, 1.0),
        float(p.start_boost_charge),
    ]


def _coarse_block(o: PlayerState, me: PlayerState) -> List[float]:
    """4-dim coarse opponent view: relative position + rank only."""
    return [
        (o.position[0] - me.position[0]) / POS_NORM,
        (o.position[1] - me.position[1]) / POS_NORM,
        (o.position[2] - me.position[2]) / POS_NORM,
        o.race_position / FIELD_SIZE,
    ]


def _team_of(agent_id: AgentID) -> int:
    return int(agent_id) // 2


def _teammate_id(agent_id: AgentID, agents: List[AgentID]) -> AgentID:
    team = _team_of(agent_id)
    for a in agents:
        if a != agent_id and _team_of(a) == team:
            return a
    raise ValueError(f"no teammate for agent {agent_id} in {agents}")


def _opponent_ids(agent_id: AgentID, agents: List[AgentID]) -> List[AgentID]:
    team = _team_of(agent_id)
    return sorted(a for a in agents if _team_of(a) != team)


#: opponent-observability ablation modes (Section "Scope" of the report).
#: how much of each OPPONENT the actor sees — the only thing that varies across
#: the ablation; self + teammate stay full and the critic stays fully global.
OPPONENT_OBS_MODES = ("none", "coarse", "full")
_OPP_BLOCK_DIM = {"none": 0, "coarse": COARSE_DIM, "full": FULL_DIM}


class AsymmetricTeamObs(ObsBuilder):
    """Actor observation (decentralized execution view).

    `opponent_obs` selects the opponent-observability ablation condition:
      "none"   — opponents not observed at all (only self + teammate + team id);
                 directed offense at opponents can then only be chance.
      "coarse" — DEFAULT; each opponent as relative position + rank (4-d), with
                 item/velocity/charge hidden (what a human reads off-screen).
      "full"   — each opponent as the complete 25-d state (enables sharp,
                 velocity/item-aware targeting).
    Self and teammate are always FULL; only the opponent block changes, so the
    three conditions differ purely in opponent information.
    """

    def __init__(self, num_agents: int = 4, opponent_obs: str = "coarse"):
        assert num_agents == 4, "AsymmetricTeamObs is defined for 2v2 (4 agents)"
        assert opponent_obs in OPPONENT_OBS_MODES, (
            f"opponent_obs must be one of {OPPONENT_OBS_MODES}, got {opponent_obs!r}")
        self.num_agents = num_agents
        self.opponent_obs = opponent_obs

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    def get_obs_size(self) -> int:
        # self full + teammate full + 2 opponents (size by mode) + team indicator
        return FULL_DIM + FULL_DIM + 2 * _OPP_BLOCK_DIM[self.opponent_obs] + 1

    def build_obs(self, agent_id: AgentID, state: KartGameState) -> np.ndarray:
        agents = list(state.players.keys())
        me = state.players[agent_id]
        mate = state.players[_teammate_id(agent_id, agents)]
        opps = [state.players[o] for o in _opponent_ids(agent_id, agents)]

        vec: List[float] = []
        vec += _full_block(me, me, relative_pos=False)
        vec += _full_block(mate, me, relative_pos=True)
        for o in opps:
            if self.opponent_obs == "coarse":
                vec += _coarse_block(o, me)
            elif self.opponent_obs == "full":
                vec += _full_block(o, me, relative_pos=True)
            # "none": opponents contribute nothing
        vec += [float(_team_of(agent_id))]  # team indicator

        out = np.asarray(vec, dtype=np.float32)
        assert out.shape[0] == self.get_obs_size(), (out.shape, self.get_obs_size())
        return out


class CentralizedTeamState(ObsBuilder):
    """Critic global state (centralized training view, ego-centric per agent)."""

    def __init__(self, num_agents: int = 4):
        assert num_agents == 4, "CentralizedTeamState is defined for 2v2 (4 agents)"
        self.num_agents = num_agents

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    def get_obs_size(self) -> int:
        # full state of self + teammate + 2 opponents + race metadata (2)
        return 4 * FULL_DIM + 2

    def build_obs(self, agent_id: AgentID, state: KartGameState) -> np.ndarray:
        agents = list(state.players.keys())
        me = state.players[agent_id]
        mate = state.players[_teammate_id(agent_id, agents)]
        opps = [state.players[o] for o in _opponent_ids(agent_id, agents)]

        vec: List[float] = []
        vec += _full_block(me, me, relative_pos=False)
        vec += _full_block(mate, me, relative_pos=True)
        for o in opps:
            vec += _full_block(o, me, relative_pos=True)
        vec += [
            state.race_info.frame_count / FRAME_NORM,
            state.race_info.player_count / FIELD_SIZE,
        ]

        out = np.asarray(vec, dtype=np.float32)
        assert out.shape[0] == self.get_obs_size(), (out.shape, self.get_obs_size())
        return out
