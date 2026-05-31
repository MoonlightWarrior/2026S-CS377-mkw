from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

from .game_state import AgentID, KartGameState


class ActionParser(ABC):
    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    @abstractmethod
    def parse_action(self, action: np.ndarray | int) -> Dict[str, Any]:
        pass


# fmt: off
# Each row: (StickX, A, B, R, X, Up, Down)
_TABLE = [
    # ── straight ─────────────────────────────────────
    ( 0.0,  1, 0, 0, 0, 0, 0),  # 0  straight + gas
    ( 0.0,  1, 0, 0, 0, 1, 0),  # 1  straight + gas + trick/wheelie
    ( 0.0,  1, 0, 0, 1, 0, 0),  # 2  straight + gas + item
    # ── steer left ───────────────────────────────────
    (-0.5,  1, 0, 0, 0, 0, 0),  # 3  slight left + gas
    (-1.0,  1, 0, 0, 0, 0, 0),  # 4  hard left + gas
    (-0.5,  1, 0, 1, 0, 0, 0),  # 5  slight left + gas + drift
    (-1.0,  1, 0, 1, 0, 0, 0),  # 6  hard left + gas + drift
    (-1.0,  1, 0, 0, 1, 0, 0),  # 7  hard left + gas + item
    # ── steer right ──────────────────────────────────
    ( 0.5,  1, 0, 0, 0, 0, 0),  # 8  slight right + gas
    ( 1.0,  1, 0, 0, 0, 0, 0),  # 9  hard right + gas
    ( 0.5,  1, 0, 1, 0, 0, 0),  # 10 slight right + gas + drift
    ( 1.0,  1, 0, 1, 0, 0, 0),  # 11 hard right + gas + drift
    ( 1.0,  1, 0, 0, 1, 0, 0),  # 12 hard right + gas + item
    # ── trick / wheelie while steering ──────────────
    (-0.5,  1, 0, 0, 0, 1, 0),  # 13 slight left + gas + trick
    ( 0.5,  1, 0, 0, 0, 1, 0),  # 14 slight right + gas + trick
    # ── brake ────────────────────────────────────────
    ( 0.0,  0, 1, 0, 0, 0, 0),  # 15 brake straight
    (-1.0,  0, 1, 0, 0, 0, 0),  # 16 brake left
    ( 1.0,  0, 1, 0, 0, 0, 0),  # 17 brake right
    # ── coast (no input) ─────────────────────────────
    ( 0.0,  0, 0, 0, 0, 0, 0),  # 18 coast
    (-1.0,  0, 0, 0, 0, 0, 0),  # 19 coast left
    ( 1.0,  0, 0, 0, 0, 0, 0),  # 20 coast right
]
# fmt: on

LOOKUP_TABLE = np.array(_TABLE, dtype=np.float32)
N_ACTIONS = len(LOOKUP_TABLE)


class DiscreteLookupAction(ActionParser):
    """
    Maps a single discrete index to a MKW controller input dict.
    Action space: Discrete(21).
    """

    def get_action_space(self) -> spaces.Space:
        return spaces.Discrete(N_ACTIONS)

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    def parse_action(self, action: np.ndarray | int) -> Dict[str, Any]:
        idx = int(action) % N_ACTIONS
        sx, a, b, r, x, up, down = LOOKUP_TABLE[idx].tolist()
        return {
            "StickX": sx,
            "A":    int(a),
            "B":    int(b),
            "R":    int(r),
            "X":    int(x),
            "Up":   int(up),
            "Down": int(down),
            "L":    0,
        }
