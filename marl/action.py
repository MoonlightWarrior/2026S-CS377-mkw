"""
MKW action parser with the CORRECT item button.

Phase 0 finding (verified via marl/diagnostics/check_items.py): on a GameCube
controller in Mario Kart Wii the item button is **L**, not X (X is look-behind).
The Vlab `DiscreteLookupAction` presses "X" for item, so karts pick items up but
can never use them. This parser is identical in structure but maps item-use to L.

Verified: with L, held items discharge immediately (confirmed_fires > 0); with X
they never fire.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

from kart_env.rl.action_parser import ActionParser
from kart_env.rl.game_state import AgentID, KartGameState

# fmt: off
# Each row: (StickX, A, B, R, Item, Up, Down)  — "Item" now drives the L button.
_TABLE = [
    ( 0.0,  1, 0, 0, 0, 0, 0),  # 0  straight + gas
    ( 0.0,  1, 0, 0, 0, 1, 0),  # 1  straight + gas + trick/wheelie
    ( 0.0,  1, 0, 0, 1, 0, 0),  # 2  straight + gas + ITEM
    (-0.5,  1, 0, 0, 0, 0, 0),  # 3  slight left + gas
    (-1.0,  1, 0, 0, 0, 0, 0),  # 4  hard left + gas
    (-0.5,  1, 0, 1, 0, 0, 0),  # 5  slight left + gas + drift
    (-1.0,  1, 0, 1, 0, 0, 0),  # 6  hard left + gas + drift
    (-1.0,  1, 0, 0, 1, 0, 0),  # 7  hard left + gas + ITEM
    ( 0.5,  1, 0, 0, 0, 0, 0),  # 8  slight right + gas
    ( 1.0,  1, 0, 0, 0, 0, 0),  # 9  hard right + gas
    ( 0.5,  1, 0, 1, 0, 0, 0),  # 10 slight right + gas + drift
    ( 1.0,  1, 0, 1, 0, 0, 0),  # 11 hard right + gas + drift
    ( 1.0,  1, 0, 0, 1, 0, 0),  # 12 hard right + gas + ITEM
    (-0.5,  1, 0, 0, 0, 1, 0),  # 13 slight left + gas + trick
    ( 0.5,  1, 0, 0, 0, 1, 0),  # 14 slight right + gas + trick
    ( 0.0,  0, 1, 0, 0, 0, 0),  # 15 brake straight
    (-1.0,  0, 1, 0, 0, 0, 0),  # 16 brake left
    ( 1.0,  0, 1, 0, 0, 0, 0),  # 17 brake right
    ( 0.0,  0, 0, 0, 0, 0, 0),  # 18 coast
    (-1.0,  0, 0, 0, 0, 0, 0),  # 19 coast left
    ( 1.0,  0, 0, 0, 0, 0, 0),  # 20 coast right
]
# fmt: on

LOOKUP_TABLE = np.array(_TABLE, dtype=np.float32)
N_ACTIONS = len(LOOKUP_TABLE)

# Action indices that fire an item (used by analysis / reward instrumentation).
ITEM_USE_ACTIONS = frozenset({2, 7, 12})


class MKWTeamAction(ActionParser):
    """Discrete(21) MKW controller mapping with item-use on the L button."""

    def get_action_space(self) -> spaces.Space:
        return spaces.Discrete(N_ACTIONS)

    def reset(self, agents: List[AgentID], initial_state: KartGameState) -> None:
        pass

    def parse_action(self, action: np.ndarray | int) -> Dict[str, Any]:
        idx = int(action) % N_ACTIONS
        sx, a, b, r, item, up, down = LOOKUP_TABLE[idx].tolist()
        return {
            "StickX": sx,
            "A":    int(a),
            "B":    int(b),
            "R":    int(r),
            "L":    int(item),   # ← item button (verified correct for MKW GCN)
            "X":    0,
            "Up":   int(up),
            "Down": int(down),
        }
