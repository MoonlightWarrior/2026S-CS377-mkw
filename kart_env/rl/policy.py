from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np

from .game_state import AgentID


class Policy(ABC):
    """
    Interface that every trainable policy must satisfy.

    get_actions  — called once per env step; receives a flat obs array per agent
    save / load  — persist and restore weights; no-ops by default so random
                   baselines don't need to implement them
    """

    @abstractmethod
    def get_actions(self, flat_obs: Dict[AgentID, np.ndarray]) -> Dict[AgentID, int]:
        """Return a discrete action index for each agent."""
        ...

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass


class RandomPolicy(Policy):
    """Uniform-random baseline — useful for smoke-testing the env pipeline."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def get_actions(self, flat_obs: Dict[AgentID, np.ndarray]) -> Dict[AgentID, int]:
        return {a: int(np.random.randint(0, self.n_actions)) for a in flat_obs}
