"""Placeholder policy that samples random actions. Replace with your PPO model."""

from typing import Dict, List
import numpy as np
from .game_state import AgentID


class RandomPolicy:
    def __init__(self, n_actions: int, agents: List[AgentID]):
        self.n_actions = n_actions
        self.agents = agents

    def get_actions(self, flat_obs: Dict[AgentID, np.ndarray]) -> Dict[AgentID, int]:
        return {a: np.random.randint(0, self.n_actions) for a in flat_obs}
