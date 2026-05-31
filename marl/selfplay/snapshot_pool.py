"""
Phase 4 — Self-play opponent pool.

The proposal trains the shared policy "against past versions of the policy to
maintain opponent diversity and reduce cycling". This holds a bounded pool of
FROZEN snapshots of the shared actor. Each episode the learner team [0,1] uses
the live policy while the opponent team [2,3] is driven by a snapshot sampled
from the pool; only the learner team's trajectory is used for the update.

The win rate of the live policy against sampled snapshots is the proposal's
training-health signal: a rising win-rate-vs-older-snapshots indicates genuine
improvement rather than strategy cycling.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

from marl.algo.mappo import SharedActor


class SnapshotPool:
    def __init__(self, capacity: int = 10, strategy: str = "uniform"):
        assert strategy in ("uniform", "latest")
        self.capacity = capacity
        self.strategy = strategy
        self._pool: deque[SharedActor] = deque(maxlen=capacity)
        self.total_pushed = 0

    def __len__(self) -> int:
        return len(self._pool)

    def push(self, actor: SharedActor) -> None:
        """Add a frozen, detached copy of the current actor (drops oldest when full)."""
        self._pool.append(actor.clone_frozen())
        self.total_pushed += 1

    def sample(self) -> Optional[SharedActor]:
        """Return a frozen opponent, or None if the pool is empty."""
        if not self._pool:
            return None
        if self.strategy == "latest":
            return self._pool[-1]
        return random.choice(list(self._pool))
