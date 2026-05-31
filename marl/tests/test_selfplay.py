"""
Offline test for the Phase 4 self-play snapshot pool.

Run:  PYTHONPATH=/cs377:$PYTHONPATH python /cs377/marl/tests/test_selfplay.py
"""
from __future__ import annotations

import torch

from marl.algo.mappo import SharedActor
from marl.selfplay import SnapshotPool

OBS, N_ACT = 59, 21


def main() -> None:
    pool = SnapshotPool(capacity=3, strategy="uniform")
    assert len(pool) == 0 and pool.sample() is None, "empty pool should sample None"

    actor = SharedActor(OBS, N_ACT, hidden=[32, 32])

    # push 5 into a capacity-3 pool -> keeps newest 3, total_pushed tracks all
    for _ in range(5):
        pool.push(actor)
    assert len(pool) == 3, len(pool)
    assert pool.total_pushed == 5, pool.total_pushed

    # sampled opponent is frozen (no grad) and detached from the live actor
    snap = pool.sample()
    assert snap is not None
    assert all(not p.requires_grad for p in snap.net.parameters()), "snapshot must be frozen"

    # independence: mutate the live actor; the snapshot must not change
    x = torch.randn(4, OBS)
    before = snap.net(x).detach().clone()
    with torch.no_grad():
        for p in actor.net.parameters():
            p.add_(1.0)   # large perturbation to the live policy
    after = snap.net(x).detach().clone()
    assert torch.allclose(before, after), "snapshot changed when live actor was mutated"

    # the frozen snapshot can still act
    acts = snap.get_actions({0: x[0].numpy(), 1: x[1].numpy()})
    assert set(acts) == {0, 1} and all(0 <= v < N_ACT for v in acts.values())

    print(f"PASS  pool caps at 3 (pushed 5), snapshots frozen+independent, "
          f"frozen opponent acts OK")


if __name__ == "__main__":
    main()
