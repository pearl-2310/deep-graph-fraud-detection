"""Week 3: circular movement of funds (layering / round-tripping).

Three complementary detectors on the account->account money-flow graph:

1. ``find_cycles``       - bounded simple cycles whose timestamps are *ordered* (money can really go around)
                           and whose value is *retained* (each hop passes on most of the money).
2. ``round_trip_flow``   - max-flow from a node's outgoing edges back to its incoming edges: how much
                           money can leave an account and return to it through the rest of the network?
3. ``suspicious_sccs``   - strongly connected components (every account reachable from every other).
"""
from __future__ import annotations


def _retention(amounts: list[float]) -> float:
    """Worst hop-to-hop size ratio along the cycle, from the best possible starting hop (so the result
    does not depend on which node the enumeration happened to start from). Capped at 1."""
    n = len(amounts)
    best = 0.0
    for s in range(n):
        worst = min(amounts[(s + i + 1) % n] / amounts[(s + i) % n] for i in range(n - 1))
        best = max(best, worst)
    return min(1.0, best)
