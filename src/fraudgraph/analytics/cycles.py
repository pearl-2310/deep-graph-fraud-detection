"""Week 3: circular movement of funds (layering / round-tripping).

Three complementary detectors on the account->account money-flow graph:

1. ``find_cycles``       - bounded simple cycles whose timestamps are *ordered* (money can really go around)
                           and whose value is *retained* (each hop passes on most of the money).
2. ``round_trip_flow``   - max-flow from a node's outgoing edges back to its incoming edges: how much
                           money can leave an account and return to it through the rest of the network?
3. ``suspicious_sccs``   - strongly connected components (every account reachable from every other).
"""
from __future__ import annotations

import networkx as nx


def _retention(amounts: list[float]) -> float:
    """Worst hop-to-hop size ratio along the cycle, from the best possible starting hop (so the result
    does not depend on which node the enumeration happened to start from). Capped at 1."""
    n = len(amounts)
    best = 0.0
    for s in range(n):
        worst = min(amounts[(s + i + 1) % n] / amounts[(s + i) % n] for i in range(n - 1))
        best = max(best, worst)
    return min(1.0, best)


def _time_feasible(first: list[int], last: list[int]) -> bool:
    """Could one unit of money travel hop 0 -> 1 -> ... in time order, for some choice of starting hop?
    Each hop is only known as a [first_ts, last_ts] window, so we greedily track the earliest feasible time."""
    n = len(first)
    for s in range(n):
        cur = first[s]
        ok = True
        for i in range(1, n):
            h = (s + i) % n
            if last[h] < cur:
                ok = False
                break
            cur = max(cur, first[h])
        if ok:
            return True
    return False


def find_cycles(d: nx.DiGraph, max_len: int = 8, min_retention: float = 0.6,
                max_scc_size: int = 500) -> list[dict]:
    """Bounded, time-feasible, value-retaining cycles.

    * Only SCCs with 3..max_scc_size nodes are searched (cycle enumeration is exponential; for a
      giant SCC run it on a community-restricted subgraph or use a streaming/GDS approach instead).
    * `retention` = worst ratio of consecutive hop sizes along the cycle, from the best start hop (capped at 1).
      Laundering loops pass on most of the money (retention close to 1); random reciprocal
      payments between friends do not.
    """
    out, seen = [], set()
    for comp in nx.strongly_connected_components(d):
        if not (3 <= len(comp) <= max_scc_size):
            continue
        sub = d.subgraph(comp)
        for cyc in nx.simple_cycles(sub, length_bound=max_len):
            if len(cyc) < 3:
                continue
            key = frozenset(cyc)
            if key in seen:
                continue
            hops = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
            amounts = [sub[a][b]["weight"] / sub[a][b]["count"] for a, b in hops]
            retention = _retention(amounts)
            if retention < min_retention:
                continue
            if not _time_feasible([sub[a][b]["first_ts"] for a, b in hops],
                                  [sub[a][b]["last_ts"] for a, b in hops]):
                continue
            seen.add(key)
            out.append({"nodes": cyc, "length": len(cyc), "total_amount": float(sum(amounts)),
                        "retention": float(retention)})
    return out
