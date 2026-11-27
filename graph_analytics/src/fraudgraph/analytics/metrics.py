
"""Week 2: exploratory graph analytics (degree, betweenness, shortest paths)."""

from __future__ import annotations

import random
import networkx as nx
import pandas as pd

from fraudgraph.config import settings


def node_metrics(u: nx.Graph, betweenness_k: int = 300) -> pd.DataFrame:
    """Calculate per-node structural graph metrics."""

    k = min(betweenness_k, u.number_of_nodes())

    bt = nx.betweenness_centrality(
        u, k=k, seed=settings.seed
    )

    deg = dict(u.degree())
    wdeg = dict(u.degree(weight="weight"))
    clust = nx.clustering(u)

    df = pd.DataFrame({
        "degree": pd.Series(deg),
        "weighted_degree": pd.Series(wdeg),
        "betweenness": pd.Series(bt),
        "clustering": pd.Series(clust),
    })

    df["kind"] = pd.Series({
        n: d["kind"] for n, d in u.nodes(data=True)
    })

    return df


def shortest_path_summary(u: nx.Graph, n_pairs: int = 500) -> dict:
    """Calculate sampled shortest-path statistics."""

    rng = random.Random(settings.seed)

    lcc = u.subgraph(
        max(nx.connected_components(u), key=len)
    )

    nodes = list(lcc.nodes)
    lengths = []

    for _ in range(n_pairs):
        a, b = rng.sample(nodes, 2)
        lengths.append(
            nx.shortest_path_length(lcc, a, b)
        )

    s = pd.Series(lengths)

    return {
        "lcc_nodes": lcc.number_of_nodes(),
        "lcc_fraction": lcc.number_of_nodes() / u.number_of_nodes(),
        "mean_path_length": float(s.mean()),
        "median_path_length": float(s.median()),
        "max_sampled_path_length": int(s.max()),
        "n_connected_components": nx.number_connected_components(u),
        "density": nx.density(u),
    }
