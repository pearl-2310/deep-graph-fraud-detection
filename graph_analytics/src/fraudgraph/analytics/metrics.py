
from __future__ import annotations

import networkx as nx
import pandas as pd
from fraudgraph.config import settings


def node_metrics(u: nx.Graph, betweenness_k: int = 300) -> pd.DataFrame:
    """Per-node structural metrics on the undirected projection.

    Betweenness is *sampled* (k pivots) - exact betweenness is O(VE) and does not scale.
    For production use Neo4j GDS `gds.betweenness.stream` with samplingSize.
    """
    k = min(betweenness_k, u.number_of_nodes())
    bt = nx.betweenness_centrality(u, k=k, seed=settings.seed)
    deg = dict(u.degree())
    wdeg = dict(u.degree(weight="weight"))
    clust = nx.clustering(u)
    df = pd.DataFrame({
        "degree": pd.Series(deg),
        "weighted_degree": pd.Series(wdeg),
        "betweenness": pd.Series(bt),
        "clustering": pd.Series(clust),
    })
    df["kind"] = pd.Series({n: d["kind"] for n, d in u.nodes(data=True)})
    return df
