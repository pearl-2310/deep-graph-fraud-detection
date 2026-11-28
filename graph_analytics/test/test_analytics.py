
"""Tests for graph metrics, community detection, embeddings and PageRank variants."""

from fraudgraph.analytics import metrics


def test_metrics_and_paths(graphs):
    _, u, _ = graphs

    m = metrics.node_metrics(u, betweenness_k=50)

    assert {"degree", "betweenness", "clustering"} <= set(m.columns)
    assert len(m) == u.number_of_nodes()

    sp = metrics.shortest_path_summary(u, n_pairs=50)

    assert sp["mean_path_length"] > 1
