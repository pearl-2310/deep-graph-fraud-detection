
"""Tests for fraud taxonomy definitions and graph artifact loading."""

import pickle

import networkx as nx
import pytest

from fraudgraph.reporting.taxonomy import TYPES, _load_graph


def create_graph_artifact(directory, graph_data):
    """Write a graph artifact for isolated loader tests."""
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "graph.pkl").open("wb") as file:
        pickle.dump(graph_data, file)


def test_taxonomy_defines_all_expected_types():
    expected_types = {
        "CIRCULAR_RING",
        "MULE_HUB",
        "SYNTHETIC_CLUSTER",
        "UNCLASSIFIED",
    }

    assert set(TYPES) == expected_types

    for name, description in TYPES.values():
        assert isinstance(name, str)
        assert name.strip()
        assert isinstance(description, str)
        assert description.strip()


def test_load_graph_returns_both_graphs(tmp_path):
    undirected = nx.Graph()
    undirected.add_edge("account_1", "account_2")

    transfers = nx.DiGraph()
    transfers.add_edge("account_1", "account_2", amount=100)

    artifacts_dir = tmp_path / "artifacts"
    create_graph_artifact(
        artifacts_dir,
        {
            "undirected": undirected,
            "transfers": transfers,
        },
    )

    loaded_undirected, loaded_transfers = _load_graph(artifacts_dir)

    assert isinstance(loaded_undirected, nx.Graph)
    assert not loaded_undirected.is_directed()
    assert isinstance(loaded_transfers, nx.DiGraph)
    assert loaded_undirected.number_of_edges() == 1
    assert loaded_transfers.number_of_edges() == 1


def test_load_graph_rejects_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Artifact directory"):
        _load_graph(missing_dir)


def test_load_graph_rejects_missing_file(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Graph artifact"):
        _load_graph(artifacts_dir)


def test_load_graph_rejects_invalid_pickle(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    (artifacts_dir / "graph.pkl").write_bytes(b"not a valid pickle")

    with pytest.raises(ValueError, match="Unable to load graph artifact"):
        _load_graph(artifacts_dir)


def test_load_graph_rejects_non_mapping_artifact(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    create_graph_artifact(artifacts_dir, ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="must contain a mapping"):
        _load_graph(artifacts_dir)


def test_load_graph_rejects_missing_graph_keys(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    create_graph_artifact(
        artifacts_dir,
        {"undirected": nx.Graph()},
    )

    with pytest.raises(ValueError, match="undirected.*transfers"):
        _load_graph(artifacts_dir)


def test_load_graph_rejects_directed_undirected_graph(tmp_path):
    artifacts_dir = tmp_path / "artifacts"

    create_graph_artifact(
        artifacts_dir,
        {
            "undirected": nx.DiGraph(),
            "transfers": nx.DiGraph(),
        },
    )

    with pytest.raises(ValueError, match="undirected NetworkX graph"):
        _load_graph(artifacts_dir)


def test_load_graph_rejects_invalid_transfers_graph(tmp_path):
    artifacts_dir = tmp_path / "artifacts"

    create_graph_artifact(
        artifacts_dir,
        {
            "undirected": nx.Graph(),
            "transfers": "not a graph",
        },
    )

    with pytest.raises(ValueError, match="transfers.*NetworkX graph"):
        _load_graph(artifacts_dir)
