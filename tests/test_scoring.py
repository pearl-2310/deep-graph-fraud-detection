
"""Unit tests for fraud risk scoring."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from fraudgraph.scoring import Scorer, risk_level


class MockModel:
    """Controlled model used for isolated scorer tests."""

    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        return np.array([[0.2, 0.8]])


@pytest.fixture
def scorer():
    """Create an in-memory scorer with controlled synthetic data."""
    instance = Scorer.__new__(Scorer)

    graph = nx.Graph()

    graph.add_node("user_1", kind="user")
    graph.add_node("user_2", kind="user")
    graph.add_node("device_1", kind="device")
    graph.add_node("ip_1", kind="ip")

    graph.add_edges_from([
        ("user_1", "device_1"),
        ("user_1", "ip_1"),
        ("user_2", "device_1"),
        ("user_2", "ip_1"),
    ])

    instance.u = graph
    instance.d = nx.DiGraph()

    instance.model = MockModel()

    instance.columns = [
        "degree",
        "weighted_degree",
        "betweenness",
        "clustering",
        "is_account",
        "comm_size",
        "comm_density",
        "comm_n_accounts",
        "max_device_share",
        "max_ip_share",
        "n_cycles",
        "round_trip_ratio",
        "emb_0",
        "emb_1",
    ]

    instance.features = pd.DataFrame(
        [
            {
                "n_cycles": 2,
                "round_trip_ratio": 0.6,
                "max_device_share": 3,
                "max_ip_share": 2,
                "degree": 15,
            },
            {
                "n_cycles": 0,
                "round_trip_ratio": 0.1,
                "max_device_share": 1,
                "max_ip_share": 1,
                "degree": 2,
            },
        ],
        index=["user_1", "user_2"],
    )

    instance.scores = pd.Series(
        {
            "user_1": 0.9,
            "user_2": 0.3,
        }
    )

    instance.emb = pd.DataFrame(
        {
            "emb_0": [0.1, 0.3, 0.5, 0.7],
            "emb_1": [0.2, 0.4, 0.6, 0.8],
        },
        index=["user_1", "user_2", "device_1", "ip_1"],
    )

    instance.part = pd.Series(
        {
            "user_1": 1,
            "user_2": 1,
        }
    )

    instance.comm = pd.DataFrame(
        {
            "size": [10],
            "density": [0.4],
            "n_accounts": [5],
        },
        index=[1],
    )

    return instance


# ---------------------------------------------------------
# Risk classification tests
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.9, "HIGH"),
        (0.8, "HIGH"),
        (0.6, "MEDIUM"),
        (0.5, "MEDIUM"),
        (0.3, "LOW"),
        (0.2, "LOW"),
        (0.05, "MINIMAL"),
        (0.0, "MINIMAL"),
    ],
)
def test_risk_levels(probability, expected):
    assert risk_level(probability) == expected


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, float("nan"), float("inf"), float("-inf")],
)
def test_risk_level_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        risk_level(probability)


# ---------------------------------------------------------
# Known entity scoring tests
# ---------------------------------------------------------


def test_score_known_entity(scorer):
    result = scorer.score_entity("user_1")

    assert result["id"] == "user_1"
    assert result["kind"] == "user"
    assert result["risk_score"] == 0.9
    assert result["risk_level"] == "HIGH"
    assert result["community"] == 1
    assert result["n_neighbors"] == 2
    assert result["high_risk_neighbors"] == 0


def test_known_entity_returns_fraud_signals(scorer):
    result = scorer.score_entity("user_1")

    assert len(result["signals"]) == 4
    assert any("circular fund-flow cycle" in signal for signal in result["signals"])
    assert any("return to the account" in signal for signal in result["signals"])


def test_known_entity_limits_neighbors(scorer):
    result = scorer.score_entity("user_1", max_neighbors=1)

    assert len(result["neighbors"]) == 1
    assert result["n_neighbors"] == 2


def test_known_entity_rejects_unknown_id(scorer):
    with pytest.raises(KeyError):
        scorer.score_entity("unknown_user")


def test_known_entity_rejects_negative_neighbor_limit(scorer):
    with pytest.raises(ValueError):
        scorer.score_entity("user_1", max_neighbors=-1)


def test_known_entity_rejects_invalid_neighbor_limit(scorer):
    with pytest.raises(TypeError):
        scorer.score_entity("user_1", max_neighbors=1.5)


# ---------------------------------------------------------
# New entity scoring tests
# ---------------------------------------------------------


def test_score_new_entity(scorer):
    result = scorer.score_new_entity(
        kind="account",
        neighbors=["user_1", "device_1"],
    )

    assert result["risk_score"] == 0.8
    assert result["risk_level"] == "HIGH"
    assert result["n_known_neighbors"] == 2
    assert result["high_risk_neighbors"] == ["user_1"]
    assert result["approximate"] is True


def test_new_entity_deduplicates_neighbors(scorer):
    result = scorer.score_new_entity(
        kind="account",
        neighbors=["user_1", "user_1"],
    )

    assert result["n_known_neighbors"] == 1


def test_new_entity_rejects_empty_neighbors(scorer):
    with pytest.raises(ValueError):
        scorer.score_new_entity("account", [])


def test_new_entity_rejects_unknown_neighbors(scorer):
    with pytest.raises(ValueError, match="Unknown neighbour IDs"):
        scorer.score_new_entity(
            "account",
            ["unknown_user"],
        )


def test_new_entity_rejects_invalid_kind(scorer):
    with pytest.raises(ValueError):
        scorer.score_new_entity("", ["user_1"])


def test_new_entity_rejects_invalid_neighbor_type(scorer):
    with pytest.raises(ValueError):
        scorer.score_new_entity("account", ["user_1", 123])


# ---------------------------------------------------------
# Artifact validation tests
# ---------------------------------------------------------


def test_missing_artifacts_produce_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing scoring artifacts"):
        Scorer(artifacts_dir=tmp_path)
