
"""Tests for scoring artifact loading and validation."""

import pickle

import joblib
import networkx as nx
import pandas as pd
import pytest

from fraudgraph.scoring import Scorer


class ValidArtifactModel:
    """Minimal model for testing artifact loading."""

    classes_ = [0, 1]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


class InvalidArtifactModel:
    """Model without the required prediction interface."""

    pass


def create_artifacts(directory, model=None):
    """Create a complete, isolated set of synthetic scoring artifacts."""

    directory.mkdir(parents=True, exist_ok=True)

    graph = nx.Graph()
    graph.add_node("account_1", kind="account")
    graph.add_node("account_2", kind="account")
    graph.add_edge("account_1", "account_2", weight=1)

    bundle = {
        "model": model or ValidArtifactModel(),
        "columns": ["degree"],
    }

    with (directory / "model.joblib").open("wb") as file:
        joblib.dump(bundle, file)

    graph_data = {
        "undirected": graph,
        "transfers": nx.DiGraph(),
    }

    with (directory / "graph.pkl").open("wb") as file:
        pickle.dump(graph_data, file)

    features = pd.DataFrame(
        {
            "n_cycles": [0, 0],
            "round_trip_ratio": [0.0, 0.0],
            "max_device_share": [0, 0],
            "max_ip_share": [0, 0],
            "degree": [1, 1],
        },
        index=["account_1", "account_2"],
    )
    features.index.name = "id"
    features.to_csv(directory / "features.csv")

    scores = pd.DataFrame(
        {"risk": [0.8, 0.2]},
        index=["account_1", "account_2"],
    )
    scores.index.name = "id"
    scores.to_csv(directory / "scores.csv")

    embeddings = pd.DataFrame(
        {
            "emb_0": [0.1, 0.2],
            "emb_1": [0.3, 0.4],
        },
        index=["account_1", "account_2"],
    )
    embeddings.index.name = "id"
    embeddings.to_csv(directory / "embeddings.csv")

    communities = pd.DataFrame(
        {"community": [1, 1]},
        index=["account_1", "account_2"],
    )
    communities.index.name = "id"
    communities.to_csv(directory / "communities.csv")

    community_table = pd.DataFrame(
        {
            "community": [1],
            "size": [2],
            "density": [1.0],
            "n_accounts": [2],
        }
    )
    community_table.to_csv(
        directory / "community_table.csv",
        index=False,
    )


def test_scorer_loads_valid_artifacts(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    create_artifacts(artifacts_dir)

    scorer = Scorer(artifacts_dir=artifacts_dir)

    assert scorer.model is not None
    assert scorer.u.number_of_nodes() == 2
    assert scorer.d.is_directed()
    assert len(scorer.features) == 2
    assert len(scorer.emb) == 2

    result = scorer.score_entity("account_1")

    assert result["risk_score"] == 0.8
    assert result["risk_level"] == "HIGH"


def test_scorer_rejects_model_without_predict_proba(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    create_artifacts(
        artifacts_dir,
        model=InvalidArtifactModel(),
    )

    with pytest.raises(
        ValueError,
        match="predict_proba",
    ):
        Scorer(artifacts_dir=artifacts_dir)
