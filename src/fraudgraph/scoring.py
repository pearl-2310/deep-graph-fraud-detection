
"""Reliable neighbourhood-based fraud risk scoring."""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from fraudgraph.config import settings


REQUIRED_ARTIFACTS = (
    "model.joblib",
    "graph.pkl",
    "features.csv",
    "scores.csv",
    "embeddings.csv",
    "communities.csv",
    "community_table.csv",
)

SIGNAL_COLUMNS = (
    "n_cycles",
    "round_trip_ratio",
    "max_device_share",
    "max_ip_share",
    "degree",
)


def risk_level(p: float) -> str:
    """Convert a probability into a business-readable risk category."""
    try:
        probability = float(p)
    except (TypeError, ValueError) as exc:
        raise ValueError("Risk probability must be numeric.") from exc

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Risk probability must be finite and between 0 and 1.")

    if probability >= 0.8:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    if probability >= 0.2:
        return "LOW"
    return "MINIMAL"


class Scorer:
    """Score known entities and estimate risk for new entities."""

    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = Path(
            artifacts_dir if artifacts_dir is not None else settings.artifacts_dir
        )

        self._validate_artifact_files()

        try:
            bundle = joblib.load(self.artifacts_dir / "model.joblib")

            with (self.artifacts_dir / "graph.pkl").open("rb") as file:
                graph_data = pickle.load(file)

            self.features = pd.read_csv(
                self.artifacts_dir / "features.csv",
                index_col="id",
            )
            score_data = pd.read_csv(
                self.artifacts_dir / "scores.csv",
                index_col="id",
            )
            self.emb = pd.read_csv(
                self.artifacts_dir / "embeddings.csv",
                index_col="id",
            )
            community_data = pd.read_csv(
                self.artifacts_dir / "communities.csv",
                index_col="id",
            )
            community_table = pd.read_csv(
                self.artifacts_dir / "community_table.csv"
            )

        except Exception as exc:
            raise RuntimeError(
                f"Unable to load scoring artifacts from "
                f"{self.artifacts_dir}: {exc}"
            ) from exc

        if not isinstance(bundle, Mapping):
            raise ValueError("Model artifact must contain a mapping.")

        if not isinstance(graph_data, Mapping):
            raise ValueError("Graph artifact must contain a mapping.")

        self.model = bundle.get("model")
        self.columns = bundle.get("columns")

        self.u = graph_data.get("undirected")
        self.d = graph_data.get("transfers")

        self._validate_loaded_data(
            bundle,
            graph_data,
            score_data,
            community_data,
            community_table,
        )

        self.scores = score_data["risk"].astype(float)
        self.part = community_data["community"]
        self.comm = community_table.set_index("community")

    def _validate_artifact_files(self) -> None:
        """Check that all required artifacts exist before loading."""
        if not self.artifacts_dir.is_dir():
            raise FileNotFoundError(
                f"Artifact directory does not exist: {self.artifacts_dir}"
            )

        missing = [
            name
            for name in REQUIRED_ARTIFACTS
            if not (self.artifacts_dir / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing scoring artifacts in {self.artifacts_dir}: "
                + ", ".join(missing)
            )

    def _validate_loaded_data(
        self,
        bundle: object,
        graph_data: object,
        score_data: pd.DataFrame,
        community_data: pd.DataFrame,
        community_table: pd.DataFrame,
    ) -> None:
        """Validate artifact structure and basic data consistency."""
        if "model" not in bundle or "columns" not in bundle:
            raise ValueError(
                "Model artifact must contain 'model' and 'columns'."
            )

        if not callable(getattr(self.model, "predict_proba", None)):
            raise ValueError(
                "Model must provide a callable predict_proba method."
            )

        if (
            not isinstance(self.columns, (list, tuple))
            or not self.columns
            or not all(
                isinstance(column, str) and column
                for column in self.columns
            )
            or len(set(self.columns)) != len(self.columns)
        ):
            raise ValueError(
                "Model feature columns must be a non-empty unique list."
            )

        if "undirected" not in graph_data or "transfers" not in graph_data:
            raise ValueError(
                "Graph artifact must contain 'undirected' and 'transfers'."
            )

        if not isinstance(self.u, nx.Graph) or self.u.is_directed():
            raise ValueError(
                "'undirected' must be an undirected NetworkX graph."
            )

        if not isinstance(self.d, nx.Graph):
            raise ValueError(
                "'transfers' must be a NetworkX graph."
            )

        if not isinstance(score_data, pd.DataFrame) or "risk" not in score_data:
            raise ValueError("Score artifact must contain a 'risk' column.")

        if (
            not isinstance(community_data, pd.DataFrame)
            or "community" not in community_data
        ):
            raise ValueError(
                "Community artifact must contain a 'community' column."
            )

        if not isinstance(community_table, pd.DataFrame):
            raise ValueError("Community table must be a DataFrame.")

        required_community_columns = {
            "community",
            "size",
            "density",
            "n_accounts",
        }
        missing_community_columns = (
            required_community_columns - set(community_table.columns)
        )

        if missing_community_columns:
            raise ValueError(
                "Community table is missing columns: "
                + ", ".join(sorted(missing_community_columns))
            )

        if self.features.empty or self.emb.empty:
            raise ValueError(
                "Feature and embedding artifacts must not be empty."
            )

        if (
            not self.features.index.is_unique
            or not self.emb.index.is_unique
            or not score_data.index.is_unique
            or not community_data.index.is_unique
        ):
            raise ValueError(
                "Entity IDs must be unique in all scoring artifacts."
            )

        if not set(self.features.index).issubset(self.u.nodes):
            raise ValueError(
                "Some feature entities are missing from the graph."
            )

        if not set(self.features.index).issubset(score_data.index):
            raise ValueError(
                "Some feature entities are missing risk scores."
            )

        if not set(self.features.index).issubset(community_data.index):
            raise ValueError(
                "Some feature entities are missing community assignments."
            )

        if not set(self.emb.index).issubset(self.u.nodes):
            raise ValueError(
                "Some embedding entities are missing from the graph."
            )

        if not all(
            column in self.features.columns
            for column in SIGNAL_COLUMNS
        ):
            raise ValueError(
                "Feature artifact is missing required fraud signal columns."
            )

        numeric_scores = pd.to_numeric(
            score_data["risk"],
            errors="coerce",
        )

        if (
            numeric_scores.isna().any()
            or not np.isfinite(numeric_scores.to_numpy()).all()
            or not numeric_scores.between(0, 1).all()
        ):
            raise ValueError(
                "Risk scores must be finite probabilities between 0 and 1."
            )

        if not community_table["community"].is_unique:
            raise ValueError(
                "Community table must contain unique community IDs."
            )

        if not set(community_data["community"]).issubset(
            set(community_table["community"])
        ):
            raise ValueError(
                "Some entity community assignments are missing from "
                "the community table."
            )

    def signals(self, row: pd.Series) -> list[str]:
        """Generate human-readable indicators for a known entity."""
        output = []

        if row["n_cycles"] > 0:
            output.append(
                f"participates in {int(row['n_cycles'])} circular fund-flow cycle(s)"
            )

        if row["round_trip_ratio"] >= 0.5:
            output.append(
                f"{row['round_trip_ratio']:.0%} of outgoing funds can "
                "return to the account"
            )

        if row["max_device_share"] >= 3:
            output.append(
                f"device shared with {int(row['max_device_share'])} other users"
            )

        if row["max_ip_share"] >= 3:
            output.append(
                f"IP address shared with {int(row['max_ip_share'])} other users"
            )

        if row["degree"] >= 15:
            output.append(
                f"unusually high connectivity (degree {int(row['degree'])})"
            )

        return output

    def score_entity(
        self,
        entity_id: str,
        max_neighbors: int = 25,
    ) -> dict:
        """Return risk score and supporting signals for a known entity."""
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("Entity ID must be a non-empty string.")

        if isinstance(max_neighbors, bool) or not isinstance(max_neighbors, int):
            raise TypeError("max_neighbors must be a non-negative integer.")

        if max_neighbors < 0:
            raise ValueError("max_neighbors cannot be negative.")

        if entity_id not in self.features.index:
            raise KeyError(f"Unknown entity: {entity_id}")

        if entity_id not in self.u:
            raise ValueError(
                f"Entity {entity_id!r} is missing from the graph."
            )

        if (
            entity_id not in self.scores.index
            or entity_id not in self.part.index
        ):
            raise ValueError(
                f"Entity {entity_id!r} has incomplete scoring data."
            )

        probability = float(self.scores[entity_id])
        level = risk_level(probability)

        neighbors = list(self.u[entity_id])

        neighbor_results = [
            {
                "id": neighbor,
                "kind": self.u.nodes[neighbor].get("kind", "unknown"),
                "risk": (
                    float(self.scores[neighbor])
                    if neighbor in self.scores.index
                    else None
                ),
            }
            for neighbor in neighbors
        ]

        neighbor_results.sort(
            key=lambda item: (
                item["risk"] is not None,
                item["risk"] if item["risk"] is not None else -1,
            ),
            reverse=True,
        )

        return {
            "id": entity_id,
            "kind": self.u.nodes[entity_id].get("kind", "unknown"),
            "risk_score": round(probability, 4),
            "risk_level": level,
            "community": int(self.part[entity_id]),
            "signals": self.signals(self.features.loc[entity_id]),
            "n_neighbors": len(neighbors),
            "high_risk_neighbors": sum(
                1
                for item in neighbor_results
                if item["risk"] is not None and item["risk"] >= 0.5
            ),
            "neighbors": neighbor_results[:max_neighbors],
        }

    def score_new_entity(
        self,
        kind: str,
        neighbors: list[str],
    ) -> dict:
        """Estimate risk for a new entity from its known one-hop neighbours.

        The embedding is approximated by the mean of neighbour embeddings.
        This is not full graph-model inference for a newly inserted node.
        """
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Entity kind must be a non-empty string.")

        if not isinstance(neighbors, list) or not neighbors:
            raise ValueError(
                "neighbors must be a non-empty list of entity IDs."
            )

        if any(
            not isinstance(neighbor, str) or not neighbor.strip()
            for neighbor in neighbors
        ):
            raise ValueError(
                "Every neighbour ID must be a non-empty string."
            )

        known = list(dict.fromkeys(neighbors))

        unknown = [
            neighbor
            for neighbor in known
            if neighbor not in self.emb.index
        ]

        if unknown:
            raise ValueError(
                "Unknown neighbour IDs: " + ", ".join(unknown)
            )

        for neighbor in known:
            if neighbor not in self.u:
                raise ValueError(
                    f"Neighbour {neighbor!r} is missing from the graph."
                )

        embeddings = self.emb.loc[known]

        if embeddings.isna().any().any():
            raise ValueError(
                "Neighbour embeddings contain missing values."
            )

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in embeddings.dtypes
        ):
            raise ValueError(
                "Neighbour embeddings must be numeric."
            )

        if not np.isfinite(embeddings.to_numpy(dtype=float)).all():
            raise ValueError(
                "Neighbour embeddings contain non-finite values."
            )

        embedding_mean = embeddings.mean().to_numpy()
        neighbor_count = len(known)

        links = sum(
            1
            for index, first in enumerate(known)
            for second in known[index + 1:]
            if self.u.has_edge(first, second)
        )

        kinds = [
            self.u.nodes[neighbor].get("kind")
            for neighbor in known
        ]

        device_share = max(
            [
                self.u.degree(neighbor)
                for neighbor, node_kind in zip(known, kinds)
                if node_kind == "device"
            ] or [0]
        )

        ip_share = max(
            [
                self.u.degree(neighbor)
                for neighbor, node_kind in zip(known, kinds)
                if node_kind == "ip"
            ] or [0]
        )

        for neighbor, node_kind in zip(known, kinds):
            if node_kind == "user" and neighbor in self.features.index:
                device_share = max(
                    device_share,
                    self.features.loc[neighbor, "max_device_share"],
                )
                ip_share = max(
                    ip_share,
                    self.features.loc[neighbor, "max_ip_share"],
                )

        community_ids = [
            self.part[neighbor]
            for neighbor in known
            if neighbor in self.part.index
        ]

        community_mode = (
            pd.Series(community_ids).mode()
            if community_ids
            else pd.Series(dtype=int)
        )

        community = (
            self.comm.loc[community_mode.iloc[0]]
            if not community_mode.empty
            else None
        )

        row = {
            "degree": neighbor_count,
            "weighted_degree": neighbor_count,
            "betweenness": 0.0,
            "clustering": (
                links / (neighbor_count * (neighbor_count - 1) / 2)
                if neighbor_count > 1
                else 0.0
            ),
            "is_account": int(kind == "account"),
            "comm_size": community["size"] if community is not None else 1,
            "comm_density": (
                community["density"] if community is not None else 0.0
            ),
            "comm_n_accounts": (
                community["n_accounts"] if community is not None else 0
            ),
            "max_device_share": device_share,
            "max_ip_share": ip_share,
            "n_cycles": 0,
            "round_trip_ratio": 0.0,
        }

        row.update({
            f"emb_{index}": value
            for index, value in enumerate(embedding_mean)
        })

        missing_columns = [
            column
            for column in self.columns
            if column not in row
        ]

        if missing_columns:
            raise ValueError(
                "Unable to construct model features. Missing columns: "
                + ", ".join(missing_columns)
            )

        features = pd.DataFrame(
            [row],
            columns=self.columns,
        )

        if features.isna().any().any():
            raise ValueError(
                "Constructed model features contain missing values."
            )

        try:
            feature_values = features.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Constructed model features must be numeric."
            ) from exc

        if not np.isfinite(feature_values).all():
            raise ValueError(
                "Constructed model features contain non-finite values."
            )

        try:
            probabilities = np.asarray(
                self.model.predict_proba(features),
                dtype=float,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model prediction failed: {exc}"
            ) from exc

        if probabilities.shape != (1, 2):
            raise ValueError(
                "Model must return exactly two class probabilities "
                "for one entity."
            )

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                "Model returned invalid class probabilities."
            )

        classes = getattr(self.model, "classes_", None)

        if classes is not None:
            class_list = list(classes)

            if 1 not in class_list:
                raise ValueError(
                    "Model does not contain the positive fraud class."
                )

            positive_index = class_list.index(1)
        else:
            positive_index = 1

        probability = float(probabilities[0, positive_index])
        level = risk_level(probability)

        high_risk = [
            neighbor
            for neighbor in known
            if (
                neighbor in self.scores.index
                and self.scores[neighbor] >= 0.5
            )
        ]

        return {
            "risk_score": round(probability, 4),
            "risk_level": level,
            "n_known_neighbors": neighbor_count,
            "high_risk_neighbors": high_risk,
            "approximate": True,
        }

    def cluster_members(self, community: int) -> list[str]:
        """Return entity IDs assigned to a community."""
        if isinstance(community, bool) or not isinstance(community, int):
            raise TypeError("Community ID must be an integer.")

        return list(
            self.part[self.part == community].index
        )
