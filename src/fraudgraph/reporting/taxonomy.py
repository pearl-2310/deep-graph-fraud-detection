
"""Fraud typologies and validated graph artifact loading."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path

import networkx as nx


TYPES = {
    "CIRCULAR_RING": (
        "Circular laundering ring",
        "A loop of accounts that pass funds around in time order with little value lost per hop "
        "(layering / round-tripping). Sparse internal transfers (~1 per account), high cycle coverage, "
        "round-trip max-flow close to 100% of outflow, often shared device/IP across owners.",
    ),
    "MULE_HUB": (
        "Hub-and-spoke mule network",
        "One collection account receives many small payments from outside accounts (victims), fans the money "
        "out to a set of mule accounts, which forward it to a collector. Signature: extreme in-degree at the hub, "
        "fan-out then fan-in, partial shared infrastructure.",
    ),
    "SYNTHETIC_CLUSTER": (
        "Synthetic-identity cluster",
        "Many 'users' sharing one device and IP, with dense mutual transfers that fabricate transaction history "
        "(bust-out preparation). Signature: high device/IP sharing, dense internal transfers (>2.5 per account).",
    ),
    "UNCLASSIFIED": (
        "Unclassified suspicious cluster",
        "Flagged by the model but matches no known signature - triage manually.",
    ),
}


def _load_graph(artifacts_dir: Path) -> tuple[nx.Graph, nx.Graph]:
    """Load and validate the undirected and transaction graphs.

    Only load graph artifacts from trusted sources because pickle files
    can execute arbitrary code during deserialization.
    """
    artifacts_dir = Path(artifacts_dir)

    if not artifacts_dir.is_dir():
        raise FileNotFoundError(
            f"Artifact directory does not exist: {artifacts_dir}"
        )

    graph_path = artifacts_dir / "graph.pkl"

    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Graph artifact does not exist: {graph_path}"
        )

    try:
        with graph_path.open("rb") as file:
            graph_data = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(
            f"Unable to load graph artifact: {graph_path}"
        ) from exc

    if not isinstance(graph_data, Mapping):
        raise ValueError("Graph artifact must contain a mapping.")

    if "undirected" not in graph_data or "transfers" not in graph_data:
        raise ValueError(
            "Graph artifact must contain 'undirected' and 'transfers'."
        )

    undirected = graph_data["undirected"]
    transfers = graph_data["transfers"]

    if not isinstance(undirected, nx.Graph) or undirected.is_directed():
        raise ValueError(
            "'undirected' must be an undirected NetworkX graph."
        )

    if not isinstance(transfers, nx.Graph):
        raise ValueError(
            "'transfers' must be a NetworkX graph."
        )

    return undirected, transfers
