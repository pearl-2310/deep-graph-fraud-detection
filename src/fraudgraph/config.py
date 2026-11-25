"""Central configuration. Override with environment variables or a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("FG_DATA_DIR", ROOT / "data"))
    artifacts_dir: Path = Path(os.getenv("FG_ARTIFACTS_DIR", ROOT / "artifacts"))
    reports_dir: Path = Path(os.getenv("FG_REPORTS_DIR", ROOT / "reports"))
    seed: int = int(os.getenv("FG_SEED", "42"))

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "frauddetect123")

    embedding_dim: int = int(os.getenv("FG_EMBED_DIM", "32"))
    fraud_threshold: float = float(os.getenv("FG_FRAUD_THRESHOLD", "0.5"))


settings = Settings()
