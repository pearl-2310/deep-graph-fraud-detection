"""Week 4: neighbourhood-based risk scoring (framework-free, so it is unit-testable without FastAPI)."""
from __future__ import annotations


def risk_level(p: float) -> str:
    return "HIGH" if p >= 0.8 else "MEDIUM" if p >= 0.5 else "LOW" if p >= 0.2 else "MINIMAL"
