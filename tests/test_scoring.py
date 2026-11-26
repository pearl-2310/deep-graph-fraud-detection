from fraudgraph.scoring import risk_level


def test_risk_levels():
    assert [risk_level(p) for p in (0.9, 0.6, 0.3, 0.05)] == ["HIGH", "MEDIUM", "LOW", "MINIMAL"]
