# Deep Graph Network Analytics for Syndicated Fraud Detection

Rule-based checks look at one account at a time and miss **rings**: groups of accounts, devices and synthetic identities
that only look suspicious *together*. This project models transactions as a multi-partite property graph and uses
community detection, embeddings, PageRank variants, cycle / max-flow analysis and a gradient-boosted classifier to find
and explain those rings, then serves the scores through an API and an investigator dashboard.

```
CSV / claims / P2P  ->  property graph (Neo4j + NetworkX)  ->  metrics - communities - embeddings
                                                                       |
         dashboard (Cytoscape.js)  <-  taxonomy report  <-  XGBoost  <-  PageRank - cycles - max-flow
                                   <-  FastAPI /score  <-----------'
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python -m fraudgraph.pipeline        # generate data -> analyse -> train -> report   (~1 min)
pytest -q                            # unit + end-to-end tests (API tests need fastapi+httpx)
uvicorn fraudgraph.api.main:app --reload     # http://localhost:8000/docs
streamlit run dashboard/app.py               # investigator console
```
Optional Neo4j: `docker compose up -d` then `python -m fraudgraph.pipeline --stage analyze --neo4j`
(browse the graph at http://localhost:7474, user `neo4j`, password in `.env.example`).