# Git workflow & daily commit plan

The brief requires a GitHub repo with **commits every day** and professional history. Commit as you actually do the work
(small, focused commits are better than one big daily dump). Suggested split - adapt it to what you really finish each day.

## Branching
```
main                      <- always runnable; protected, merge via pull request
 +- feature/week1-graph-model
 +- feature/week2-analytics
 +- feature/week3-detection-model
 +- feature/week4-api-dashboard
```
Open a PR per feature branch, squash-merge, tag the milestones: `v0.2-mid-review`, `v1.0-final`.

## Commit message convention (Conventional Commits)
`feat: add cycle detector with time-feasibility check` · `fix: make cycle retention rotation-invariant` ·
`test: ring split never leaks across train/test` · `docs: explain graph schema` · `chore: add CI workflow`

## Day-by-day plan
| Week | Day | Suggested commits |
|---|---|---|
| 1 | Mon | `chore: init repo, .gitignore, pyproject, README skeleton` |
| 1 | Tue | `feat(data): synthetic generator with planted fraud rings` |
| 1 | Wed | `feat(graph): relational tables -> NetworkX property graph` · `docs: graph schema` |
| 1 | Thu | `chore: docker-compose for Neo4j + GDS` · `feat(graph): Neo4j batch loader` |
| 1 | Fri | `test: schema + loader tests` · `docs: Cypher cheat-sheet` |
| 2 | Mon | `feat(analytics): degree, sampled betweenness, shortest-path summary` |
| 2 | Tue | `feat(analytics): Louvain (+ Leiden option) and community table` |
| 2 | Wed | `feat(analytics): node2vec embeddings` · `feat: embedding sanity check` |
| 2 | Thu | `feat(analytics): GraphSAGE (PyG) embeddings` |
| 2 | Fri | `test: analytics tests` · **tag `v0.2-mid-review`** (property graph + embeddings + communities) |
| 3 | Mon | `feat(ranking): PageRank, personalized PageRank, TrustRank-style scoring` |
| 3 | Tue | `feat(cycles): bounded cycle search with time + value-retention filters` |
| 3 | Wed | `feat(cycles): max-flow round-trip detector, SCC analysis` |
| 3 | Thu | `feat(model): label-free features + ring-held-out split` |
| 3 | Fri | `feat(model): XGBoost training, ablation study, metrics.json` |
| 4 | Mon | `feat(api): FastAPI /score endpoints` · `feat: inductive scoring for new entities` |
| 4 | Tue | `feat(dashboard): Cytoscape.js cluster viewer in Streamlit` |
| 4 | Wed | `feat(report): typology classifier + taxonomy report generator` |
| 4 | Thu | `docs: README results, limitations` · `ci: GitHub Actions` |
| 4 | Fri | polish, run full pipeline, regenerate report · **tag `v1.0-final`** |

## First push
```bash
git init && git add . && git commit -m "chore: initial project scaffold"
git branch -M main
git remote add origin https://github.com/<you>/graph-fraud-detection.git
git push -u origin main
```
