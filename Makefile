.PHONY: install data run test api dashboard neo4j lint
install:   ; pip install -r requirements.txt && pip install -e .
data:      ; python -m fraudgraph.data.generate
run:       ; python -m fraudgraph.pipeline            # generate -> analyze -> train -> report
test:      ; pytest -q
api:       ; uvicorn fraudgraph.api.main:app --reload
dashboard: ; streamlit run dashboard/app.py
neo4j:     ; docker compose up -d && python -m fraudgraph.graph.neo4j_loader
lint:      ; ruff check src tests dashboard
