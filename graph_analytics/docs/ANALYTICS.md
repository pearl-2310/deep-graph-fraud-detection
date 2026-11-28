
# Graph analytics notes

How the analytics layer works and why each choice was made.

## Graph projections

Algorithms run on one of two projections of the property graph: an *undirected simple graph* (edge weight = number of parallel links) for communities, embeddings and centrality, and an *account-to-account directed money-flow graph* for cycles and flow.

## Metrics

Degree, weighted degree and clustering are exact. Betweenness is *sampled* with k pivot nodes because exact betweenness is O(V*E). Shortest-path statistics are sampled on the largest connected component.
