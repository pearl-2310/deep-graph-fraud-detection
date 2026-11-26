# Graph Analytics Notes

## 1. Introduction

The Graph Analytics module analyzes relationships between different bank accounts and transactions. It helps identify important account connections, suspicious transaction patterns, and groups of interconnected accounts.

This module uses graph-based techniques to understand how accounts are connected and how money flows between them.

## 2. Graph Projections

The analytics layer works with two types of graph projections created from the property graph.

### 2.1 Undirected Simple Graph

An undirected simple graph represents connections between accounts without considering the direction of transactions.

- Each node represents an account.
- Each edge represents a connection between two accounts.
- Edge weight represents the number of parallel links between accounts.

**Applications:**
- Community Detection
- Graph Embeddings
- Centrality Analysis

### 2.2 Directed Money-Flow Graph

A directed money-flow graph represents the direction of financial transactions between accounts.

- Each node represents a bank account.
- Each directed edge represents a money transfer.
- The direction indicates the sender and receiver of the transaction.

**Applications:**
- Transaction Cycle Detection
- Money Flow Analysis
- Suspicious Transaction Pattern Analysis

## 3. Purpose of Graph Projections

The two graph projections are used because different graph algorithms require different representations of account relationships.

The undirected graph helps analyze account connectivity and communities, while the directed graph helps understand transaction direction and money movement.

## 4. Conclusion

Graph projections provide the foundation for the Graph Analytics and Embeddings module. They allow different algorithms to analyze account relationships, detect interconnected groups, study transaction flows, and support further fraud analysis.
