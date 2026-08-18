Overview

An infrastructure benchmarking project comparing two leading open-source vector databases — Milvus and Weaviate — for the embedding-based similarity search workloads used across the facial recognition and scenery-matching systems built at Rapidev. The goal was to give the engineering team empirical, deployment-realistic data to inform which vector database to standardize on.

Key Features
Shared embedding source: Used a single DinoV2 embedding model for both databases to isolate database performance from model performance — evaluating raw vector search speed/accuracy only, no post-processing.
Unified comparison gateway: Single FastAPI service exposing /insert and /search endpoints that write/query both databases simultaneously and return results side-by-side.
Multi-environment deployment testing:
Google Colab — Embedded Weaviate + Milvus Lite, with Google Drive-mounted persistence (since Embedded Weaviate is Linux-only and doesn't run natively on Windows).
Weaviate Cloud — API-key authenticated managed deployment.
Local Docker — self-hosted Weaviate container alongside local Milvus Lite for a fully offline comparison.
Production debugging: Diagnosed and fixed a gRPC connection-flooding bug (GOAWAY / too_many_pings errors) in the Weaviate v4 Python client by switching from a per-request client to a single persistent global client instance.
Tech Stack

Python · FastAPI · Milvus (Milvus Lite) · Weaviate (v4 client, Cloud + Docker) · DinoV2 · Docker

Architecture
Raw Image → DinoV2 Embedding (shared)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Milvus Lite              Weaviate
 (COSINE index)          (Cloud / Docker)
        │                       │
        └───────────┬───────────┘
                     ▼
        FastAPI Comparison Gateway
        (/insert, /search — both DBs)
Setup
bash
git clone <repo-url>
cd vector-db-benchmark
pip install -r requirements.txt
docker compose up -d   # spins up local Weaviate
uvicorn main:app --reload

Set WEAVIATE_URL and WEAVIATE_API_KEY environment variables if using Weaviate Cloud instead of local Docker.

Key Learnings
Both Milvus and Weaviate perform vector-only search when fed pre-computed embeddings directly — neither does native semantic/text search unless configured with their own text-vectorization modules, which was out of scope for this comparison.
Client connection lifecycle management matters as much as the database engine itself — the gRPC flooding issue was a client-side architecture bug, not a Weaviate performance limitation, and disappeared once a persistent client was used.
Notes

Developed as part of an internship at Rapidev. Benchmark result tables are in /results.
