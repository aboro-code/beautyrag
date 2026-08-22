# BeautyRAG — Hybrid Search & RAG Shopping Assistant

Async FastAPI backend for grounded Q&A and recommendations over a skincare/beauty
product catalog, using hybrid (vector + keyword) search over Postgres/pgvector.

_This README is being filled in as the project is built — architecture diagram,
API docs, and the evaluation table land once those pieces exist._

## Future improvements

- **Automated test suite.** Right now correctness is checked manually (`/health`,
  ad-hoc queries) plus the retrieval-quality eval harness (`eval/run_eval.py`).
  A `pytest` suite covering the API routes (e.g. unknown `product_id`, empty
  search results, malformed requests) would be a natural next step beyond the
  weekend build.
