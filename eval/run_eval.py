"""Evaluation harness: runs the hand-labeled queries in eval/queries.py against all
three /search retrieval modes and reports Precision@5, Recall@5, and MRR for each,
writing the results table into README.md between the EVAL_TABLE markers.

Run: python -m eval.run_eval
"""

import asyncio
from pathlib import Path

from app.db import AsyncSessionLocal
from app.search import hybrid_search, keyword_search, vector_search
from eval.queries import LABELED_QUERIES

K = 5
MODES = {"keyword": keyword_search, "vector": vector_search, "hybrid": hybrid_search}
DISPLAY_NAMES = {"keyword": "Keyword (full-text)", "vector": "Vector only", "hybrid": "Hybrid (RRF)"}


def precision_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for pid in retrieved_ids[:k] if pid in relevant) / k


def recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for pid in retrieved_ids[:k] if pid in relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in relevant:
            return 1 / rank
    return 0.0


async def evaluate_mode(session, search_fn) -> dict[str, float]:
    precisions, recalls, rrs = [], [], []
    for query_text, relevant_ids in LABELED_QUERIES:
        relevant = set(relevant_ids)
        results = await search_fn(session, query_text, K)
        retrieved_ids = [r.product_id for r in results]

        precisions.append(precision_at_k(retrieved_ids, relevant, K))
        recalls.append(recall_at_k(retrieved_ids, relevant, K))
        rrs.append(reciprocal_rank(retrieved_ids, relevant, K))

    n = len(LABELED_QUERIES)
    return {
        "precision_at_5": sum(precisions) / n,
        "recall_at_5": sum(recalls) / n,
        "mrr": sum(rrs) / n,
    }


def build_table(results: dict[str, dict[str, float]]) -> str:
    lines = ["| Mode | Precision@5 | Recall@5 | MRR |", "|---|---|---|---|"]
    for mode_name in ["keyword", "vector", "hybrid"]:
        m = results[mode_name]
        lines.append(
            f"| {DISPLAY_NAMES[mode_name]} | {m['precision_at_5']:.3f} | {m['recall_at_5']:.3f} | {m['mrr']:.3f} |"
        )
    return "\n".join(lines)


def update_readme(table: str) -> None:
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    start_marker, end_marker = "<!-- EVAL_TABLE_START -->", "<!-- EVAL_TABLE_END -->"
    block = f"{start_marker}\n{table}\n{end_marker}"

    if start_marker in content and end_marker in content:
        pre = content.split(start_marker)[0]
        post = content.split(end_marker)[1]
        content = pre + block + post
    else:
        content += (
            f"\n\n## Retrieval evaluation ({len(LABELED_QUERIES)} hand-labeled queries)\n\n{block}\n"
        )

    readme_path.write_text(content, encoding="utf-8")
    print(f"\nUpdated {readme_path}")


async def main() -> None:
    async with AsyncSessionLocal() as session:
        results = {}
        for mode_name, search_fn in MODES.items():
            print(f"Evaluating mode={mode_name} over {len(LABELED_QUERIES)} queries...")
            results[mode_name] = await evaluate_mode(session, search_fn)

    table = build_table(results)
    print("\n" + table)
    update_readme(table)


if __name__ == "__main__":
    asyncio.run(main())
