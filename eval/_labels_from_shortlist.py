"""One-off helper: regenerates eval/queries.py's LABELED_QUERIES from
eval/eval_ground_truth_labels.json.

That JSON is the actual ground truth: a two-stage process where Claude Code produced
a purely mechanical SQL/ILIKE candidate shortlist per query (no embeddings, no
ranking, no relevance judgment -- see eval/candidate_shortlist.md, not committed),
and a human then judged every candidate row by hand. This replaces the eval set
previously built by eval/_build_queries.py, whose labels came from targeted
category/keyword filters rather than individual human judgment.

Run once after eval_ground_truth_labels.json changes; the output (eval/queries.py)
is committed as static data so the eval itself doesn't depend on re-reading the JSON.

Run: python -m eval._labels_from_shortlist
"""

import json
from pathlib import Path

GROUND_TRUTH_PATH = Path(__file__).parent / "eval_ground_truth_labels.json"
QUERIES_PATH = Path(__file__).parent / "queries.py"


def main() -> None:
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    methodology_note = ground_truth["methodology_note"]
    labeled = ground_truth["labeled_queries"]

    lines = [f'"""{methodology_note}"""', "", "LABELED_QUERIES: list[tuple[str, list[str]]] = ["]
    for qdata in labeled.values():
        lines.append(f"    {(qdata['query_text'], qdata['relevant_product_ids'])!r},")
    lines.append("]")

    QUERIES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {QUERIES_PATH} with {len(labeled)} labeled queries.")


if __name__ == "__main__":
    main()
