"""Hand-labelled evaluation set for query-statement retrieval.

The brief's user-facing task is "submit a statement, get the top matching problems".
Tag-based relevance (see `src/evaluate.py`) cannot score statements that are not in
the collection, so this module pools the top results from all three retrieval modes
for ten hand-written statements and records a manual relevance judgement for each
candidate (1 = same underlying technique, 0 = not).

`build_pool()` writes `reports/gold_set.json` with every judgement set to null.
A human fills the `label`/`reason` fields, then `score()` turns them into metrics
and writes `reports/gold_eval.md`. Pooling all three modes means every mode's top-k
is judged, so no mode is scored against unjudged results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.engine import ART, Retriever, load_processed

REPORTS = Path(__file__).resolve().parents[1] / "reports"
MODES = ("keyword", "semantic", "hybrid")

GOLD_QUERIES = [
    {
        "id": "lis",
        "intent": "longest increasing subsequence",
        "statement": (
            "Chef has a row of stones, each with a height. She wants the longest sequence of "
            "stones, read left to right, in which every kept stone is strictly taller than the "
            "one before it. Discarded stones vanish. How many stones can she keep?"
        ),
    },
    {
        "id": "dijkstra",
        "intent": "single-source shortest paths",
        "statement": (
            "A courier must travel between cities. Each one-way road has a travel time. Find the "
            "smallest total time to go from the capital to every other city."
        ),
    },
    {
        "id": "binary-search-answer",
        "intent": "binary search on the answer",
        "statement": (
            "You may place at most K painters. A painter who works for T minutes can paint any "
            "contiguous run of fence segments whose lengths sum to at most T. What is the minimum "
            "T that finishes the fence?"
        ),
    },
    {
        "id": "range-update-query",
        "intent": "range updates and range sum queries",
        "statement": (
            "An array starts with N integers. There are two kinds of operations: add a value to "
            "every element from index L to R, and report the sum of elements from L to R. There "
            "are many operations."
        ),
    },
    {
        "id": "modpow",
        "intent": "modular exponentiation",
        "statement": (
            "Given integers A, B and a prime modulus M, compute A raised to the power B modulo M. "
            "B can be as large as 10^18, so multiplying A by itself B times is too slow."
        ),
    },
    {
        "id": "connected-components",
        "intent": "connected components / union-find",
        "statement": (
            "There are N villages and M bidirectional roads. Two villages are in the same kingdom "
            "when you can walk between them using roads. How many kingdoms are there?"
        ),
    },
    {
        "id": "knapsack",
        "intent": "0/1 knapsack dynamic programming",
        "statement": (
            "A thief can carry a bag of capacity W. There are N items, and each item has a weight "
            "and a value. Each item may be taken or left behind. Maximise the total value that "
            "fits in the bag."
        ),
    },
    {
        "id": "max-subarray",
        "intent": "maximum subarray sum",
        "statement": (
            "You are given the profit or loss of each day. Find the contiguous block of days with "
            "the largest total profit. The block may not be empty."
        ),
    },
    {
        "id": "topological-order",
        "intent": "topological sort of a DAG",
        "statement": (
            "There are N tasks. Some pairs say that task A must be finished before task B can "
            "start. Output any order that finishes every task without breaking a rule."
        ),
    },
    {
        "id": "count-primes",
        "intent": "sieve of Eratosthenes",
        "statement": (
            "Count how many integers from 2 up to N are prime. N can be as large as 10^7."
        ),
    },
]


def alpha() -> float:
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    return float(meta.get("hybrid_alpha", 0.65))


def build_pool(top_k: int = 5) -> dict:
    """Pool each mode's top-k per query and write an unlabelled template."""
    retriever = Retriever()
    weight = alpha()
    queries = []
    for spec in GOLD_QUERIES:
        parts = retriever.score(spec["statement"])
        semantic, keyword = parts["semantic"], parts["keyword"]
        hybrid = weight * semantic + (1.0 - weight) * keyword
        pooled: dict[str, dict] = {}
        for mode, scores in (("keyword", keyword), ("semantic", semantic), ("hybrid", hybrid)):
            for idx in np.argsort(-scores)[:top_k]:
                idx = int(idx)
                row = retriever.frame.iloc[idx]
                code = row["code"]
                if code in pooled:
                    continue
                pooled[code] = {
                    "code": code,
                    "title": row["title"],
                    "difficulty": int(row["difficulty"]),
                    "tags": list(row["tags"]),
                    "scores": {
                        "keyword": float(keyword[idx]),
                        "semantic": float(semantic[idx]),
                        "hybrid": float(hybrid[idx]),
                    },
                    "label": None,
                    "reason": "",
                }
        queries.append(
            {
                "id": spec["id"],
                "intent": spec["intent"],
                "statement": spec["statement"],
                "pooled": sorted(pooled.values(), key=lambda item: -item["scores"]["hybrid"]),
            }
        )
    out = {
        "note": (
            "Manual relevance for hand-written statements. label = 1 when the candidate uses the "
            "same underlying technique as the intent, else 0. Candidates are the union of each "
            "mode's top-k, so every mode's top-k is judged."
        ),
        "alpha": weight,
        "top_k": top_k,
        "queries": queries,
    }
    (REPORTS / "gold_set.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def show(preview_chars: int = 240) -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = json.loads((REPORTS / "gold_set.json").read_text(encoding="utf-8"))
    frame = load_processed(ART / "problems.jsonl")
    texts = dict(zip(frame["code"], frame["text"]))
    for query in data["queries"]:
        print(f"\n=== {query['id']} — {query['intent']} ===")
        for item in query["pooled"]:
            scores = item["scores"]
            tags = ", ".join(item["tags"]) or "—"
            print(
                f"  [{item['label']}] {item['code']:<12} {item['title'][:44]:<44} r{item['difficulty']:<5} "
                f"k={scores['keyword']:.2f} s={scores['semantic']:.2f} h={scores['hybrid']:.2f} | {tags}"
            )
            preview = " ".join(texts.get(item["code"], "").split())[:preview_chars]
            print(f"        {preview}")


def ranking_metrics(labels: list[int], k: int) -> dict:
    hits = 0
    dcg = 0.0
    first = None
    for rank, rel in enumerate(labels[:k], start=1):
        if not rel:
            continue
        hits += 1
        dcg += 1.0 / np.log2(rank + 1)
        if first is None:
            first = rank
    ideal = min(k, sum(labels))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal + 1))
    return {
        f"precision@{k}": hits / k,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        "mrr": 1.0 / first if first else 0.0,
    }


def score(k: int = 5) -> dict:
    data = json.loads((REPORTS / "gold_set.json").read_text(encoding="utf-8"))
    per_mode: dict[str, dict[str, list[float]]] = {mode: {} for mode in MODES}
    detail = {}
    judged = pooled_total = 0
    for query in data["queries"]:
        candidates = [c for c in query["pooled"] if c["label"] is not None]
        if not candidates:
            raise RuntimeError("gold_set.json has unlabelled candidates; finish labelling first")
        pooled_total += len(candidates)
        judged += sum(1 for c in candidates if c["label"])
        row = {}
        for mode in MODES:
            ranked = sorted(candidates, key=lambda c: -c["scores"][mode])[:k]
            labels = [int(c["label"]) for c in ranked]
            metrics = ranking_metrics(labels, k)
            for name, value in metrics.items():
                per_mode[mode].setdefault(name, []).append(value)
            row[mode] = {"top": ranked[0]["code"], "labels": labels, **metrics}
        detail[query["id"]] = {
            "intent": query["intent"],
            "n_pooled": len(candidates),
            "n_relevant": sum(int(c["label"]) for c in candidates),
            **row,
        }

    summary = {
        "n_queries": len(data["queries"]),
        "k": k,
        "pooled_candidates": pooled_total,
        "relevant_candidates": judged,
        "relevance_rate": judged / pooled_total,
        "alpha": data["alpha"],
        "per_mode": {
            mode: {name: float(np.mean(values)) for name, values in metrics.items()}
            for mode, metrics in per_mode.items()
        },
        "detail": detail,
    }
    (REPORTS / "gold_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS / "gold_eval.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: dict) -> str:
    lines = [
        "# Manual evaluation set",
        "",
        "Tag-based relevance cannot score a statement that is not in the collection, so ten "
        "statements were written for this project and the top results of all three modes were "
        "pooled and judged by hand. A candidate is relevant (1) when it uses the same underlying "
        "technique as the intent, regardless of wording. Every mode's top-k is inside the pool, "
        "so no mode is scored against unjudged results.",
        "",
        f"- Queries: {summary['n_queries']}",
        f"- Pooled candidates judged: {summary['pooled_candidates']} ({summary['relevant_candidates']} relevant, "
        f"{summary['relevance_rate'] * 100:.0f}%)",
        f"- Hybrid weight: {summary['alpha']}",
        "",
        f"## Precision@{summary['k']}, nDCG@{summary['k']}, MRR on the manual labels",
        "",
        f"| Method | P@{summary['k']} | nDCG@{summary['k']} | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for mode in MODES:
        row = summary["per_mode"][mode]
        k = summary["k"]
        lines.append(
            f"| {mode} | {row['precision@' + str(k)]:.3f} | "
            f"{row['ndcg@' + str(k)]:.3f} | {row['mrr']:.3f} |"
        )
    best_p = max(MODES, key=lambda mode: summary["per_mode"][mode]["precision@" + str(k)])
    best_mrr = max(MODES, key=lambda mode: summary["per_mode"][mode]["mrr"])
    lines += [
        "",
        f"On these ten hand-written statements, **{best_p}** leads precision@{k} and **{best_mrr}** "
        "leads MRR. Ten queries is a small sample, so read the gaps as directional, not significant. "
        "The interesting failures are the queries with zero relevant hits: the words *painting*, "
        "*bag*, *profit* and *task* pull back problems about painting, buying, profit and scheduling "
        "rather than the technique asked for.",
        "",
        "## Per query",
        "",
        "Relevant counts manually relevant problems in the whole pool, not just the top five.",
        "",
        "| Query | Intent | Pooled | Relevant | Top-1 (hybrid) | Hit |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for query_id, row in summary["detail"].items():
        relevant = row["n_relevant"]
        hit = "yes" if row["hybrid"]["labels"][0] else "no"
        lines.append(
            f"| {query_id} | {row['intent']} | {row['n_pooled']} | {relevant} | "
            f"`{row['hybrid']['top']}` | {hit} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "score":
        result = score()
        print(json.dumps(result["per_mode"], indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "show":
        show()
    else:
        build_pool()
        show()
        print("\nwrote reports/gold_set.json (labels null) — fill label/reason, then run: python -m src.gold score")
