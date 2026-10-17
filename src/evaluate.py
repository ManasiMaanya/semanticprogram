"""Tag-based retrieval evaluation, hybrid-weight selection, and error analysis.

Topic tags are held out of the index. A retrieved problem is relevant when it
shares a specific tag with the query. Tags that cover more than 20% of the
tagged collection are treated as generic and ignored, so "Algorithms" does not
mark hundreds of unrelated problems as relevant. Dynamic programming stays in,
because it is a real technique even though it is common.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.engine import ART, Retriever, load_processed

REPORTS = Path(__file__).resolve().parents[1] / "reports"


TAG_ALIASES = {
    "graph algos": "graphs",
    "graph algorithms": "graphs",
    "tree algos": "trees",
    "tree algorithms": "trees",
    "dynamic programming": "dp",
    "dynamic-programming": "dp",
    "ad-hoc": "ad hoc",
    "basic math": "math",
    "mathematics": "math",
    "simple math": "math",
    "string algos": "strings",
    "string algorithms": "strings",
    "sorting algos": "sorting",
    "binary search algos": "binary search",
    "greedy algorithms": "greedy",
    "shortest path algorithms": "shortest paths",
}


def canonicalize(tag: str) -> str:
    tag = tag.lower().strip()
    return TAG_ALIASES.get(tag, tag)


# These labels mean "this is a programming problem", not a technique.
ALWAYS_GENERIC = {
    "algorithms",
    "mathematics",
    "math",
    "basic math",
    "simple math",
    "data structures",
    "basic programming concepts",
}


def specific_tags(frame: pd.DataFrame, max_share: float = 0.20) -> tuple[list[set[str]], set[str]]:
    counts: dict[str, int] = {}
    lowered = []
    for tags in frame["tags"]:
        keys = {canonicalize(tag) for tag in tags} - ALWAYS_GENERIC
        lowered.append(keys)
        for tag in keys:
            counts[tag] = counts.get(tag, 0) + 1
    tagged = max(1, sum(1 for keys in lowered if keys))
    generic = set(ALWAYS_GENERIC) | {tag for tag, count in counts.items() if count / tagged > max_share}
    specific = [keys - generic for keys in lowered]
    return specific, generic


def tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z]{3,}", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def family_key(title: str) -> str:
    import re

    title = title.lower()
    title = re.sub(r"\(.*?\)", " ", title)
    title = re.sub(r"\b(easy|hard|medium|version)\b", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def ranking_metrics(flags: list[bool], n_relevant: int, k: int) -> dict[str, float]:
    hits = 0
    precision_sum = 0.0
    dcg = 0.0
    first = None
    for rank, rel in enumerate(flags[:k], start=1):
        if not rel:
            continue
        hits += 1
        precision_sum += hits / rank
        dcg += 1.0 / np.log2(rank + 1)
        if first is None:
            first = rank
    ideal_hits = min(k, n_relevant)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        f"precision@{k}": hits / k,
        f"recall@{k}": hits / n_relevant if n_relevant else 0.0,
        f"ap@{k}": precision_sum / ideal_hits if ideal_hits else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        "mrr": 1.0 / first if first else 0.0,
    }


def ranked_flags(order: np.ndarray, query_index: int, relevant: set[int], k: int) -> list[bool]:
    flags = []
    for index in order:
        if int(index) == query_index:
            continue
        flags.append(int(index) in relevant)
        if len(flags) >= k:
            break
    return flags


def evaluate_split(
    semantic: np.ndarray,
    keyword: np.ndarray,
    queries: list[int],
    relevance: list[set[int]],
    alpha: float,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for query in queries:
        n_rel = len(relevance[query])
        if n_rel == 0:
            continue
        scores = alpha * semantic[query] + (1.0 - alpha) * keyword[query]
        order = np.argsort(-scores)
        flags = ranked_flags(order, query, relevance[query], 10)
        stats = ranking_metrics(flags, n_rel, 10)
        for name, value in ranking_metrics(flags, n_rel, 5).items():
            if name != "mrr":
                stats[name] = value
        for name, value in stats.items():
            buckets.setdefault(name, []).append(value)
    return {name: float(np.mean(values)) for name, values in buckets.items()}


def mode_scores(semantic, keyword, mode: str, alpha: float) -> np.ndarray:
    if mode == "semantic":
        return semantic
    if mode == "keyword":
        return keyword
    return alpha * semantic + (1.0 - alpha) * keyword


def run() -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = load_processed(ART / "problems.jsonl")
    embeddings = np.load(ART / "embeddings.npy")
    from joblib import load

    vectorizer = load(ART / "tfidf.joblib")
    tfidf = normalize(vectorizer.transform(frame["text"].tolist()))
    keyword = cosine_similarity(tfidf)
    semantic = embeddings @ embeddings.T

    tag_sets, generic = specific_tags(frame)
    relevance = []
    for i, tags in enumerate(tag_sets):
        if not tags:
            relevance.append(set())
            continue
        hits = {j for j, other in enumerate(tag_sets) if j != i and tags & other}
        relevance.append(hits)

    queries = [i for i, rel in enumerate(relevance) if len(rel) >= 3 and tag_sets[i]]
    rng = np.random.default_rng(17)
    rng.shuffle(queries)
    cut = len(queries) // 2
    tune, test = queries[:cut], queries[cut:]

    grid = [0.0, 0.35, 0.5, 0.65, 0.8, 1.0]
    tuning = []
    best_alpha, best_map = 0.65, -1.0
    for alpha in grid:
        metrics = evaluate_split(semantic, keyword, tune, relevance, alpha)
        tuning.append({"alpha": alpha, **metrics})
        if metrics.get("ap@10", -1) > best_map:
            best_alpha, best_map = alpha, metrics["ap@10"]

    comparison = {}
    for mode, alpha in (("keyword", 0.0), ("semantic", 1.0), ("hybrid", best_alpha)):
        comparison[mode] = evaluate_split(semantic, keyword, test, relevance, alpha)

    # Qualitative slices use the full query pool and the selected hybrid weight.
    cross_story = []
    errors = []
    seen_pairs = set()
    texts = frame["text"].tolist()
    for query in queries:
        order = [int(i) for i in np.argsort(-semantic[query]) if int(i) != query]
        top = order[0]
        shared = sorted(tag_sets[query] & tag_sets[top])
        overlap = jaccard(texts[query], texts[top])
        record = {
            "query_code": frame.at[query, "code"],
            "query_title": frame.at[query, "title"],
            "query_tags": sorted(tag_sets[query]),
            "neighbor_code": frame.at[top, "code"],
            "neighbor_title": frame.at[top, "title"],
            "neighbor_tags": sorted(tag_sets[top]),
            "shared_tags": shared,
            "semantic": float(semantic[query, top]),
            "keyword": float(keyword[query, top]),
            "lexical_jaccard": overlap,
        }
        for neighbor in order[:20]:
            shared_n = sorted(tag_sets[query] & tag_sets[neighbor])
            overlap_n = jaccard(texts[query], texts[neighbor])
            pair = tuple(sorted((frame.at[query, "code"], frame.at[neighbor, "code"])))
            if (
                shared_n
                and overlap_n < 0.10
                and semantic[query, neighbor] >= 0.55
                and pair not in seen_pairs
            ):
                seen_pairs.add(pair)
                cross_story.append(
                    {
                        **record,
                        "neighbor_code": frame.at[neighbor, "code"],
                        "neighbor_title": frame.at[neighbor, "title"],
                        "neighbor_tags": sorted(tag_sets[neighbor]),
                        "shared_tags": shared_n,
                        "semantic": float(semantic[query, neighbor]),
                        "keyword": float(keyword[query, neighbor]),
                        "lexical_jaccard": overlap_n,
                        "rank": order.index(neighbor) + 1,
                    }
                )
                break
        same_problem = family_key(frame.at[query, "title"]) == family_key(frame.at[top, "title"])
        if not shared and tag_sets[top] and not same_problem:
            kind = "shared wording, different concept" if overlap >= 0.12 or keyword[query, top] >= 0.2 else "similar narrative frame"
            errors.append({**record, "kind": kind})

    cross_story.sort(key=lambda row: row["semantic"] - row["lexical_jaccard"], reverse=True)
    errors.sort(key=lambda row: row["semantic"], reverse=True)

    summary = {
        "n_problems": int(len(frame)),
        "n_queries": len(queries),
        "n_tune": len(tune),
        "n_test": len(test),
        "median_relevant": float(np.median([len(relevance[i]) for i in queries])),
        "mean_relevant": float(np.mean([len(relevance[i]) for i in queries])),
        "generic_tags_ignored": sorted(generic),
        "best_alpha": best_alpha,
        "tuning": tuning,
        "test_comparison": comparison,
        "cross_story_count": len(cross_story),
        "error_count": len(errors),
        "cross_story_examples": cross_story[:12],
        "error_examples": errors[:12],
    }
    (REPORTS / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS / "evaluation.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"best_alpha": best_alpha, "test": comparison}, indent=2))
    return summary


def render_markdown(summary: dict) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        "Relevance is automatic and tag-based. A problem is relevant to a query when they share at least one specific topic tag. Tags used by more than 20% of tagged problems are ignored (`Algorithms`, `Mathematics`, `Data Structures`). Dynamic programming is kept. The query problem itself is excluded. Hybrid weight `alpha` is the semantic share, chosen by AP@10 on the tuning half and reported on the held-out half.",
        "",
        f"- Problems indexed: {summary['n_problems']}",
        f"- Queries with at least 3 relevant problems: {summary['n_queries']} (tune {summary['n_tune']}, test {summary['n_test']})",
        f"- Median relevant problems per query: {summary['median_relevant']:.0f}. Recall@10 stays small because most queries have many more than 10 tagged cousins.",
        f"- Selected hybrid alpha: {summary['best_alpha']}",
        f"- Generic tags ignored: {', '.join(summary['generic_tags_ignored']) or 'none'}",
        "",
        "## Held-out comparison",
        "",
        "| Method | P@5 | P@10 | R@10 | AP@10 | nDCG@10 | MRR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in ("keyword", "semantic", "hybrid"):
        metrics = summary["test_comparison"][mode]
        lines.append(
            "| {mode} | {p5:.3f} | {p:.3f} | {r:.3f} | {a:.3f} | {n:.3f} | {m:.3f} |".format(
                mode=mode,
                p5=metrics["precision@5"],
                p=metrics["precision@10"],
                r=metrics["recall@10"],
                a=metrics["ap@10"],
                n=metrics["ndcg@10"],
                m=metrics["mrr"],
            )
        )
    lines += [
        "",
        "## Same concept, different wording",
        "",
        "Each pair shares a specific topic tag, has token Jaccard under 0.10, and semantic similarity of at least 0.55. The neighbor is inside the top 20 by embedding similarity, so the stories can differ while the technique matches.",
        "",
    ]
    if not summary["cross_story_examples"]:
        lines.append("No pair met the threshold.")
    for row in summary["cross_story_examples"]:
        lines.append(
            f"- `{row['query_code']}` {row['query_title']} → `{row['neighbor_code']}` {row['neighbor_title']} "
            f"(shared: {', '.join(row['shared_tags'])}; semantic {row['semantic']:.2f}; Jaccard {row['lexical_jaccard']:.3f})"
        )
    lines += ["", "## Near misses", "", "Nearest neighbor shares no specific tag. `kind` is a guess from lexical overlap.", ""]
    for row in summary["error_examples"]:
        lines.append(
            f"- `{row['query_code']}` {row['query_title']} → `{row['neighbor_code']}` {row['neighbor_title']} "
            f"({row['kind']}; semantic {row['semantic']:.2f}; keyword {row['keyword']:.2f})"
        )
    lines.append("")
    return "\n".join(lines)


def demo_queries(alpha: float) -> list[dict]:
    """Statements written for this project, not copied from CodeChef."""
    prompts = [
        {
            "name": "Longest increasing subsequence",
            "text": "Chef has a row of stones, each with a height. He wants the longest sequence of stones he can keep, from left to right, so that every kept stone is strictly taller than the one before it. Stones he discards disappear. How many stones can he keep?",
        },
        {
            "name": "Shortest path with a disguise",
            "text": "A courier must travel between cities. Each road has a travel time. Find the smallest total time to go from the capital to every other city. Roads are one-way.",
        },
        {
            "name": "Binary search on the answer",
            "text": "You may place at most K painters. A painter who works for T minutes can paint any set of fence segments whose lengths sum to at most T, and the segments assigned to one painter must be contiguous. What is the minimum T that finishes the fence?",
        },
        {
            "name": "Range sum queries",
            "text": "An array starts with N integers. You must answer two kinds of operations: add a value to every element from index L to R, and report the sum of elements from L to R. There are many operations.",
        },
        {
            "name": "Modular exponentiation",
            "text": "Given integers A, B and a prime modulus M, compute A raised to the power B modulo M. B can be as large as 10^18, so multiplying A by itself B times is too slow.",
        },
        {
            "name": "Connected components",
            "text": "There are N villages and M bidirectional roads. Two villages are in the same kingdom when you can walk between them using roads. How many kingdoms are there?",
        },
    ]
    retriever = Retriever()
    results = []
    for prompt in prompts:
        hits = retriever.search(prompt["text"], mode="hybrid", k=5, alpha=alpha)
        results.append(
            {
                "name": prompt["name"],
                "query": prompt["text"],
                "hits": hits.to_dict(orient="records"),
            }
        )
    (REPORTS / "demo_queries.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
