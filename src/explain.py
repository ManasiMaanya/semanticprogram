"""Per-cluster explanations from tags, distinctive terms, and representative problems.

The narrative is generated deterministically from the cluster's own metadata, so it
stays in sync with the artifacts. It answers "what makes these problems similar?" for
each cluster without a human writing sixteen blurbs by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.engine import ART, GENERIC_TAGS, load_processed

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _normalize_tag(tag: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", tag.lower()).strip()


def tag_share(frame, members: np.ndarray) -> list[dict]:
    """Share of cluster members carrying each specific (non-generic) tag.

    Spelling variants are merged ("Dynamic Programming" and "dynamic-programming"),
    using the most common original spelling as the display form.
    """
    counts: dict[str, int] = {}
    spellings: dict[str, dict[str, int]] = {}
    for tags in frame.iloc[members]["tags"]:
        for tag in tags:
            key = _normalize_tag(tag)
            if key in GENERIC_TAGS:
                continue
            counts[key] = counts.get(key, 0) + 1
            spellings.setdefault(key, {})
            spellings[key][tag] = spellings[key].get(tag, 0) + 1
    size = int(len(members))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "tag": max(spellings[key].items(), key=lambda item: item[1])[0],
            "count": count,
            "share": count / size,
        }
        for key, count in ranked
    ]


def representatives(frame, embeddings: np.ndarray, members: np.ndarray, n: int = 3) -> list[dict]:
    """The n problems closest to the cluster centroid, i.e. its most typical members."""
    centroid = embeddings[members].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    similarity = embeddings[members] @ centroid
    order = np.argsort(-similarity)[:n]
    rows = []
    for position in order:
        row = frame.iloc[int(members[int(position)])]
        rows.append(
            {
                "code": row["code"],
                "title": row["title"],
                "difficulty": int(row["difficulty"]),
                "similarity": float(similarity[int(position)]),
                "tags": list(row["tags"]),
                "link": row["link"],
            }
        )
    return rows


def narrative(cluster_id: int, label: str, size: int, tags: list[dict], terms: list[str], reps: list[dict]) -> str:
    sentences = [f"Cluster {cluster_id} holds {size} problems. Its shared theme is {label}."]
    strong = [tag for tag in tags[:2] if tag["share"] >= 0.15]
    if strong:
        primary = strong[0]
        clause = f"{primary['share'] * 100:.0f}% of members carry *{primary['tag']}*"
        if len(strong) > 1:
            clause += f" and {strong[1]['share'] * 100:.0f}% carry *{strong[1]['tag']}*"
        sentences.append(
            f"The common thread is {primary['tag']}: {clause}. "
            "These problems are linked by the technique, not a shared story."
        )
    else:
        sentences.append(
            "No single specific tag dominates, so these problems sit together on vocabulary "
            "and problem shape rather than one named technique."
        )
    if terms:
        sentences.append("The most distinctive words are " + ", ".join(f"`{word}`" for word in terms) + ".")
    if reps:
        codes = ", ".join(f"`{row['code']}`" for row in reps)
        sentences.append(f"Problems nearest the cluster centre, its most typical members, are {codes}.")
    return " ".join(sentences)


def explain_all() -> dict:
    frame = load_processed(ART / "problems.jsonl")
    embeddings = np.load(ART / "embeddings.npy")
    clusters = json.loads((ART / "clusters.json").read_text(encoding="utf-8"))
    labels = frame["cluster"].to_numpy()

    explained: dict[int, dict] = {}
    for cluster_id, info in sorted(clusters.items(), key=lambda item: int(item[0])):
        members = np.where(labels == int(cluster_id))[0]
        tags = tag_share(frame, members)
        terms = info["distinctive_terms"]
        reps = representatives(frame, embeddings, members, n=3)
        difficulty = frame.iloc[members]["difficulty"]
        explained[int(cluster_id)] = {
            "label": info["label"],
            "size": int(len(members)),
            "top_specific_tags": tags[:6],
            "distinctive_terms": terms,
            "difficulty_min": int(difficulty.min()),
            "difficulty_max": int(difficulty.max()),
            "representatives": reps,
            "narrative": narrative(int(cluster_id), info["label"], len(members), tags, terms, reps),
        }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "clusters_explained.json").write_text(
        json.dumps(explained, indent=2), encoding="utf-8"
    )
    (REPORTS / "clusters_explained.md").write_text(render_markdown(explained), encoding="utf-8")
    return explained


def render_markdown(explained: dict) -> str:
    lines = [
        "# What each cluster is about",
        "",
        "Every cluster is summarised from its own members: the specific topic tags they share, "
        "the TF-IDF words that set the cluster apart, and the three problems closest to its "
        "centre (the most typical members). Tags are labels, not search features. The narrative "
        "is generated by `src/explain.py`, so it stays consistent with `artifacts/`.",
        "",
        f"Problems indexed: {sum(row['size'] for row in explained.values())} across {len(explained)} clusters.",
        "",
    ]
    for cluster_id, row in sorted(explained.items()):
        lines.append(f"## Cluster {cluster_id} — {row['label']} ({row['size']} problems)")
        lines.append("")
        lines.append(row["narrative"])
        lines.append("")
        if row["top_specific_tags"]:
            tags = ", ".join(f"{tag['tag']} ({tag['share'] * 100:.0f}%)" for tag in row["top_specific_tags"])
            lines.append(f"- **Specific tags:** {tags}")
        else:
            lines.append("- **Specific tags:** none — this cluster is named from its vocabulary alone")
        if row["distinctive_terms"]:
            lines.append(f"- **Distinctive words:** {', '.join(row['distinctive_terms'])}")
        lines.append(f"- **Difficulty range:** {row['difficulty_min']}–{row['difficulty_max']}")
        lines.append("- **Representative problems:**")
        for rep in row["representatives"]:
            tags = ", ".join(rep["tags"]) or "—"
            lines.append(
                f"  - [`{rep['code']}`]({rep['link']}) — {rep['title']} "
                f"(rating {rep['difficulty']}, cosine {rep['similarity']:.2f}; tags: {tags})"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    result = explain_all()
    print(f"explained {len(result)} clusters -> reports/clusters_explained.md")
