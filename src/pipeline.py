"""Run preprocessing, indexing, evaluation, and plots."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluate import demo_queries, run as evaluate
from src.preprocess import preprocess, save
from src.engine import build
from src.explain import explain_all
from src.visualize import plot

ROOT = Path(__file__).resolve().parents[1]


def exploration(frame) -> dict:
    from src.preprocess import load_raw

    raw = load_raw()
    difficulties = frame["difficulty"].astype(int)
    tag_counts: dict[str, int] = {}
    for tags in frame["tags"]:
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:20]
    return {
        "downloaded": len(raw),
        "kept_after_cleaning": int(len(frame)),
        "dropped": len(raw) - int(len(frame)),
        "with_concept_tags": int((frame["tags"].map(len) > 0).sum()),
        "without_concept_tags": int((frame["tags"].map(len) == 0).sum()),
        "difficulty_min": int(difficulties.min()),
        "difficulty_median": float(difficulties.median()),
        "difficulty_max": int(difficulties.max()),
        "median_words": float(frame["text"].str.split().map(len).median()),
        "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
    }


def main() -> None:
    frame = preprocess()
    path = save(frame)
    summary = exploration(frame)
    summary["processed_path"] = str(path)
    print(json.dumps({k: summary[k] for k in summary if k != "top_tags"}, indent=2), flush=True)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "exploration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build(frame)
    plot()
    explained = explain_all()
    print(f"explained {len(explained)} clusters", flush=True)
    metrics = evaluate()
    meta_path = ROOT / "artifacts" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["hybrid_alpha"] = metrics["best_alpha"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    demos = demo_queries(metrics["best_alpha"])
    print(f"demo queries: {len(demos)}", flush=True)
    (ROOT / "reports" / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
