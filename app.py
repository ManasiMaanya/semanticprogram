"""Interactive explorer for similar CodeChef problems."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.engine import ART, Retriever

REPORTS = Path(__file__).resolve().parent / "reports"

st.set_page_config(page_title="Déjà Code", layout="wide")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_data
def load_explanations() -> dict:
    path = REPORTS / "clusters_explained.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def show_hits(hits: pd.DataFrame) -> None:
    for _, row in hits.iterrows():
        tags = ", ".join(row["tags"]) if isinstance(row["tags"], list) else ""
        st.markdown(
            f"**[{row['code']}]({row['link']}) — {row['title']}**  \n"
            f"Score {row['score']:.3f} · semantic {row.get('semantic', float('nan')):.3f} · "
            f"keyword {row.get('keyword', float('nan')):.3f} · rating {int(row['difficulty'])}  \n"
            f"Cluster: {row.get('cluster_label', '')}  \n"
            f"Tags: {tags or '—'}"
        )
        if row.get("preview"):
            st.caption(row["preview"])


def main() -> None:
    st.title("Déjà Code")
    st.write(
        "Find CodeChef problems that share an algorithm or data structure, even when the story is different. "
        "Search mixes BGE sentence embeddings with TF-IDF. Topic tags are labels, not search features."
    )
    if not (ART / "embeddings.npy").exists():
        st.error("Index not built yet. Run `python -m src.pipeline` first.")
        return

    retriever = get_retriever()
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    st.caption(
        f"{meta['n_problems']} problems · {meta['model']} · {meta['k']} clusters · "
        f"silhouette {meta['silhouette']:.3f}"
    )

    tab_search, tab_clusters, tab_recommend = st.tabs(["Search", "Clusters", "Practice next"])

    with tab_search:
        query = st.text_area(
            "Problem statement",
            height=180,
            placeholder="Paste a statement, or describe the task in your own words.",
        )
        mode = st.radio("Retrieval", ["hybrid", "semantic", "keyword"], horizontal=True)
        top_k = st.slider("Results", 3, 15, 5)
        alpha = float(meta.get("hybrid_alpha", 0.65))
        if st.button("Find similar problems", type="primary") and query.strip():
            hits = retriever.search(query.strip(), mode=mode, k=top_k, alpha=alpha)
            show_hits(hits)

    with tab_clusters:
        clusters = retriever.clusters
        labels = []
        for cluster_id, info in sorted(clusters.items(), key=lambda item: int(item[0])):
            labels.append(f"{cluster_id}: {info['label']} ({info['size']})")
        choice = st.selectbox("Cluster", labels)
        cluster_id = choice.split(":", 1)[0]
        info = clusters[cluster_id]
        tag_line = ", ".join(f"{item['tag']} ({item['count']})" for item in info["top_tags"][:5]) or "no strong tags"
        st.write(f"Distinctive words: {', '.join(info['distinctive_terms'])}")
        st.write(f"Tags: {tag_line}")
        explained = load_explanations().get(cluster_id)
        if explained:
            st.info(explained["narrative"])
            reps = ", ".join(f"`{item['code']}`" for item in explained["representatives"])
            st.markdown(f"**Most typical members:** {reps}")
        members = retriever.frame[retriever.frame["cluster"] == int(cluster_id)].sort_values("difficulty")
        st.dataframe(
            members[["code", "title", "difficulty", "tag_text", "link"]].head(30),
            width="stretch",
            hide_index=True,
        )
        figure = ART / "figures" / "clusters_pca.png"
        if figure.exists():
            st.image(str(figure), caption="PCA of BGE embeddings")

    with tab_recommend:
        raw = st.text_input("Solved problem codes", placeholder="FCTRL, HS08TEST, FLOW001")
        if st.button("Recommend") and raw.strip():
            codes = [part.strip() for part in raw.replace(";", ",").split(",")]
            try:
                recs = retriever.recommend(codes, k=8)
            except KeyError as exc:
                st.warning(str(exc))
            else:
                for _, row in recs.iterrows():
                    tags = ", ".join(row["tags"]) if isinstance(row["tags"], list) else ""
                    st.markdown(
                        f"**[{row['code']}]({row['link']}) — {row['title']}**  \n"
                        f"Rating {int(row['difficulty'])} · {row['cluster_label']}  \n"
                        f"Tags: {tags or '—'}"
                    )


if __name__ == "__main__":
    main()
