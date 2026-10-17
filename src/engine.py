"""TF-IDF, sentence embeddings, hybrid retrieval, and clustering."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "problems.jsonl"
ART = ROOT / "artifacts"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Tags that describe "a programming problem" rather than a technique.
GENERIC_TAGS = {
    "algorithms",
    "mathematics",
    "math",
    "basic math",
    "simple math",
    "data structures",
    "basic programming concepts",
    "observation",
    "ad hoc",
    "constructive",
    "advanced algorithms",
    "advanced data structures",
    "brute force",
}


def load_processed(path: Path = PROCESSED) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame["tags"] = frame["tags"].apply(lambda value: list(value or []))
    return frame.reset_index(drop=True)


def fit_tfidf(texts: list[str]) -> TfidfVectorizer:
    # Sublinear TF and bigrams catch phrases like "binary search". Contest
    # boilerplate ("chef", "test case") is stopped so it cannot rank a hit.
    stops = list(
        ENGLISH_STOP_WORDS.union(
            {
                "chef",
                "chefs",
                "codechef",
                "test",
                "case",
                "cases",
                "integer",
                "integers",
                "line",
                "lines",
                "input",
                "output",
                "print",
                "single",
                "given",
                "task",
            }
        )
    )
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stops,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=25000,
        sublinear_tf=True,
    )
    vectorizer.fit(texts)
    return vectorizer


def encode_sentences(texts: list[str], batch_size: int = 32) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def lexical_overlap(query: str, document: str) -> float:
    q = set(re_tokens(query))
    d = set(re_tokens(document))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def re_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]{3,}", text.lower())


def specific_tag_sets(frame: pd.DataFrame) -> list[set[str]]:
    sets = []
    for tags in frame["tags"]:
        sets.append({tag.lower() for tag in tags if tag.lower() not in GENERIC_TAGS})
    return sets


def tag_purity(labels: np.ndarray, tag_sets: list[set[str]]) -> float:
    scores = []
    for cluster_id in set(labels.tolist()):
        members = [tag_sets[i] for i, label in enumerate(labels) if label == cluster_id and tag_sets[i]]
        if len(members) < 5:
            continue
        counts: dict[str, int] = {}
        for tags in members:
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        if not counts:
            continue
        scores.append(max(counts.values()) / len(members))
    return float(np.mean(scores)) if scores else 0.0


def choose_k(embeddings: np.ndarray, tag_sets: list[set[str]], random_state: int = 17) -> tuple[int, dict]:
    """Pick k by concept purity, and keep the silhouette of that choice.

    Silhouette on these embeddings stays low because every statement shares a
    contest style. Purity asks whether a cluster concentrates a real topic tag.
    """
    best: dict | None = None
    for k in range(8, 19, 2):
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(embeddings)
        sample_index = np.arange(len(embeddings))
        if len(embeddings) > 800:
            sample_index = np.random.default_rng(random_state).choice(len(embeddings), 800, replace=False)
        silhouette = float(silhouette_score(embeddings[sample_index], labels[sample_index], metric="euclidean"))
        purity = tag_purity(labels, tag_sets)
        row = {"k": k, "silhouette": silhouette, "tag_purity": purity, "labels": labels}
        if best is None or (purity, silhouette) > (best["tag_purity"], best["silhouette"]):
            best = row
    assert best is not None
    return int(best["k"]), {key: value for key, value in best.items() if key != "labels"} | {"labels": best["labels"]}


def cluster_labels(frame: pd.DataFrame, labels: np.ndarray, tfidf_matrix, vectorizer) -> dict:
    """Name each cluster from topic tags, then from distinctive TF-IDF terms."""
    names = {}
    feature_names = np.array(vectorizer.get_feature_names_out())
    for cluster_id in sorted(set(labels.tolist())):
        member = labels == cluster_id
        counts: dict[str, int] = {}
        for tags in frame.loc[member, "tags"]:
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        ranked_tags = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        size = int(member.sum())
        specific = [(tag, count) for tag, count in ranked_tags if tag.lower() not in GENERIC_TAGS]
        # A tag labels the cluster only when it covers a meaningful share of members.
        strong = [tag for tag, count in specific if count >= max(3, int(0.12 * size))][:3]
        if not strong:
            strong = [tag for tag, _ in specific[:2]]
        centroid = np.asarray(tfidf_matrix[member].mean(axis=0)).ravel()
        top_idx = centroid.argsort()[::-1][:6]
        terms = [feature_names[i] for i in top_idx if centroid[i] > 0][:4]
        title = ", ".join(strong) if strong else ", ".join(terms[:3]) or f"Cluster {cluster_id}"
        names[int(cluster_id)] = {
            "label": title,
            "size": size,
            "top_tags": [{"tag": tag, "count": count} for tag, count in ranked_tags[:6]],
            "distinctive_terms": terms,
        }
    return names


def build(frame: pd.DataFrame | None = None) -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    frame = frame if frame is not None else load_processed()
    texts = frame["text"].tolist()
    print(f"fitting TF-IDF on {len(texts)} problems", flush=True)
    vectorizer = fit_tfidf(texts)
    tfidf = vectorizer.transform(texts)
    print("encoding sentence embeddings", flush=True)
    embeddings = encode_sentences(texts)
    reduced = normalize(PCA(n_components=50, random_state=17).fit_transform(embeddings))
    k, selection = choose_k(reduced, specific_tag_sets(frame))
    labels = selection.pop("labels")
    print(
        f"k-means k={k} silhouette={selection['silhouette']:.3f} purity={selection['tag_purity']:.3f}",
        flush=True,
    )
    names = cluster_labels(frame, labels, tfidf, vectorizer)
    pca = PCA(n_components=2, random_state=17)
    coords = pca.fit_transform(embeddings)

    frame = frame.copy()
    frame["cluster"] = labels
    frame["cluster_label"] = [names[int(i)]["label"] for i in labels]
    joblib.dump(vectorizer, ART / "tfidf.joblib")
    np.save(ART / "embeddings.npy", embeddings)
    np.save(ART / "coords.npy", coords.astype(np.float32))
    (ART / "clusters.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    (ART / "meta.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "n_problems": int(len(frame)),
                "k": int(k),
                "silhouette": selection["silhouette"],
                "tag_purity": selection["tag_purity"],
                "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (ART / "problems.jsonl").open("w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "frame": frame,
        "vectorizer": vectorizer,
        "tfidf": tfidf,
        "embeddings": embeddings,
        "coords": coords,
        "clusters": names,
    }


class Retriever:
    def __init__(self) -> None:
        self.frame = load_processed(ART / "problems.jsonl")
        self.vectorizer: TfidfVectorizer = joblib.load(ART / "tfidf.joblib")
        self.embeddings = np.load(ART / "embeddings.npy")
        self.tfidf = normalize(self.vectorizer.transform(self.frame["text"].tolist()))
        self.clusters = json.loads((ART / "clusters.json").read_text(encoding="utf-8"))
        self._model = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
        prefixed = [QUERY_PREFIX + text for text in texts]
        return self._model.encode(
            prefixed, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def score(self, query: str) -> dict[str, np.ndarray]:
        semantic = self._encode([query]) @ self.embeddings.T
        keyword = cosine_similarity(normalize(self.vectorizer.transform([query])), self.tfidf).ravel()
        semantic = np.asarray(semantic).ravel()
        return {"semantic": semantic, "keyword": keyword}

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        k: int = 8,
        alpha: float = 0.65,
        exclude_code: str | None = None,
    ) -> pd.DataFrame:
        parts = self.score(query)
        if mode == "semantic":
            scores = parts["semantic"]
        elif mode == "keyword":
            scores = parts["keyword"]
        else:
            scores = alpha * parts["semantic"] + (1.0 - alpha) * parts["keyword"]
        order = np.argsort(-scores)
        rows = []
        for index in order:
            code = self.frame.at[index, "code"]
            if exclude_code and code == exclude_code:
                continue
            row = self.frame.iloc[int(index)]
            rows.append(
                {
                    "code": row["code"],
                    "title": row["title"],
                    "difficulty": int(row["difficulty"]),
                    "tags": row["tags"],
                    "cluster": int(row["cluster"]),
                    "cluster_label": row["cluster_label"],
                    "score": float(scores[index]),
                    "semantic": float(parts["semantic"][index]),
                    "keyword": float(parts["keyword"][index]),
                    "link": row["link"],
                    "preview": row["text"][:320].replace("\n", " "),
                }
            )
            if len(rows) >= k:
                break
        return pd.DataFrame(rows)

    def recommend(self, solved_codes: list[str], k: int = 8) -> pd.DataFrame:
        """Suggest unsolved problems that share concepts and sit slightly above the user's level."""
        codes = [code.strip().upper() for code in solved_codes if code.strip()]
        index = {code: i for i, code in enumerate(self.frame["code"])}
        known = [index[code] for code in codes if code in index]
        if not known:
            raise KeyError("none of the solved codes are in the indexed collection")
        profile = self.embeddings[known].mean(axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-9)
        solved_rating = float(np.median(self.frame.iloc[known]["difficulty"]))
        semantic = self.embeddings @ profile
        # Prefer the next difficulty band. Problems far easier or much harder are down-weighted.
        difficulty = self.frame["difficulty"].to_numpy(dtype=float)
        gap = difficulty - solved_rating
        band = np.exp(-0.5 * ((gap - 150) / 250) ** 2)
        scores = 0.75 * semantic + 0.25 * band
        for i in known:
            scores[i] = -1
        order = np.argsort(-scores)[:k]
        rows = []
        for i in order:
            row = self.frame.iloc[int(i)]
            rows.append(
                {
                    "code": row["code"],
                    "title": row["title"],
                    "difficulty": int(row["difficulty"]),
                    "tags": row["tags"],
                    "cluster_label": row["cluster_label"],
                    "score": float(scores[i]),
                    "link": row["link"],
                }
            )
        return pd.DataFrame(rows)
