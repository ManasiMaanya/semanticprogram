# Déjà Code

Semantic search over CodeChef problems. The index retrieves questions that share an algorithm or data structure even when the story, nouns, and variable names are different.

## Data

Problems come from CodeChef's public problem API (`/api/list/problems` and `/api/contests/{contest}/problems/{code}`), not from the 2016 Kaggle dump. That dump is mostly submitted source code. This task needs statements and topic tags.

`src/download.py` keeps every rated problem (rating 1–5000; about 5,500 problems as of the download) in `data/raw/catalog.json`, then fetches a stratified sample of 1,400 problems spread across rating bands. Each record stores the statement, constraints, topic tags, difficulty, and a link. Tags are evaluation labels. They are not retrieval features.

Refresh the sample with:

```bash
python src/download.py
```

The downloader resumes from `data/raw/problems.jsonl`.

## Preprocessing

`src/preprocess.py` turns each problem into one document:

- HTML and LaTeX markers are stripped.
- Boilerplate such as "All submissions for this problem are available" and the unpublished-statement template are removed.
- The embedded text is the title, statement, input format, output format, and constraints. Sample explanations and editorials are left out so the model cannot read the intended algorithm off the editorial.
- Problems with fewer than 40 words, or the CodeChef statement template, are dropped.
- Author handles (lowercase tokens such as `abhay_1006`) are removed from the tag list. Concept tags (`Graphs`, `Kruskal's Algorithm`) are kept.

## Representations

Two indexes are built on the same cleaned text.

**Keyword baseline.** TF-IDF unigrams and bigrams (`min_df=2`, `max_df=0.85`, sublinear tf, 25,000 features). This is the traditional search the task asks to beat. It matches shared words such as "array" and "chef", which show up in unrelated problems.

**Semantic index.** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a 384-dimensional Sentence-BERT model. It was chosen over a full BERT encoder because it is trained for cosine similarity between sentences, runs on CPU for this collection size, and does not need fine-tuning on CodeChef. Vectors are L2-normalized, so inner product is cosine similarity.

**Hybrid score.** `alpha * semantic + (1 - alpha) * keyword`. `alpha` is selected on a tuning split (see Evaluation). Keyword similarity still helps when the query uses a standard name ("Dijkstra", "segment tree") that the embedding underweights.

Similarity search is exact cosine over the collection. FAISS would matter past tens of thousands of vectors; at this size a matrix multiply is simpler and exact.

## Clustering

K-means on the MiniLM vectors. `k` is chosen from {6, 8, …, 18} by silhouette score on a sample of the embeddings (Euclidean distance on normalized vectors ranks neighbors the same way cosine does).

Each cluster is labeled automatically:

1. Topic tags that cover at least 18% of the cluster (and at least 3 problems).
2. If no tag is that common, the highest-weighted TF-IDF terms in the cluster centroid.

PCA projects the embeddings to two dimensions for `artifacts/figures/clusters_pca.png`. The first two components are a map, not a claim that two axes explain the space. The explained-variance ratio is stored in `artifacts/meta.json`.

## Retrieval API

```python
from src.engine import Retriever

retriever = Retriever()
retriever.search("find the shortest travel time from the capital to every city", mode="hybrid", k=5)
retriever.recommend(["FCTRL", "FLOW001"], k=8)
```

`recommend` builds a profile from the embeddings of solved problems and prefers unsolved problems about one difficulty band higher (a Gaussian centered 150 rating points above the median of the solved set).

## Evaluation

Topic tags never enter the index. After aliasing obvious duplicates (`Graph Algos` → `graphs`, `Dynamic Programming` → `dp`), any tag attached to more than 15% of tagged problems is treated as generic and ignored. A retrieved problem is relevant when it shares at least one remaining tag with the query. The query itself is excluded.

Only problems with at least three relevant neighbors are queries. Half tune `alpha` by AP@10. The other half is the reported comparison of keyword, semantic, and hybrid search: precision@10, recall@10, AP@10, nDCG@10, and MRR. Numbers are written to `reports/evaluation.md` by the pipeline.

This is a weak label, not a human judgment. Tag vocabulary is inconsistent, some real cousins share no tag, and some problems have no tags. Those cases are called out in the error slice rather than counted as successes.

Two extra slices:

- **Same concept, different story.** Nearest hybrid neighbor, at least one shared specific tag, token Jaccard below 0.08.
- **Near misses.** Nearest neighbor has tags, but none of them match. High lexical overlap is "shared wording, different concept". Low overlap is "similar narrative frame" (the embedding latched onto the Chef-and-an-array template).

`reports/demo_queries.json` stores six statements written for this project (longest increasing subsequence, disguised shortest paths, binary search on the answer, range updates, modular exponentiation, connected components) and the problems the hybrid index returned.

## What the system gets wrong

MiniLM was not trained on competitive-programming jargon, so it can tie together problems that share a setup ("Chef has N integers and T test cases") and miss a rare technique that is only named in a tag. TF-IDF does the opposite: it misses a shortest-path problem that never says "graph". Hybrid search is the compromise, and it still fails when both signals are weak. Clustering inherits the same geometry, so a cluster label is a summary of its members, not a proof that every member uses one algorithm.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/download.py
python -m src.pipeline
streamlit run app.py
```

`python -m src.pipeline` cleans the download, fits both indexes, selects `k`, writes figures, and runs evaluation. The Streamlit app searches, browses clusters, and recommends a next problem from solved codes.

## Layout

| Path | Role |
| --- | --- |
| `src/download.py` | API catalog and stratified sample |
| `src/preprocess.py` | Cleaning and tag filtering |
| `src/engine.py` | TF-IDF, MiniLM, hybrid search, K-means, recommendations |
| `src/evaluate.py` | Metrics, error slices, demo queries |
| `src/visualize.py` | PCA map, cluster sizes, difficulty, tag chart |
| `src/pipeline.py` | End-to-end run |
| `app.py` | Streamlit explorer |
| `reports/` | Exploration, metrics, demo retrievals |
| `artifacts/` | Vectors, vectorizer, cluster names, figures |
