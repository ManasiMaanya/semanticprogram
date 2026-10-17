# Déjà Code

Semantic search over CodeChef problems. The index retrieves questions that share an algorithm or data structure even when the story, nouns, and variable names are different.

## Data

Problems come from CodeChef's public problem API (`/api/list/problems` and `/api/contests/{contest}/problems/{code}`), not from the 2016 Kaggle dump. That dump is mostly submitted source code. This task needs statements and topic tags.

`src/download.py` keeps every rated problem (rating 1–5000; 5,552 problems in the catalog) in `data/raw/catalog.json`, then fetches a stratified sample of 1,400 problems spread across rating bands. 1,367 statements came back; cleaning kept 1,316. Each record stores the statement, topic tags, difficulty, and a link. Tags are evaluation labels. They are not retrieval features.

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

**Semantic index.** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), a 384-dimensional retrieval embedding. A first pass with `all-MiniLM-L6-v2` lost to TF-IDF: it matched the contest voice ("Chef has an array") more than the task. BGE-small is trained for passage retrieval and is still small enough to encode this collection on CPU. Corpus vectors are L2-normalized. A user query is prefixed with "Represent this sentence for searching relevant passages:", which is the prefix BGE was trained with. Problem-to-problem evaluation does not use that prefix, so both sides of a pair are encoded the same way.

**Hybrid score.** `alpha * semantic + (1 - alpha) * keyword`. `alpha` is selected on a tuning split (see Evaluation). Keyword similarity still helps when the query uses a standard name ("Dijkstra", "segment tree") that the embedding underweights.

Similarity search is exact cosine over the collection. FAISS would matter past tens of thousands of vectors; at this size a matrix multiply is simpler and exact.

## Clustering

K-means runs on a 50-dimensional PCA of the embeddings. In the raw 384 dimensions every statement sits near every other statement, and silhouette stays near zero. `k` is chosen from {8, 10, …, 18} by how tightly a cluster concentrates a specific topic tag (tag purity), with silhouette as a tie-break.

Each cluster is labeled automatically:

1. Topic tags that cover at least 12% of the cluster (and at least 3 problems), up to three. Tags that only mean "this is a programming problem" (`Algorithms`, `Mathematics`, `Data Structures`, and similar) are ignored.
2. If no tag clears that bar, the two most common specific tags in the cluster.
3. If the cluster has no specific tag at all, its highest-weighted TF-IDF centroid terms.

The stored `distinctive_terms` are always the top TF-IDF centroid words, so every cluster keeps a description even when its tags are thin.

PCA projects the embeddings to two dimensions for `artifacts/figures/clusters_pca.png`. The first two components are a map, not a claim that two axes explain the space. The explained-variance ratio is stored in `artifacts/meta.json`.

`src/explain.py` writes `reports/clusters_explained.md` and `.json`. For each cluster it reports the specific tags and how much of the cluster they cover (spelling variants are merged), the distinctive words, the difficulty range, and the three problems closest to the cluster centroid — the most typical members. The narrative is generated from the cluster's own members, so it cannot drift from the artifacts, and the same text appears in the app's Clusters tab.

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

`Algorithms`, `Mathematics`, and `Data Structures` are always ignored. Any other tag on more than 20% of tagged problems is ignored too. Dynamic programming stays, because it is a technique rather than a container label. Only problems with at least three relevant neighbors are queries. Half tune `alpha` by AP@10. The other half is the reported comparison of keyword, semantic, and hybrid search: precision@5, precision@10, recall@10, AP@10, nDCG@10, and MRR. Recall@10 is small on purpose: a typical query has well over a hundred tagged cousins, and the metric only looks at ten results. Numbers are written to `reports/evaluation.md` by the pipeline.

On the held-out half the hybrid index is the best of the three. With semantic weight 0.65 it reaches P@5 0.289, P@10 0.269, AP@10 0.162, nDCG@10 0.277, and MRR 0.464, ahead of TF-IDF alone (P@10 0.259, MRR 0.456) and ahead of embeddings alone (P@10 0.246, MRR 0.447). Embeddings do not replace keywords. They add the pairs keywords miss, which is why the mixture wins.

Sixteen clusters fall out of the embeddings. Their automatic labels include dynamic programming, graphs and trees, bit manipulation, number theory, strings, game theory, and combinatorics. Silhouette is about 0.07. Statements still share a contest style, so the clusters overlap, but the labels are techniques rather than the word "Algorithms".

The index also links problems whose wording barely overlaps. `DIVQUERY` (divisibility queries) sits near `PARPERM` (a partition problem) with token Jaccard 0.05 and a shared number-theory tag. `AMXOR` (XOR sum) sits near `XOR2` (consecutive XOR) on bit manipulation, Jaccard 0.09. The failure mode is the mirror image: `GOOD1` and `GOOD2`, or `LUCKY4` and `LUCKY5`, are nearest neighbors because the stories match, while their tag sets do not. Some of those "errors" are easy/hard twins with inconsistent tags. Others really are different tasks wearing the same nouns.

This is a weak label, not a human judgment. Tag vocabulary is inconsistent, some real cousins share no tag, and some problems have no tags. Those cases are called out in the error slice rather than counted as successes.

Two extra slices:

- **Same concept, different story.** Nearest hybrid neighbor, at least one shared specific tag, token Jaccard below 0.08.
- **Near misses.** Nearest neighbor has tags, but none of them match. High lexical overlap is "shared wording, different concept". Low overlap is "similar narrative frame" (the embedding latched onto the Chef-and-an-array template).

`reports/demo_queries.json` stores six statements written for this project (longest increasing subsequence, disguised shortest paths, binary search on the answer, range updates, modular exponentiation, connected components) and the problems the hybrid index returned.

## Manual evaluation set

The tag-based numbers cannot score a statement that is not in the collection. `src/gold.py` adds a small hand-labelled set: ten statements written for this project — the six demo prompts plus 0/1 knapsack, maximum subarray, topological order, and a prime count. For each statement the top five results of keyword, semantic, and hybrid search are pooled, so every mode's top five is judged, and each candidate is labelled by hand as the same technique (1) or not (0). Ambiguous candidates count as 0.

Ninety candidates were judged; thirteen were relevant. On this set hybrid leads precision@5 (0.22) while semantic leads MRR (0.45). Ten queries is a small sample, so read the gaps as directional. The failures are the useful part: the knapsack statement returns problems about buying and bags, the painter-scheduling statement returns problems about painting, and the maximum-subarray statement returns problems about profit. The index follows the words even when the technique is different. Results are in `reports/gold_eval.md`.

## What the system gets wrong

BGE-small was not trained on competitive-programming jargon, so it can tie together problems that share a setup ("Chef has N integers and T test cases") and miss a rare technique that is only named in a tag. TF-IDF does the opposite: it misses a shortest-path problem that never says "graph". Hybrid search is the compromise, and it still fails when both signals are weak. Clustering inherits the same geometry, so a cluster label is a summary of its members, not a proof that every member uses one algorithm.

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

Run the tests with `pip install -r dev-requirements.txt` then `pytest -q` (or `python -m unittest discover -s tests`). They check that the index artifacts exist, that `meta.json` and `clusters.json` are well formed, and that search and recommendation return results in all three retrieval modes.

## Layout

| Path | Role |
| --- | --- |
| `src/download.py` | API catalog and stratified sample |
| `src/preprocess.py` | Cleaning and tag filtering |
| `src/engine.py` | TF-IDF, BGE embeddings, hybrid search, K-means, recommendations |
| `src/evaluate.py` | Metrics, error slices, demo queries |
| `src/gold.py` | Hand-labelled query set and its scoring |
| `src/explain.py` | Per-cluster explanations and representative problems |
| `src/visualize.py` | PCA map, cluster sizes, difficulty, tag chart |
| `src/pipeline.py` | End-to-end run |
| `app.py` | Streamlit explorer |
| `reports/` | Exploration, metrics, demo retrievals |
| `artifacts/` | Vectors, vectorizer, cluster names, figures |
