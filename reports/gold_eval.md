# Manual evaluation set

Tag-based relevance cannot score a statement that is not in the collection, so ten statements were written for this project and the top results of all three modes were pooled and judged by hand. A candidate is relevant (1) when it uses the same underlying technique as the intent, regardless of wording. Every mode's top-k is inside the pool, so no mode is scored against unjudged results.

- Queries: 10
- Pooled candidates judged: 90 (13 relevant, 14%)
- Hybrid weight: 0.65

## Precision@5, nDCG@5, MRR on the manual labels

| Method | P@5 | nDCG@5 | MRR |
| --- | --- | --- | --- |
| keyword | 0.200 | 0.364 | 0.350 |
| semantic | 0.140 | 0.469 | 0.450 |
| hybrid | 0.220 | 0.395 | 0.345 |

On these ten hand-written statements, **hybrid** leads precision@5 and **semantic** leads MRR. Ten queries is a small sample, so read the gaps as directional, not significant. The interesting failures are the queries with zero relevant hits: the words *painting*, *bag*, *profit* and *task* pull back problems about painting, buying, profit and scheduling rather than the technique asked for.

## Per query

Relevant counts manually relevant problems in the whole pool, not just the top five.

| Query | Intent | Pooled | Relevant | Top-1 (hybrid) | Hit |
| --- | --- | --- | --- | --- | --- |
| lis | longest increasing subsequence | 9 | 1 | `MMP` | no |
| dijkstra | single-source shortest paths | 11 | 1 | `SLMNTRV` | no |
| binary-search-answer | binary search on the answer | 9 | 0 | `PCJ18F` | no |
| range-update-query | range updates and range sum queries | 12 | 3 | `TWOCLOSE` | no |
| modpow | modular exponentiation | 9 | 2 | `SEQTOWER` | yes |
| connected-components | connected components / union-find | 7 | 3 | `MSTONES` | no |
| knapsack | 0/1 knapsack dynamic programming | 9 | 0 | `AIRLINE` | no |
| max-subarray | maximum subarray sum | 7 | 0 | `CHFPROFIT` | no |
| topological-order | topological sort of a DAG | 9 | 0 | `CLEANUP` | no |
| count-primes | sieve of Eratosthenes | 8 | 3 | `PPERM` | yes |
