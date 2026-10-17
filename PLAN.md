# Déjà Code — Delivery Plan & Progress Tracker

> Tracking document for the CC-VIT recruitment task "Semantic Similarity Search over
> Competitive Programming Problems". Update the status column and the Progress Log
> as work lands. Re-run `git rev-parse --short HEAD` after each merge and record it below.

- **Repo:** https://github.com/ManasiMaanya/semanticprogram
- **Branch:** `main`
- **Last committed revision:** `1d129bb` (tags `v1.0`; `main` level with `origin`).
- **Snapshot date:** 2026-10-17
- **Overall status:** functional and demonstrable. Every required deliverable and every
  bonus item exists and runs. Phase P0 is complete; what remains is hardening,
  documentation consistency, and evaluation depth — not missing features.

---

## 1. Completion audit (snapshot @ `0e3b113` + uncommitted working tree)

### Required deliverables

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 01 | Explore & preprocess dataset | ✅ Done | `src/preprocess.py`, `reports/exploration.json`, `reports/preprocess_summary.json` (1,367 fetched → 1,316 kept) |
| 02 | Numeric representation (embeddings/NLP) | ✅ Done | TF-IDF (`artifacts/tfidf.joblib`) + BGE-small 384-d (`artifacts/embeddings.npy`) |
| 03 | Semantic search / retrieval | ✅ Done | `Retriever.search()` in `src/engine.py`, modes `hybrid`/`semantic`/`keyword` |
| 04 | Clustering | ✅ Done | K-means on 50-d PCA, `k` chosen by tag purity, `artifacts/clusters.json` |
| 05 | Visualize clusters + explain similarity | 🟡 Partial | 4 figures in `artifacts/figures/`; labels + distinctive terms exist, but no written per-cluster narrative |
| 06 | User submits statement → top matches | ✅ Done | Streamlit `app.py` Search tab + demo CLI |
| 07 | Evaluate with IR metrics | 🟡 Partial | P@5, P@10, R@10, AP@10, nDCG@10, MRR on a held-out half — but relevance is auto/tag-based, no manual gold set |
| 08 | Analyze incorrect matches | ✅ Done | "Near misses" slice in `reports/evaluation.md`, `error_examples` in JSON |

### Bonus items

| Bonus | Status | Evidence |
| --- | --- | --- |
| Keyword vs embedding comparison | ✅ Done | `test_comparison` table (keyword / semantic / hybrid) |
| Hybrid search | ✅ Done | Tuned `alpha` (0.65 by AP@10) |
| Same algorithm, different storyline | ✅ Done | `cross_story_examples` slice |
| Auto cluster labels | ✅ Done | `cluster_labels()` from tags + TF-IDF centroid terms |
| Interactive exploration interface | ✅ Done | Streamlit: Search / Clusters / Practice-next tabs |
| Recommendations by difficulty + concepts | ✅ Done | `Retriever.recommend()` — concept profile + next difficulty band |

**Score:** 8/8 required present (2 need depth), 6/6 bonus present.

### Verified this session

- `.venv` Python 3.12 with all deps importable.
- `python -m unittest discover -s tests -v` → **5 tests, all pass** (artifacts, meta,
  clusters, three search modes, recommendation).
- Artifacts on disk: `embeddings.npy`, `tfidf.joblib`, `problems.jsonl`, `clusters.json`,
  `meta.json`, `coords.npy`, 4 PNG figures.

### Uncommitted work in the tree (must be committed)

Six files modified vs `0e3b113`: `README.md`, `app.py`, `src/engine.py`,
`src/evaluate.py`, `src/preprocess.py`, `src/visualize.py`. These are real improvements
(switch MiniLM → BGE-small, tag-purity `k` selection, narrative-text preprocessing,
richer eval slices). No commits since the initial one. Local `main` is level with
`origin/main`.

---

## 2. Known issues found in the audit

| ID | Issue | Severity |
| --- | --- | --- |
| I-1 | Improved source is **uncommitted** and unpushed | High (data-loss risk) |
| I-2 | `README.md` "What the system gets wrong" still says *MiniLM* though the model is now BGE-small | Medium — ✅ fixed in P1 |
| I-3 | `README.md` Clustering step 1 says threshold **18%**; code uses **12%** | Low — ✅ fixed in P1 |
| I-4 | Required item 05 explanation is thin — no per-cluster "what makes these similar" write-up | Medium |
| I-5 | Evaluation is tag-relevance only; task allows a *manually annotated* set, which would raise confidence | Medium |
| I-6 | No `dev-requirements.txt` / CI; tests are `unittest` and not automated | Low — ✅ `dev-requirements.txt` + `pytest.ini` added; `pytest -q` green (5 passed). CI still open |
| I-7 | Sample is 1,316 problems; larger sample is possible and cheap (resumable download) | Low |

---

## 3. Phase-wise execution plan

Estimates are focused-work hours for one person. `P0`/`P1` are effectively required to
"finish"; `P2`–`P5` are quality/bonus depth that strengthen the submission.

| Phase | Scope | Deliverable | Est. | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| **P0 — Freeze & ship** | Commit the working improvements, push to `origin/main`, tag `v1.0` | Clean `main` with a green test run | 0.5 h | — | ✅ Done (`20c3f70`) |
| **P1 — Doc & consistency** | Fix I-2, I-3; align README thresholds/model names with code; add `dev-requirements.txt` + pytest shim | README matches code; `pytest` runs | 1.0–1.5 h | P0 | ✅ Done |
| **P2 — Evaluation depth** | Manual gold set (~50 problem pairs, 2 annotators or self + rubric), report agreement; add recall@k curve and run-to-run seed variance | `reports/gold_set.*`, new section in `evaluation.md` | 3–4 h | P0 | ⬜ Not started |
| **P3 — Cluster explanation** | Representative problem per cluster + 2–3 sentence "why similar" narrative; embed in README and app Clusters tab | `reports/clusters_explained.md`, app panel | 2–3 h | P0 | ⬜ Not started |
| **P4 — Scale & performance** | Grow sample (2k–5k), re-run pipeline, optional FAISS index + latency note; measure quality vs size | Updated artifacts + short scaling note | 2–4 h | P0 | ⬜ Not started |
| **P5 — Bonus depth & polish** | Embedding-model comparison table (TF-IDF/MiniLM/BGE), explicit error taxonomy, LICENSE, demo screenshots | Comparison section + `docs/` screenshots | 2–3 h | P0, P2 | ⬜ Not started |

**Total remaining:** ≈ **11–18 focused hours**.
**Minimum to call it finished cleanly:** P0 + P1 ≈ **1.5–2 hours**.
**Recommended to be competitive:** through P3 ≈ **6–8 hours**.

---

## 4. Progress log

| Date | Phase | What changed | Revision |
| --- | --- | --- | --- |
| 2026-10-17 | Audit | Cloned/linked repo, ran tests (5/5 pass), audited deliverables, found I-1…I-7, wrote this plan | `0e3b113` + dirty tree |
| 2026-10-17 | P0 | Committed the six improved source/doc files plus `tests/`, `reports/`, and non-ignored `artifacts/` + `data/processed/`; pushed `main`; tagged `v1.0`. Tree clean, `origin/main` level. | `20c3f70`, tag `v1.0` |
| 2026-10-17 | P1 | Fixed I-2 (MiniLM → BGE-small in "what the system gets wrong"), I-3 (clustering label threshold rewritten to match the 12% code path). Added `dev-requirements.txt` (pytest) and `pytest.ini`; documented `pytest -q` in README. | `1d129bb` |

---

## 5. Working agreements / next action

1. **Next action:** Phase P2 — build a small manually annotated gold set (~50 pairs)
   to complement the tag-based relevance, and add a recall@k curve plus seed-variance
   check to `reports/evaluation.md`.
2. Keep `PLAN.md`, `README.md`, and the code thresholds in sync — the 12%/20%/"always
   generic" tag rules are the most error-prone spot.
3. Any change to preprocessing or the model invalidates `artifacts/`; re-run
   `python -m src.pipeline` and re-record numbers in `reports/evaluation.md`.
4. Record every landed change in the Progress Log above with the new short revision.
