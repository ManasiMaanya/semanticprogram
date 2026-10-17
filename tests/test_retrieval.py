"""Unit tests for Déjà Code semantic retrieval and clustering system."""

import json
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


class TestDejaCodePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.engine import Retriever, load_processed
        cls.retriever = Retriever()
        cls.problems_path = ART / "problems.jsonl"
        cls.clusters_path = ART / "clusters.json"
        cls.meta_path = ART / "meta.json"

    def test_artifacts_exist(self):
        """Verify all essential index and artifact files exist."""
        self.assertTrue(self.problems_path.exists(), "problems.jsonl missing")
        self.assertTrue((ART / "embeddings.npy").exists(), "embeddings.npy missing")
        self.assertTrue((ART / "tfidf.joblib").exists(), "tfidf.joblib missing")
        self.assertTrue(self.clusters_path.exists(), "clusters.json missing")
        self.assertTrue(self.meta_path.exists(), "meta.json missing")

    def test_meta_integrity(self):
        """Verify meta.json contains valid metadata."""
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.assertIn("model", meta)
        self.assertIn("n_problems", meta)
        self.assertIn("k", meta)
        self.assertIn("silhouette", meta)
        self.assertGreater(meta["n_problems"], 1000)
        self.assertGreaterEqual(meta["k"], 8)

    def test_clusters_integrity(self):
        """Verify clusters.json contains properly structured cluster metadata."""
        clusters = json.loads(self.clusters_path.read_text(encoding="utf-8"))
        self.assertGreater(len(clusters), 0)
        for cid, info in clusters.items():
            self.assertIn("label", info)
            self.assertIn("size", info)
            self.assertIn("top_tags", info)
            self.assertIn("distinctive_terms", info)
            self.assertGreater(info["size"], 0)

    def test_search_modes(self):
        """Verify search executes across hybrid, semantic, and keyword modes."""
        query = "find shortest path between cities using Dijkstra graph algorithm"
        for mode in ["hybrid", "semantic", "keyword"]:
            results = self.retriever.search(query, mode=mode, k=5)
            self.assertEqual(len(results), 5)
            self.assertIn("code", results.columns)
            self.assertIn("score", results.columns)
            self.assertIn("cluster_label", results.columns)
            self.assertTrue(all(results["score"] >= 0))

    def test_cluster_explanations(self):
        """Verify the generated per-cluster explanations are present and well formed."""
        path = ROOT / "reports" / "clusters_explained.json"
        self.assertTrue(path.exists(), "clusters_explained.json missing")
        explained = json.loads(path.read_text(encoding="utf-8"))
        clusters = json.loads(self.clusters_path.read_text(encoding="utf-8"))
        self.assertEqual(len(explained), len(clusters))
        for info in explained.values():
            self.assertTrue(info["narrative"].strip())
            self.assertGreaterEqual(len(info["representatives"]), 1)

    def test_gold_set_labelled(self):
        """Verify the manual gold set is fully labelled and scored."""
        gold_path = ROOT / "reports" / "gold_set.json"
        eval_path = ROOT / "reports" / "gold_eval.json"
        self.assertTrue(gold_path.exists(), "gold_set.json missing")
        self.assertTrue(eval_path.exists(), "gold_eval.json missing")
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        for query in gold["queries"]:
            for candidate in query["pooled"]:
                self.assertIsNotNone(candidate["label"], f"{query['id']}/{candidate['code']} unlabelled")
        summary = json.loads(eval_path.read_text(encoding="utf-8"))
        for mode in ("keyword", "semantic", "hybrid"):
            self.assertIn("precision@5", summary["per_mode"][mode])

    def test_recommendation_engine(self):
        """Verify recommendation engine suggests valid unsolved problems."""
        recs = self.retriever.recommend(["FCTRL"], k=5)
        self.assertEqual(len(recs), 5)
        self.assertNotIn("FCTRL", recs["code"].values)
        self.assertIn("difficulty", recs.columns)
        self.assertIn("cluster_label", recs.columns)


if __name__ == "__main__":
    unittest.main()
