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

    def test_recommendation_engine(self):
        """Verify recommendation engine suggests valid unsolved problems."""
        recs = self.retriever.recommend(["FCTRL"], k=5)
        self.assertEqual(len(recs), 5)
        self.assertNotIn("FCTRL", recs["code"].values)
        self.assertIn("difficulty", recs.columns)
        self.assertIn("cluster_label", recs.columns)


if __name__ == "__main__":
    unittest.main()
