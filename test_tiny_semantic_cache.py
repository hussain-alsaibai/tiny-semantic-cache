"""Tests for tiny-semantic-cache."""

import time
import unittest
from tiny_semantic_cache import (
    SemanticCache,
    cosine_similarity,
    euclidean_similarity,
    CacheEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def mock_embed(text: str) -> list[float]:
    """Trivial deterministic embedder for testing: hash to 4 dims."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:4]]


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=5)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0, places=5)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])


class TestEuclideanSimilarity(unittest.TestCase):
    def test_same_vector(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(euclidean_similarity(v, v), 1.0, places=5)

    def test_positive_range(self):
        v1, v2 = [0.0] * 4, [1.0] * 4
        self.assertGreater(euclidean_similarity(v1, v2), 0.0)
        self.assertLessEqual(euclidean_similarity(v1, v2), 1.0)


class TestSemanticCache(unittest.TestCase):
    def setUp(self):
        self.cache = SemanticCache(
            embedding_fn=mock_embed,
            similarity_threshold=0.90,
            max_size=10,
        )

    def test_set_and_get_exact(self):
        self.cache.set("What is Rust?", "A systems language.")
        result = self.cache.get("What is Rust?")
        self.assertEqual(result, "A systems language.")

    def test_get_or_call_cache_miss(self):
        calls = []
        def llm(prompt):
            calls.append(prompt)
            return f"Response to: {prompt}"

        result = self.cache.get_or_call("Hello world", llm)
        self.assertEqual(result, "Response to: Hello world")
        self.assertEqual(len(calls), 1)

    def test_get_or_call_cache_hit(self):
        calls = []
        def llm(prompt):
            calls.append(prompt)
            return f"Response to: {prompt}"

        self.cache.get_or_call("Hello world", llm)
        self.assertEqual(len(calls), 1)

        result = self.cache.get_or_call("Hello world", llm)
        self.assertEqual(result, "Response to: Hello world")
        self.assertEqual(len(calls), 1)  # no second call

    def test_invalidate(self):
        self.cache.set("key1", "val1")
        self.assertTrue(self.cache.invalidate("key1"))
        self.assertIsNone(self.cache.get("key1"))

    def test_invalidate_similar(self):
        self.cache.set("How do neural networks learn?", "Via backpropagation.")
        self.cache.set("What is Python?", "A programming language.")
        removed = self.cache.invalidate_similar(
            "How do NNs train?", threshold=0.90
        )
        self.assertGreaterEqual(removed, 1)

    def test_ttl_eviction(self):
        cache = SemanticCache(
            embedding_fn=mock_embed,
            similarity_threshold=0.90,
            ttl_seconds=0.1,
        )
        cache.set("key", "value")
        time.sleep(0.15)
        self.assertIsNone(cache.get("key"))

    def test_max_size_eviction(self):
        cache = SemanticCache(
            embedding_fn=mock_embed,
            similarity_threshold=0.90,
            max_size=3,
        )
        for i in range(5):
            cache.set(f"prompt {i}", f"response {i}")
        # Should have at most 3 entries
        self.assertLessEqual(cache.stats()["entries"], 3)

    def test_stats(self):
        self.cache.set("a", "b")
        self.cache.get("a")  # hit
        self.cache.get("missing")  # miss
        stats = self.cache.stats()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)

    def test_top_similar(self):
        self.cache.set("Python is great", "Yes it is.")
        self.cache.set("Rust is fast", "Very fast.")
        results = self.cache.top_similar("Tell me about Python", n=2)
        self.assertLessEqual(len(results), 2)
        # Most similar should be the Python entry
        if results:
            self.assertIn("Python", results[0]["prompt"])

    def test_clear(self):
        self.cache.set("a", "b")
        self.cache.set("c", "d")
        self.cache.clear()
        self.assertEqual(cache.stats()["entries"], 0)

    def test_error_missing_embedder(self):
        with self.assertRaises(ValueError):
            SemanticCache(embedding_fn=None)

    def test_error_invalid_threshold(self):
        with self.assertRaises(ValueError):
            SemanticCache(embedding_fn=mock_embed, similarity_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
