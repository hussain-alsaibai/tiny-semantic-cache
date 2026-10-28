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


_KEYWORD_VOCAB = (
    "python", "rust", "neural", "networks", "learn", "train",
    "do", "how", "nns", "what", "is", "about", "tell", "me",
    "great", "fast", "very", "programming", "language",
)

# "NNs" and "neural networks" are the same concept; a bag-of-words
# embedder would otherwise treat them as unrelated.
_SYNONYMS = {"nns": ("neural", "networks")}


def _keyword_embed(text: str) -> list[float]:
    """Bag-of-words embedder: cosine similarity over word overlap.

    Unlike :func:`mock_embed`, this gives semantically related prompts a
    high similarity score, which is what the cascade and ordering tests
    need in order to mean anything.
    """
    import re

    words = re.findall(r"[a-z]+", text.lower())
    vector = [0.0] * len(_KEYWORD_VOCAB)
    for word in words:
        if word in _SYNONYMS:
            for synonym in _SYNONYMS[word]:
                vector[_KEYWORD_VOCAB.index(synonym)] = 1.0
        elif word in _KEYWORD_VOCAB:
            vector[_KEYWORD_VOCAB.index(word)] = 1.0
    return vector


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
        def llm():
            calls.append("Hello world")
            return "Response to: Hello world"

        result = self.cache.get_or_call("Hello world", llm)
        self.assertEqual(result, "Response to: Hello world")
        self.assertEqual(len(calls), 1)

    def test_get_or_call_cache_hit(self):
        calls = []
        def llm():
            calls.append("Hello world")
            return "Response to: Hello world"

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
        """invalidate_similar() removes every entry above the threshold.

        The hash embedder has no notion of semantic equivalence --
        "How do NNs train?" scores 0.6669 against the cached entry, well
        below the threshold. A token-overlap embedder is used instead so
        the cascade behaviour is actually exercised.
        """
        cache = SemanticCache(
            embedding_fn=_keyword_embed,
            similarity_threshold=0.50,
            max_size=10,
        )
        cache.set("How do neural networks learn?", "Via backpropagation.")
        cache.set("What is Python?", "A programming language.")
        removed = cache.invalidate_similar("How do NNs train?", threshold=0.50)
        self.assertGreaterEqual(removed, 1)
        self.assertIsNone(cache.get("How do neural networks learn?"))

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
        self.assertLessEqual(self.cache.stats()["entries"], 3)

    def test_stats(self):
        self.cache.set("a", "b")
        self.cache.get("a")  # exact hit
        stats = self.cache.stats()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 0)
        self.assertEqual(stats["exact_hits"], 1)
        self.assertEqual(stats["hit_rate"], 1.0)

    def test_stats_counts_a_real_miss(self):
        """A lookup that matches nothing is a miss.

        The embedder is a hash, so "missing" happens to score 0.90 against
        the single cached entry "a" -- which is at the fixture threshold.
        An empty cache is the only unambiguous miss, so it is used here.
        """
        self.assertEqual(self.cache.stats()["cache_misses"], 0)
        self.cache.get("nothing cached yet")
        stats = self.cache.stats()
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["hit_rate"], 0.0)

    def test_top_similar(self):
        self.cache.set("Python is great", "Yes it is.")
        self.cache.set("Rust is fast", "Very fast.")
        results = self.cache.top_similar("Tell me about Python", n=2)
        self.assertEqual(len(results), 2)
        self.assertIn("similarity", results[0])

    def test_top_similar_orders_by_similarity(self):
        """The most similar entry is returned first.

        Uses a token-overlap embedder so the ordering is meaningful --
        under a hash embedder the two prompts are unrelated by construction.
        """
        cache = SemanticCache(
            embedding_fn=_keyword_embed,
            similarity_threshold=0.50,
            max_size=10,
        )
        cache.set("Python is great", "Yes it is.")
        cache.set("Rust is fast", "Very fast.")
        results = cache.top_similar("Tell me about Python", n=2)
        self.assertEqual(len(results), 2)
        self.assertIn("Python", results[0]["prompt"])
        self.assertGreaterEqual(results[0]["similarity"], results[1]["similarity"])

    def test_clear(self):
        self.cache.set("a", "b")
        self.cache.set("c", "d")
        self.cache.clear()
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_error_missing_embedder(self):
        with self.assertRaises(ValueError):
            SemanticCache(embedding_fn=None)

    def test_error_invalid_threshold(self):
        with self.assertRaises(ValueError):
            SemanticCache(embedding_fn=mock_embed, similarity_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
