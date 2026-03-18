"""
Hybrid Intelligence Portfolio System — News Embedding Engine
=============================================================
Agent 5: Sentence-transformer vector embeddings for semantic
similarity, duplicate detection, and article clustering.

Uses: sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast, free)

Capabilities:
  1. Generate embeddings for article text
  2. Detect near-duplicate articles via cosine similarity
  3. Cluster related articles about the same event
  4. Track narrative trends via embedding drift
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class NewsEmbeddingEngine:
    """
    Vector embedding engine for financial news articles.
    
    Uses sentence-transformers for fast, high-quality embeddings.
    Falls back to TF-IDF-like hashing when sentence-transformers
    is unavailable.
    """

    def __init__(self):
        from config.settings import NewsConfig
        self._config = NewsConfig
        self._model = None
        self._model_available = None
        self._embedding_cache = {}  # article_id -> embedding

    def embed_articles(self, articles: list[dict]) -> list[dict]:
        """
        Generate embeddings for a batch of articles.
        
        Args:
            articles: Processed article dicts with 'combined_text' and 'article_id'
            
        Returns:
            Articles enriched with 'embedding' field
        """
        texts = [a.get("combined_text", a.get("title", "")) for a in articles]
        embeddings = self._encode_batch(texts)

        results = []
        for article, emb in zip(articles, embeddings):
            enriched = article.copy()
            enriched["embedding"] = emb
            # Cache for later similarity lookups
            self._embedding_cache[article.get("article_id", "")] = emb
            results.append(enriched)

        return results

    def find_duplicates(
        self,
        articles: list[dict],
        threshold: float = 0.78,
    ) -> list[dict]:
        """
        Remove near-duplicate articles using cosine similarity.
        
        Args:
            articles: Articles with 'embedding' field
            threshold: Similarity threshold above which articles are considered duplicates
            
        Returns:
            Deduplicated list (keeps first occurrence)
        """
        if not articles:
            return []

        unique = [articles[0]]
        unique_embeddings = [articles[0].get("embedding")]

        for article in articles[1:]:
            emb = article.get("embedding")
            if emb is None:
                unique.append(article)
                continue

            is_duplicate = False
            for existing_emb in unique_embeddings:
                if existing_emb is not None:
                    sim = self._cosine_similarity(emb, existing_emb)
                    if sim >= threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(article)
                unique_embeddings.append(emb)

        removed = len(articles) - len(unique)
        if removed > 0:
            logger.info(f"Embedding dedup: removed {removed} near-duplicates (threshold={threshold})")

        return unique

    def cluster_articles(
        self,
        articles: list[dict],
        threshold: float = 0.78,
    ) -> dict[int, list[dict]]:
        """
        Cluster related articles about the same event/topic.
        
        Simple agglomerative clustering using cosine similarity.
        
        Returns:
            Dict mapping cluster_id -> list of articles
        """
        if not articles:
            return {}

        n = len(articles)
        cluster_ids = list(range(n))  # Each article starts in its own cluster

        # Pairwise similarity check
        for i in range(n):
            for j in range(i + 1, n):
                emb_i = articles[i].get("embedding")
                emb_j = articles[j].get("embedding")
                if emb_i is not None and emb_j is not None:
                    sim = self._cosine_similarity(emb_i, emb_j)
                    if sim >= threshold:
                        # Merge clusters: assign smaller cluster_id
                        old_id = cluster_ids[j]
                        new_id = cluster_ids[i]
                        for k in range(n):
                            if cluster_ids[k] == old_id:
                                cluster_ids[k] = new_id

        # Group articles by cluster
        clusters = {}
        for idx, cid in enumerate(cluster_ids):
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(articles[idx])

        logger.info(f"Article clustering: {n} articles -> {len(clusters)} clusters")

        return clusters

    # ═════════════════════════════════════════════════════
    #  ENCODING
    # ═════════════════════════════════════════════════════

    def _encode_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Encode a batch of texts into embeddings."""
        # Try sentence-transformers first
        embeddings = self._encode_sentence_transformers(texts)
        if embeddings is not None:
            return embeddings

        # Fallback: simple hash-based pseudo-embeddings
        return [self._hash_embedding(t) for t in texts]

    def _encode_sentence_transformers(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Encode using sentence-transformers model."""
        if self._model_available is False:
            return None

        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._config.EMBEDDING_MODEL)
                self._model_available = True
                logger.info(f"Sentence-transformer model '{self._config.EMBEDDING_MODEL}' loaded.")

            # Truncate texts for efficiency
            truncated = [t[:512] if t else "" for t in texts]
            embeddings = self._model.encode(truncated, show_progress_bar=False)

            return [emb.tolist() for emb in embeddings]

        except ImportError:
            logger.info("sentence-transformers not installed. Using hash-based fallback.")
            self._model_available = False
        except Exception as e:
            logger.warning(f"Sentence-transformer encoding failed: {e}")
            self._model_available = False

        return None

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> list[float]:
        """
        Fallback: Generate a deterministic pseudo-embedding from text hash.
        
        This provides basic duplicate detection capability even without
        sentence-transformers installed.
        """
        import hashlib

        if not text:
            return [0.0] * dim

        # Create multiple hash seeds from different text segments
        embedding = []
        words = text.lower().split()
        for i in range(dim):
            # Use word-level n-grams for some semantic sensitivity
            start = i % max(len(words), 1)
            segment = " ".join(words[start:start + 3])
            seed = f"{segment}_{i}"
            h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
            # Map to [-1, 1] range
            val = (h % 10000) / 10000.0 * 2 - 1
            embedding.append(val)

        # Normalize to unit length
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    # ═════════════════════════════════════════════════════
    #  SIMILARITY
    # ═════════════════════════════════════════════════════

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
