"""
Hybrid Intelligence Portfolio System — Impact Scorer
=====================================================
Agent 5: Multi-factor impact scoring engine for financial news.

Calculates a composite impact score for each article based on:
  1. Source reputation weight (tier-based credibility)
  2. Sentiment strength (abs(score) × confidence)
  3. Topic importance (market-moving weight)
  4. Temporal recency (exponential decay)
  5. Entity relevance (how directly entities map to portfolio)
  6. Cluster boost (multiple articles = higher impact)

Final formula:
  impact = source_w × sentiment_strength × topic_w × recency × entity_relevance × cluster_boost
"""

import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ImpactScorer:
    """
    Multi-factor news impact scoring engine.
    
    Produces a 0.0-1.0 impact score reflecting how likely
    an article is to affect portfolio allocation decisions.
    """

    def __init__(self):
        from config.settings import NewsConfig
        self._config = NewsConfig
        self._scoring_stats = {
            "total_scored": 0,
            "high_impact": 0,    # > 0.6
            "medium_impact": 0,  # 0.3-0.6
            "low_impact": 0,     # < 0.3
        }

    @property
    def stats(self) -> dict:
        return self._scoring_stats.copy()

    def score_batch(
        self,
        articles: list[dict],
        clusters: dict[int, list[dict]] = None,
    ) -> list[dict]:
        """
        Calculate impact scores for a batch of articles.
        
        Args:
            articles: Articles with sentiment analysis results
            clusters: Optional article clusters from embedding engine
            
        Returns:
            Articles enriched with 'impact' dict
        """
        self._scoring_stats = {
            "total_scored": 0,
            "high_impact": 0,
            "medium_impact": 0,
            "low_impact": 0,
        }

        # Build cluster size lookup
        cluster_sizes = {}
        if clusters:
            for cid, cluster_articles in clusters.items():
                for article in cluster_articles:
                    cluster_sizes[article.get("article_id", "")] = len(cluster_articles)

        results = []
        for article in articles:
            impact = self._score_single(article, cluster_sizes)
            enriched = article.copy()
            enriched["impact"] = impact
            results.append(enriched)
            self._scoring_stats["total_scored"] += 1

            # Categorize
            score = impact["impact_score"]
            if score >= 0.6:
                self._scoring_stats["high_impact"] += 1
            elif score >= 0.3:
                self._scoring_stats["medium_impact"] += 1
            else:
                self._scoring_stats["low_impact"] += 1

        logger.info(
            f"Impact Scoring Complete: {self._scoring_stats['total_scored']} articles | "
            f"High: {self._scoring_stats['high_impact']}, "
            f"Medium: {self._scoring_stats['medium_impact']}, "
            f"Low: {self._scoring_stats['low_impact']}"
        )

        return results

    def _score_single(self, article: dict, cluster_sizes: dict) -> dict:
        """Calculate all impact factors for a single article."""

        # --- Factor 1: Source reputation ---
        source_weight = self._get_source_weight(article.get("source", "unknown"))

        # --- Factor 2: Sentiment strength ---
        sentiment = article.get("sentiment", {})
        raw_score = abs(sentiment.get("score", 0.0))
        confidence = sentiment.get("confidence", 0.3)
        sentiment_strength = raw_score * confidence

        # --- Factor 3: Topic importance ---
        topic = article.get("topic", "general_market")
        topic_importance = self._config.TOPIC_WEIGHTS.get(topic, 0.5)

        # --- Factor 4: Temporal recency ---
        recency = self._calculate_recency(article.get("published_at", ""))

        # --- Factor 5: Entity relevance ---
        entity_relevance = self._calculate_entity_relevance(
            article.get("entities", []),
            article.get("asset_classes", []),
        )

        # --- Factor 6: Cluster boost ---
        article_id = article.get("article_id", "")
        cluster_size = cluster_sizes.get(article_id, 1)
        cluster_boost = self._calculate_cluster_boost(cluster_size)

        # --- Final composite score ---
        # Weighted geometric-like combination for more balanced scoring
        raw_impact = (
            source_weight *
            max(sentiment_strength, 0.1) *  # Floor to avoid zeroing out
            topic_importance *
            recency *
            max(entity_relevance, 0.2) *
            cluster_boost
        )

        # Normalize to 0-1 range (the max raw value would be ~3.0)
        impact_score = min(raw_impact / 1.5, 1.0)

        return {
            "impact_score": round(impact_score, 4),
            "source_weight": round(source_weight, 4),
            "sentiment_strength": round(sentiment_strength, 4),
            "topic_importance": round(topic_importance, 4),
            "temporal_recency": round(recency, 4),
            "entity_relevance": round(entity_relevance, 4),
            "cluster_boost": round(cluster_boost, 4),
        }

    # ═════════════════════════════════════════════════════
    #  FACTOR CALCULATIONS
    # ═════════════════════════════════════════════════════

    def _get_source_weight(self, source: str) -> float:
        """Look up source reputation weight."""
        source_lower = source.lower().strip()

        # Direct match
        if source_lower in self._config.SOURCE_WEIGHTS:
            return self._config.SOURCE_WEIGHTS[source_lower]

        # Partial match (e.g., "CNBC News" → "cnbc")
        for key, weight in self._config.SOURCE_WEIGHTS.items():
            if key in source_lower or source_lower in key:
                return weight

        return self._config.SOURCE_WEIGHTS.get("unknown", 0.4)

    def _calculate_recency(self, published_at: str) -> float:
        """
        Calculate temporal recency factor using exponential decay.
        
        Uses half-life from config: articles lose 50% impact every N hours.
        """
        if not published_at:
            return 0.5  # Unknown date → moderate recency

        try:
            from ml.news_collector import NewsCollector
            pub_time = NewsCollector._parse_datetime(published_at)
            if pub_time is None:
                return 0.5

            now = datetime.now(timezone.utc)
            age_hours = (now - pub_time).total_seconds() / 3600.0

            if age_hours < 0:
                age_hours = 0  # Future date → treat as just now

            # Exponential decay: f(t) = 0.5^(t / half_life)
            half_life = self._config.IMPACT_HALF_LIFE_HOURS
            decay = math.pow(0.5, age_hours / half_life)

            return max(decay, 0.05)  # Minimum floor of 5%

        except Exception:
            return 0.5

    @staticmethod
    def _calculate_entity_relevance(
        entities: list[str],
        asset_classes: list[str],
    ) -> float:
        """
        Calculate how relevant the extracted entities are to portfolio allocation.
        
        Higher score for articles mentioning actual portfolio assets.
        """
        if not entities and not asset_classes:
            return 0.2  # Generic financial news

        # Portfolio-relevant asset classes
        portfolio_assets = {"crypto", "equities", "bonds", "commodities", "forex"}
        relevant_classes = set(asset_classes) & portfolio_assets
        
        if not relevant_classes:
            return 0.3

        # More relevant classes = higher score
        relevance = min(len(relevant_classes) / 3.0, 1.0)

        # Boost for multiple entities (detailed article)
        entity_count_bonus = min(len(entities) / 5.0, 0.3)

        return min(relevance + entity_count_bonus, 1.0)

    @staticmethod
    def _calculate_cluster_boost(cluster_size: int) -> float:
        """
        Calculate cluster boost factor.
        
        Multiple articles about the same event increase impact.
        Max boost is 2.0x for clusters of 5+ articles.
        """
        if cluster_size <= 1:
            return 1.0
        elif cluster_size == 2:
            return 1.2
        elif cluster_size == 3:
            return 1.4
        elif cluster_size == 4:
            return 1.6
        else:
            return min(1.0 + (cluster_size * 0.2), 2.5)
