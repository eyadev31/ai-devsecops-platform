"""
Hybrid Intelligence Portfolio System — Temporal Aggregator
===========================================================
Agent 5: Sentiment momentum computation across time windows.

Aggregates individual article sentiments into temporal signals:
  - 1 hour, 6 hours, 24 hours, 3 days, 7 days
  - Per-asset-class sentiment trends
  - Momentum direction (accelerating / stable / decelerating)
  - Overall regime sentiment classification

This creates momentum signals that are critical for detecting
sentiment shifts before they materialize in price action.
Used by Agent 3 (Allocation Optimizer) to bias portfolio weights.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """
    Temporal sentiment aggregation engine.
    
    Computes exponentially-weighted sentiment averages across
    multiple time windows, broken down by asset class.
    """

    def __init__(self):
        from config.settings import NewsConfig
        self._config = NewsConfig
        self._aggregation_stats = {
            "articles_aggregated": 0,
            "windows_computed": 0,
            "asset_classes_tracked": 0,
        }

    @property
    def stats(self) -> dict:
        return self._aggregation_stats.copy()

    def aggregate(self, articles: list[dict]) -> dict:
        """
        Compute temporal sentiment aggregation.
        
        Args:
            articles: Scored articles with sentiment, impact, published_at
            
        Returns:
            TemporalSentiment-compatible dict
        """
        self._aggregation_stats = {
            "articles_aggregated": len(articles),
            "windows_computed": 0,
            "asset_classes_tracked": 0,
        }

        now = datetime.now(timezone.utc)

        # --- Overall sentiment per window ---
        overall = {}
        for window_name, hours in self._config.TEMPORAL_WINDOWS.items():
            cutoff = now - timedelta(hours=hours)
            window_articles = self._filter_by_time(articles, cutoff, now)
            overall[window_name] = self._compute_weighted_sentiment(window_articles, now)
            self._aggregation_stats["windows_computed"] += 1

        # --- Per-asset-class sentiment ---
        asset_classes = ["crypto", "equities", "bonds", "commodities"]
        per_asset = {}
        for asset_class in asset_classes:
            asset_articles = [
                a for a in articles
                if asset_class in a.get("asset_classes", [])
            ]
            if asset_articles:
                self._aggregation_stats["asset_classes_tracked"] += 1
                asset_sentiment = {}
                for window_name, hours in self._config.TEMPORAL_WINDOWS.items():
                    cutoff = now - timedelta(hours=hours)
                    window_articles = self._filter_by_time(asset_articles, cutoff, now)
                    asset_sentiment[window_name] = self._compute_weighted_sentiment(window_articles, now)
                per_asset[asset_class] = asset_sentiment
            else:
                per_asset[asset_class] = {w: 0.0 for w in self._config.TEMPORAL_WINDOWS}

        # --- Compute momentum ---
        momentum_direction, momentum_strength = self._compute_momentum(overall)

        # --- Classify regime sentiment ---
        regime = self._classify_regime(overall)

        result = {
            "overall_1h": overall.get("1h", 0.0),
            "overall_6h": overall.get("6h", 0.0),
            "overall_24h": overall.get("24h", 0.0),
            "overall_3d": overall.get("3d", 0.0),
            "overall_7d": overall.get("7d", 0.0),
            "crypto_sentiment": per_asset.get("crypto", {}),
            "equities_sentiment": per_asset.get("equities", {}),
            "bonds_sentiment": per_asset.get("bonds", {}),
            "commodities_sentiment": per_asset.get("commodities", {}),
            "sentiment_momentum": momentum_direction,
            "momentum_strength": round(momentum_strength, 4),
            "regime_sentiment": regime,
        }

        logger.info(
            f"Temporal Aggregation Complete: {len(articles)} articles | "
            f"Regime: {regime} | Momentum: {momentum_direction} ({momentum_strength:+.2f}) | "
            f"1h={overall.get('1h', 0):.2f}, 24h={overall.get('24h', 0):.2f}, 7d={overall.get('7d', 0):.2f}"
        )

        return result

    # ═════════════════════════════════════════════════════
    #  COMPUTATION HELPERS
    # ═════════════════════════════════════════════════════

    def _compute_weighted_sentiment(
        self,
        articles: list[dict],
        now: datetime,
    ) -> float:
        """
        Compute exponentially-weighted average sentiment.
        
        More recent articles receive higher weight.
        Impact score is also used as a weight multiplier.
        """
        if not articles:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for article in articles:
            sentiment = article.get("sentiment", {})
            score = sentiment.get("score", 0.0)
            confidence = sentiment.get("confidence", 0.3)
            impact_score = article.get("impact", {}).get("impact_score", 0.5)

            # Recency weight (exponential decay)
            age_hours = self._get_age_hours(article.get("published_at", ""), now)
            half_life = self._config.IMPACT_HALF_LIFE_HOURS
            recency_weight = math.pow(0.5, age_hours / half_life) if age_hours >= 0 else 1.0

            # Combined weight: confidence × impact × recency
            weight = confidence * impact_score * recency_weight

            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        result = weighted_sum / total_weight
        return round(max(-1.0, min(1.0, result)), 4)

    @staticmethod
    def _compute_momentum(overall: dict) -> tuple[str, float]:
        """
        Compute sentiment momentum direction and strength.
        
        Momentum = short-term sentiment trend relative to long-term.
        Positive momentum = sentiment improving (bullish acceleration).
        Negative momentum = sentiment deteriorating (bearish acceleration).
        """
        short_term = overall.get("1h", overall.get("6h", 0.0))
        medium_term = overall.get("24h", 0.0)
        long_term = overall.get("7d", 0.0)

        # Momentum: difference between short and long-term sentiment
        if long_term != 0:
            momentum = short_term - long_term
        else:
            momentum = short_term - medium_term

        strength = max(-1.0, min(1.0, momentum))

        # Classify direction
        if strength > 0.3:
            direction = "accelerating_bullish"
        elif strength > 0.1:
            direction = "bullish"
        elif strength < -0.3:
            direction = "accelerating_bearish"
        elif strength < -0.1:
            direction = "bearish"
        else:
            direction = "stable"

        return direction, strength

    @staticmethod
    def _classify_regime(overall: dict) -> str:
        """
        Classify the overall sentiment regime.
        
        Based on the weighted combination of sentiment windows.
        """
        # Weight more recent windows more heavily
        weights = {"1h": 0.30, "6h": 0.25, "24h": 0.20, "3d": 0.15, "7d": 0.10}
        
        weighted_avg = 0.0
        total_w = 0.0
        for window, w in weights.items():
            val = overall.get(window, 0.0)
            weighted_avg += val * w
            total_w += w

        if total_w > 0:
            weighted_avg /= total_w

        if weighted_avg >= 0.5:
            return "very_bullish"
        elif weighted_avg >= 0.2:
            return "bullish"
        elif weighted_avg >= -0.2:
            return "neutral"
        elif weighted_avg >= -0.5:
            return "bearish"
        else:
            return "very_bearish"

    # ═════════════════════════════════════════════════════
    #  UTILITIES
    # ═════════════════════════════════════════════════════

    @staticmethod
    def _filter_by_time(
        articles: list[dict],
        cutoff: datetime,
        now: datetime,
    ) -> list[dict]:
        """Filter articles to those within the time window."""
        result = []
        for article in articles:
            pub_str = article.get("published_at", "")
            if not pub_str:
                continue
            try:
                from ml.news_collector import NewsCollector
                pub_time = NewsCollector._parse_datetime(pub_str)
                if pub_time and pub_time >= cutoff:
                    result.append(article)
            except Exception:
                result.append(article)  # Include if can't parse
        return result

    @staticmethod
    def _get_age_hours(published_at: str, now: datetime) -> float:
        """Get article age in hours."""
        if not published_at:
            return 24.0  # Default to 1 day old
        try:
            from ml.news_collector import NewsCollector
            pub_time = NewsCollector._parse_datetime(published_at)
            if pub_time:
                return max((now - pub_time).total_seconds() / 3600.0, 0.0)
        except Exception:
            pass
        return 24.0
