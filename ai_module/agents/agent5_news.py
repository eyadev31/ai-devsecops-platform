"""
Hybrid Intelligence Portfolio System -- Agent 5 Orchestrator
===============================================================
News Sentiment Intelligence Agent

The "Wall Street Wire" of the system.
Transforms financial news into structured market signals.

Pipeline (10 Steps):
  1. Collect news from multiple sources (RSS, NewsAPI, GDELT, CryptoPanic)
  2. Clean & filter articles (NLP pipeline)
  3. Score financial relevance
  4. Extract entities (financial NER)
  5. Generate embeddings (sentence-transformers)
  6. Deduplicate & cluster related articles
  7. Analyze sentiment (hybrid FinBERT + LLM)
  8. Score impact (multi-factor)
  9. Detect events (black swan / macro)
  10. Aggregate temporal momentum + LLM synthesis

Output feeds into:
  - Agent 3 (Allocation Optimizer) — bias portfolio weights
  - Agent 4 (Risk Supervisor) — event risk overlay
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ml.news_collector import NewsCollector
from ml.news_processor import NewsProcessor
from ml.sentiment_engine import SentimentEngine
from ml.news_embedding import NewsEmbeddingEngine
from ml.impact_scorer import ImpactScorer
from ml.event_detector import EventDetector
from ml.temporal_aggregator import TemporalAggregator
from llm.news_analyst import NewsAnalyst

logger = logging.getLogger(__name__)


class Agent5NewsIntelligence:
    """
    Agent 5 -- News Sentiment Intelligence Agent.

    Collects, processes, and analyzes financial news from multiple
    sources to produce structured market signals for portfolio
    allocation and risk management.

    Follows same conventions as Agents 1-4:
      - run(mock=False) for live execution
      - run_mock() for mock execution
      - Structured execution logging
      - Schema-validated output
    """

    def __init__(self):
        self._collector = NewsCollector()
        self._processor = NewsProcessor()
        self._sentiment = SentimentEngine()
        self._embedding = NewsEmbeddingEngine()
        self._impact = ImpactScorer()
        self._event_detector = EventDetector()
        self._temporal = TemporalAggregator()
        self._analyst = NewsAnalyst()
        self._execution_log = []

    def run(
        self,
        mock: bool = False,
        agent1_output: Optional[dict] = None,
    ) -> dict:
        """
        Execute the full Agent 5 pipeline.

        Args:
            mock: If True, use mock data (no API calls required)
            agent1_output: Optional Agent 1 context for regime-aware analysis

        Returns:
            Complete Agent 5 output JSON (validated against schema)
        """
        if mock:
            return self._run_pipeline(mock=True, agent1_output=agent1_output)
        return self._run_pipeline(mock=False, agent1_output=agent1_output)

    def run_mock(self, agent1_output: Optional[dict] = None) -> dict:
        """Run Agent 5 with mock data (no API calls)."""
        return self.run(mock=True, agent1_output=agent1_output)

    def _run_pipeline(
        self,
        mock: bool = False,
        agent1_output: Optional[dict] = None,
    ) -> dict:
        """Execute the full 10-step News Intelligence pipeline."""
        self._log_banner("AGENT 5 -- NEWS SENTIMENT INTELLIGENCE")
        self._execution_log.clear()
        start = time.time()
        data_quality = "full"

        # ══════════════════════════════════════════════════
        # STEP 1: NEWS COLLECTION
        # ══════════════════════════════════════════════════
        self._log_step(1, 10, "Collecting news from multiple sources...")
        step_start = time.time()

        try:
            if mock:
                raw_articles = self._collector.collect_mock()
                data_quality = "mock"
            else:
                raw_articles = self._collector.collect(
                    max_articles=100,
                    max_age_hours=168,  # 7 days
                )
                if not raw_articles:
                    logger.warning("No articles collected from any source. Falling back to mock data.")
                    raw_articles = self._collector.collect_mock()
                    data_quality = "degraded"
        except Exception as e:
            logger.error(f"Collection failed: {e}. Using mock data.")
            raw_articles = self._collector.collect_mock()
            data_quality = "degraded"

        collection_stats = self._collector.stats
        self._log_execution("collection", step_start, {
            "articles_raw": len(raw_articles),
            "sources_queried": collection_stats.get("sources_queried", []),
            "sources_failed": collection_stats.get("sources_failed", []),
        })

        # ══════════════════════════════════════════════════
        # STEP 2: TEXT CLEANING & FILTERING
        # ══════════════════════════════════════════════════
        self._log_step(2, 10, "Cleaning and filtering articles...")
        step_start = time.time()

        processed_articles = self._processor.process_batch(
            raw_articles, min_relevance=0.15
        )

        self._log_execution("processing", step_start, {
            "input": len(raw_articles),
            "output": len(processed_articles),
            "filtered": len(raw_articles) - len(processed_articles),
        })

        # ══════════════════════════════════════════════════
        # STEP 3: GENERATE EMBEDDINGS
        # ══════════════════════════════════════════════════
        self._log_step(3, 10, "Generating vector embeddings...")
        step_start = time.time()

        embedded_articles = self._embedding.embed_articles(processed_articles)

        self._log_execution("embedding", step_start, {
            "articles_embedded": len(embedded_articles),
        })

        # ══════════════════════════════════════════════════
        # STEP 4: SEMANTIC DEDUPLICATION
        # ══════════════════════════════════════════════════
        self._log_step(4, 10, "Running semantic deduplication...")
        step_start = time.time()

        unique_articles = self._embedding.find_duplicates(
            embedded_articles, threshold=0.85
        )

        self._log_execution("deduplication", step_start, {
            "before": len(embedded_articles),
            "after": len(unique_articles),
            "removed": len(embedded_articles) - len(unique_articles),
        })

        # ══════════════════════════════════════════════════
        # STEP 5: ARTICLE CLUSTERING
        # ══════════════════════════════════════════════════
        self._log_step(5, 10, "Clustering related articles...")
        step_start = time.time()

        clusters = self._embedding.cluster_articles(
            unique_articles, threshold=0.70
        )

        self._log_execution("clustering", step_start, {
            "articles": len(unique_articles),
            "clusters": len(clusters),
        })

        # ══════════════════════════════════════════════════
        # STEP 6: SENTIMENT ANALYSIS
        # ══════════════════════════════════════════════════
        self._log_step(6, 10, "Running hybrid sentiment analysis (FinBERT + LLM)...")
        step_start = time.time()

        sentiment_articles = self._sentiment.analyze_batch(unique_articles)
        sentiment_stats = self._sentiment.stats

        self._log_execution("sentiment_analysis", step_start, {
            "analyzed": len(sentiment_articles),
            "finbert_used": sentiment_stats.get("finbert_used", 0),
            "llm_used": sentiment_stats.get("llm_used", 0),
            "fallback_used": sentiment_stats.get("fallback_used", 0),
            "agreements": sentiment_stats.get("model_agreements", 0),
        })

        # ══════════════════════════════════════════════════
        # STEP 7: IMPACT SCORING
        # ══════════════════════════════════════════════════
        self._log_step(7, 10, "Computing multi-factor impact scores...")
        step_start = time.time()

        scored_articles = self._impact.score_batch(
            sentiment_articles, clusters=clusters
        )
        impact_stats = self._impact.stats

        # Sort by impact (highest first)
        scored_articles.sort(
            key=lambda a: a.get("impact", {}).get("impact_score", 0),
            reverse=True,
        )

        self._log_execution("impact_scoring", step_start, {
            "scored": len(scored_articles),
            "high_impact": impact_stats.get("high_impact", 0),
            "medium_impact": impact_stats.get("medium_impact", 0),
            "low_impact": impact_stats.get("low_impact", 0),
        })

        # ══════════════════════════════════════════════════
        # STEP 8: EVENT DETECTION
        # ══════════════════════════════════════════════════
        self._log_step(8, 10, "Scanning for black swan / macro events...")
        step_start = time.time()

        event_result = self._event_detector.detect_events(scored_articles)

        self._log_execution("event_detection", step_start, {
            "events_detected": event_result.get("event_count", 0),
            "highest_severity": event_result.get("highest_severity", "none"),
            "has_critical": event_result.get("has_critical_event", False),
        })

        # ══════════════════════════════════════════════════
        # STEP 9: TEMPORAL AGGREGATION
        # ══════════════════════════════════════════════════
        self._log_step(9, 10, "Computing temporal sentiment momentum...")
        step_start = time.time()

        temporal_result = self._temporal.aggregate(scored_articles)

        self._log_execution("temporal_aggregation", step_start, {
            "regime": temporal_result.get("regime_sentiment", "neutral"),
            "momentum": temporal_result.get("sentiment_momentum", "stable"),
            "1h": temporal_result.get("overall_1h", 0),
            "24h": temporal_result.get("overall_24h", 0),
            "7d": temporal_result.get("overall_7d", 0),
        })

        # ══════════════════════════════════════════════════
        # STEP 10: LLM MARKET SIGNAL SYNTHESIS
        # ══════════════════════════════════════════════════
        self._log_step(10, 10, "Synthesizing final market signal (LLM)...")
        step_start = time.time()

        market_signal, llm_analysis = self._analyst.analyze(
            articles=scored_articles[:15],  # Top 15 by impact
            temporal_sentiment=temporal_result,
            event_detection=event_result,
            agent1_context=agent1_output,
        )

        self._log_execution("market_signal_synthesis", step_start, {
            "signal_type": market_signal.get("signal_type", "neutral"),
            "confidence": market_signal.get("confidence", 0),
            "llm_calls": self._analyst.llm_calls,
        })

        # ══════════════════════════════════════════════════
        # ASSEMBLE FINAL OUTPUT
        # ══════════════════════════════════════════════════
        total_ms = (time.time() - start) * 1000

        # Build article list for output (top 20 by impact, without embeddings)
        output_articles = []
        for a in scored_articles[:20]:
            output_articles.append({
                "article_id": a.get("article_id", ""),
                "title": a.get("title", ""),
                "summary": a.get("summary", ""),
                "source": a.get("source", "unknown"),
                "source_url": a.get("url", ""),
                "published_at": a.get("published_at", ""),
                "collected_at": a.get("collected_at", ""),
                "sentiment": a.get("sentiment", {}),
                "entities": {
                    "entities": a.get("entities", []),
                    "tickers": [],  # Could be enriched further
                    "asset_classes": a.get("asset_classes", []),
                    "organizations": a.get("organizations", []),
                    "topics": [a.get("topic", "general_market")],
                },
                "impact": a.get("impact", {}),
                "topic": a.get("topic", "general_market"),
                "relevance_score": a.get("relevance_score", 0.5),
            })

        # Data freshness: most recent article
        data_freshness = ""
        if scored_articles:
            data_freshness = scored_articles[0].get("published_at", "")

        # Models used
        models_used = ["news_collector_v1", "news_processor_v1"]
        if sentiment_stats.get("finbert_used", 0) > 0:
            models_used.append("ProsusAI/finbert")
        if sentiment_stats.get("llm_used", 0) > 0 or self._analyst.llm_calls > 0:
            models_used.append("llm_core_model")
        if sentiment_stats.get("fallback_used", 0) > 0:
            models_used.append("rule_based_sentiment")
        models_used.extend(["impact_scorer_v1", "event_detector_v1", "temporal_aggregator_v1"])

        output = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "data_freshness": data_freshness,
            "articles": output_articles,
            "article_count": len(scored_articles),
            "temporal_sentiment": temporal_result,
            "event_detection": event_result,
            "market_signal": market_signal,
            "llm_analysis": llm_analysis,
            "agent_metadata": {
                "agent_id": "agent5_news_intelligence",
                "version": "1.0.0",
                "execution_time_ms": round(total_ms),
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "articles_collected": collection_stats.get("total_raw", 0),
                "articles_processed": len(scored_articles),
                "sources_queried": collection_stats.get("sources_queried", []),
                "sources_failed": collection_stats.get("sources_failed", []),
                "models_used": models_used,
                "data_quality": data_quality,
                "execution_log": self._execution_log.copy(),
            },
        }

        # ══════════════════════════════════════════════════
        # LOG SUMMARY
        # ══════════════════════════════════════════════════
        signal = market_signal
        logger.info(
            f"\n{'='*60}\n"
            f"AGENT 5 COMPLETE\n"
            f"  Articles:     {len(scored_articles)} processed (from {collection_stats.get('total_raw', 0)} raw)\n"
            f"  Signal:       {signal.get('signal_type', 'N/A')} ({signal.get('signal_strength', 'N/A')})\n"
            f"  Confidence:   {signal.get('confidence', 0):.1%}\n"
            f"  Regime:       {temporal_result.get('regime_sentiment', 'N/A')}\n"
            f"  Momentum:     {temporal_result.get('sentiment_momentum', 'N/A')}\n"
            f"  Events:       {event_result.get('event_count', 0)} ({event_result.get('highest_severity', 'none')})\n"
            f"  Bias:         {json.dumps(signal.get('recommended_bias', {}))}\n"
            f"  Exec Time:    {total_ms:.0f}ms\n"
            f"  Data Quality: {data_quality}\n"
            f"{'='*60}"
        )

        return output

    # ════════════════════════════════════════════════════
    #  LOGGING
    # ════════════════════════════════════════════════════

    def _log_banner(self, title: str):
        logger.info(f"\n{'+'*60}")
        logger.info(f"|  {title:<56}|")
        logger.info(f"|  Hybrid Intelligence Portfolio System v1.0.0{' '*11}|")
        logger.info(f"{'+'*60}")

    @staticmethod
    def _log_step(step: int, total: int, msg: str):
        logger.info(f"STEP {step}/{total} -- {msg}")

    def _log_execution(self, step_name: str, start_time: float, details: dict = None):
        duration = (time.time() - start_time) * 1000
        log_entry = {
            "step": step_name,
            "status": "success",
            "duration_ms": round(duration),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
        if details:
            log_entry["details"] = details
        self._execution_log.append(log_entry)
        logger.info(f"  └── {step_name} completed in {duration:.0f}ms")
