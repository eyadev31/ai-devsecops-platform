"""
Hybrid Intelligence Portfolio System — News API Routes
========================================================
REST API endpoints for Agent 5: News Sentiment Intelligence.

Endpoints:
  GET  /api/news/latest     — Latest processed news signals
  GET  /api/news/sentiment  — Current aggregated sentiment by asset class
  GET  /api/news/events     — Detected events and alerts
  POST /api/news/analyze    — Trigger fresh news analysis
"""

import logging
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/news", tags=["News Intelligence"])

# Cache the latest news analysis result
_latest_analysis = None


@router.get("/latest")
async def get_latest_news(
    limit: int = Query(default=10, ge=1, le=50, description="Number of articles to return"),
):
    """
    Get the latest processed news signals.
    Returns top articles by impact score with sentiment analysis.
    """
    global _latest_analysis

    if _latest_analysis is None:
        # Run initial analysis with mock data
        _latest_analysis = _run_analysis(mock=False)

    articles = _latest_analysis.get("articles", [])[:limit]

    return {
        "status": "success",
        "timestamp": _latest_analysis.get("timestamp", ""),
        "article_count": len(articles),
        "total_processed": _latest_analysis.get("article_count", 0),
        "articles": articles,
        "market_signal": _latest_analysis.get("market_signal", {}),
    }


@router.get("/sentiment")
async def get_sentiment():
    """
    Get current aggregated sentiment by asset class.
    Returns temporal sentiment across all time windows.
    """
    global _latest_analysis

    if _latest_analysis is None:
        _latest_analysis = _run_analysis(mock=False)

    temporal = _latest_analysis.get("temporal_sentiment", {})
    signal = _latest_analysis.get("market_signal", {})

    return {
        "status": "success",
        "timestamp": _latest_analysis.get("timestamp", ""),
        "overall": {
            "1h": temporal.get("overall_1h", 0.0),
            "6h": temporal.get("overall_6h", 0.0),
            "24h": temporal.get("overall_24h", 0.0),
            "3d": temporal.get("overall_3d", 0.0),
            "7d": temporal.get("overall_7d", 0.0),
        },
        "by_asset_class": {
            "crypto": temporal.get("crypto_sentiment", {}),
            "equities": temporal.get("equities_sentiment", {}),
            "bonds": temporal.get("bonds_sentiment", {}),
            "commodities": temporal.get("commodities_sentiment", {}),
        },
        "momentum": temporal.get("sentiment_momentum", "stable"),
        "momentum_strength": temporal.get("momentum_strength", 0.0),
        "regime": temporal.get("regime_sentiment", "neutral"),
        "signal": {
            "type": signal.get("signal_type", "neutral"),
            "confidence": signal.get("confidence", 0.0),
            "strength": signal.get("signal_strength", "weak"),
            "bias": signal.get("recommended_bias", {}),
        },
    }


@router.get("/events")
async def get_events():
    """
    Get detected market events and risk alerts.
    Returns event detection results with severity and recommended actions.
    """
    global _latest_analysis

    if _latest_analysis is None:
        _latest_analysis = _run_analysis(mock=False)

    events = _latest_analysis.get("event_detection", {})

    return {
        "status": "success",
        "timestamp": _latest_analysis.get("timestamp", ""),
        "event_count": events.get("event_count", 0),
        "has_critical_event": events.get("has_critical_event", False),
        "highest_severity": events.get("highest_severity", "none"),
        "recommended_action": events.get("recommended_action", "monitor"),
        "risk_alert": events.get("risk_alert", ""),
        "events": events.get("events_detected", []),
    }


@router.post("/analyze")
async def trigger_analysis(
    mock: bool = Query(default=False, description="Use mock data (set true only for testing)"),
):
    """
    Trigger a fresh news analysis.
    Returns complete Agent 5 output.
    """
    global _latest_analysis

    try:
        _latest_analysis = _run_analysis(mock=mock)

        return {
            "status": "success",
            "timestamp": _latest_analysis.get("timestamp", ""),
            "article_count": _latest_analysis.get("article_count", 0),
            "market_signal": _latest_analysis.get("market_signal", {}),
            "temporal_sentiment": _latest_analysis.get("temporal_sentiment", {}),
            "event_detection": _latest_analysis.get("event_detection", {}),
            "llm_analysis": _latest_analysis.get("llm_analysis", {}),
            "metadata": _latest_analysis.get("agent_metadata", {}),
        }

    except Exception as e:
        logger.error(f"News analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")


def _run_analysis(mock: bool = False) -> dict:
    """Run Agent 5 analysis (internal helper)."""
    from agents.agent5_news import Agent5NewsIntelligence

    agent = Agent5NewsIntelligence()
    return agent.run(mock=mock)
