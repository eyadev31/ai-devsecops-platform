"""
Hybrid Intelligence Portfolio System — Agent 5 Output Schema
================================================================
Pydantic models for validating and documenting the structured
JSON output of Agent 5 (News Sentiment Intelligence Agent).

This schema defines the contract between Agent 5 and downstream
agents (Agent 3 Allocation Optimizer, Agent 4 Risk Supervisor).

Responsibilities:
  1. Validation — reject malformed or incomplete news signals
  2. Documentation — define the exact contract for downstream agents
  3. Type safety — catch structural issues early
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════
#  SUB-MODELS
# ═══════════════════════════════════════════════════════

class SentimentResult(BaseModel):
    """Dual-model sentiment analysis result."""
    label: str = Field(..., description="positive | neutral | negative")
    score: float = Field(..., ge=-1.0, le=1.0, description="Numeric sentiment: -1 (bearish) to +1 (bullish)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Model confidence in sentiment")
    model_source: str = Field(default="finbert", description="finbert | llm | hybrid")
    finbert_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Raw FinBERT numeric score")
    llm_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Raw LLM numeric score")
    agreement: bool = Field(default=True, description="Whether FinBERT and LLM agree on direction")


class EntityExtraction(BaseModel):
    """Detected financial entities with asset mapping."""
    entities: list[str] = Field(default_factory=list, description="Raw extracted entity names")
    tickers: list[str] = Field(default_factory=list, description="Mapped ticker symbols")
    asset_classes: list[str] = Field(default_factory=list, description="Affected asset classes: crypto, equities, bonds, commodities, forex")
    organizations: list[str] = Field(default_factory=list, description="Organizations mentioned (Fed, SEC, etc.)")
    topics: list[str] = Field(default_factory=list, description="Classified topics: interest_rates, regulation, etc.")


class ImpactAssessment(BaseModel):
    """Multi-factor impact scoring."""
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Final composite impact score")
    source_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Source reputation factor")
    sentiment_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="abs(sentiment) × confidence")
    topic_importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Topic weight")
    temporal_recency: float = Field(default=1.0, ge=0.0, le=1.0, description="Recency decay factor")
    entity_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant entities are to portfolio")
    cluster_boost: float = Field(default=1.0, ge=1.0, le=3.0, description="Boost from multiple related articles")


class NewsArticle(BaseModel):
    """Individual processed news article."""
    article_id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article headline")
    summary: str = Field(default="", description="Article summary/body text")
    source: str = Field(..., description="News source name")
    source_url: str = Field(default="", description="Original article URL")
    published_at: str = Field(..., description="Publication timestamp ISO format")
    collected_at: str = Field(default="", description="Collection timestamp ISO format")
    sentiment: SentimentResult
    entities: EntityExtraction
    impact: ImpactAssessment
    topic: str = Field(default="general_market", description="Primary classified topic")
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Financial relevance score")


class EventDetection(BaseModel):
    """Detected macro/black swan events."""
    events_detected: list[dict] = Field(default_factory=list, description="List of detected events with type, severity, details")
    has_critical_event: bool = Field(default=False, description="Whether a critical/black_swan event was detected")
    highest_severity: str = Field(default="none", description="none | routine | significant | major | critical | black_swan")
    event_count: int = Field(default=0, ge=0, description="Total events detected")
    risk_alert: str = Field(default="", description="Human-readable risk alert message")
    recommended_action: str = Field(default="monitor", description="monitor | review | hedge | derisk | emergency_exit")

    @field_validator("highest_severity")
    @classmethod
    def validate_severity(cls, v):
        valid = {"none", "routine", "significant", "major", "critical", "black_swan"}
        if v not in valid:
            return "none"
        return v


class TemporalSentiment(BaseModel):
    """Aggregated sentiment across time windows."""
    overall_1h: float = Field(default=0.0, ge=-1.0, le=1.0, description="Avg sentiment last 1 hour")
    overall_6h: float = Field(default=0.0, ge=-1.0, le=1.0, description="Avg sentiment last 6 hours")
    overall_24h: float = Field(default=0.0, ge=-1.0, le=1.0, description="Avg sentiment last 24 hours")
    overall_3d: float = Field(default=0.0, ge=-1.0, le=1.0, description="Avg sentiment last 3 days")
    overall_7d: float = Field(default=0.0, ge=-1.0, le=1.0, description="Avg sentiment last 7 days")
    crypto_sentiment: dict[str, float] = Field(default_factory=dict, description="Per-window crypto sentiment")
    equities_sentiment: dict[str, float] = Field(default_factory=dict, description="Per-window equities sentiment")
    bonds_sentiment: dict[str, float] = Field(default_factory=dict, description="Per-window bonds sentiment")
    commodities_sentiment: dict[str, float] = Field(default_factory=dict, description="Per-window commodities sentiment")
    sentiment_momentum: str = Field(default="stable", description="accelerating_bullish | bullish | stable | bearish | accelerating_bearish")
    momentum_strength: float = Field(default=0.0, ge=-1.0, le=1.0, description="Momentum magnitude and direction")
    regime_sentiment: str = Field(default="neutral", description="very_bearish | bearish | neutral | bullish | very_bullish")


class MarketSignal(BaseModel):
    """Final structured market signal for downstream agents."""
    signal_type: str = Field(..., description="bullish | bearish | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    affected_assets: list[str] = Field(default_factory=list, description="Asset classes affected")
    signal_strength: str = Field(default="moderate", description="weak | moderate | strong | extreme")
    primary_driver: str = Field(default="", description="Main news theme driving the signal")
    recommended_bias: dict[str, float] = Field(
        default_factory=dict,
        description="Per-asset-class allocation bias: positive = overweight, negative = underweight"
    )
    narrative: str = Field(default="", description="Institutional-quality market intelligence briefing")
    risk_events: list[str] = Field(default_factory=list, description="Active risk events to monitor")


class LLMNewsAnalysis(BaseModel):
    """LLM-generated qualitative news analysis."""
    market_narrative: str = Field(default="", description="Institutional-quality news intelligence briefing")
    key_themes: list[str] = Field(default_factory=list, description="Top 3-5 dominant market themes")
    risk_assessment: str = Field(default="", description="Overall risk assessment from news")
    allocation_implications: str = Field(default="", description="How news should affect portfolio allocation")
    confidence_level: float = Field(default=0.5, ge=0.0, le=1.0, description="LLM confidence in analysis")
    contrarian_signals: list[str] = Field(default_factory=list, description="Signals that contradict consensus")


class Agent5Metadata(BaseModel):
    """Agent 5 execution metadata."""
    agent_id: str = Field(default="agent5_news_intelligence")
    version: str = Field(default="1.0.0")
    execution_time_ms: float = Field(default=0.0, description="Total pipeline execution time")
    timestamp: str = Field(default="", description="Execution timestamp")
    articles_collected: int = Field(default=0, ge=0, description="Total raw articles collected")
    articles_processed: int = Field(default=0, ge=0, description="Articles after filtering/dedup")
    sources_queried: list[str] = Field(default_factory=list, description="News sources successfully queried")
    sources_failed: list[str] = Field(default_factory=list, description="News sources that failed")
    models_used: list[str] = Field(default_factory=list, description="ML/NLP models used")
    data_quality: str = Field(default="full", description="full | partial | degraded | mock")
    execution_log: list[Any] = Field(default_factory=list, description="Step-by-step execution log")


# ═══════════════════════════════════════════════════════
#  MAIN OUTPUT MODEL
# ═══════════════════════════════════════════════════════

class Agent5Output(BaseModel):
    """
    Complete Agent 5 output schema.
    This is the contract between Agent 5 and downstream agents (3, 4).
    
    Agent 5 produces:
      1. Processed news articles with sentiment & impact
      2. Aggregated temporal sentiment momentum signals
      3. Event detection alerts
      4. Final market signal for portfolio allocation
      5. LLM-generated qualitative analysis
    """
    timestamp: str = Field(..., description="UTC timestamp of analysis")
    data_freshness: str = Field(default="", description="Timestamp of most recent article")

    # Core intelligence outputs
    articles: list[NewsArticle] = Field(default_factory=list, description="Processed news articles (top by impact)")
    article_count: int = Field(default=0, ge=0, description="Total articles processed")

    # Aggregated signals
    temporal_sentiment: TemporalSentiment = Field(default_factory=TemporalSentiment)
    event_detection: EventDetection = Field(default_factory=EventDetection)
    market_signal: MarketSignal

    # LLM analysis
    llm_analysis: LLMNewsAnalysis = Field(default_factory=LLMNewsAnalysis)

    # Execution metadata
    agent_metadata: Agent5Metadata = Field(default_factory=Agent5Metadata)

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-03-06T17:00:00Z",
                "data_freshness": "2026-03-06T16:55:00Z",
                "article_count": 47,
                "market_signal": {
                    "signal_type": "bullish",
                    "confidence": 0.73,
                    "affected_assets": ["crypto", "equities"],
                    "signal_strength": "moderate",
                    "primary_driver": "Federal Reserve signals dovish stance",
                    "recommended_bias": {
                        "crypto": 0.10,
                        "equities": 0.05,
                        "bonds": -0.05,
                        "commodities": 0.02,
                    },
                },
            }
        }


# ═══════════════════════════════════════════════════════
#  VALIDATION HELPER
# ═══════════════════════════════════════════════════════

def validate_news_output(output: dict) -> tuple[bool, str]:
    """
    Validate Agent 5 output against the schema.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        Agent5Output.model_validate(output)
        return True, ""
    except Exception as e:
        return False, str(e)
