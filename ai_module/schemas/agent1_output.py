"""
Hybrid Intelligence Portfolio System — Agent 1 Output Schema
==============================================================
Pydantic models for validating and documenting the structured
JSON output of Agent 1 (Macro & Market Intelligence System).

These schemas serve as:
  1. Runtime validation — ensure output completeness
  2. Documentation — define the exact contract for downstream agents
  3. Type safety — catch structural issues early
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class MarketRegime(BaseModel):
    """Market regime classification from ensemble detector."""
    primary_regime: str = Field(..., description="Primary regime label (e.g., 'bull_low_vol', 'bear_high_vol')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in regime classification")
    hmm_regime: str = Field(default="unknown", description="HMM model regime prediction")
    rf_regime: str = Field(default="unknown", description="Random Forest regime prediction")
    models_agree: bool = Field(default=False, description="Whether HMM and RF agree")
    regime_duration_days: int = Field(default=0, ge=0, description="Days in current regime")
    transition_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of regime change")
    
    # ── Coherence Redesign Fields ──
    observations_count: int = Field(default=252, ge=0, description="Sample size N for HMM fitting")
    convergence_warning: bool = Field(default=False, description="Whether the inner model failed to mathematically converge")
    adjusted_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Statistical sample-size discounted confidence")
    effective_risk_state: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk blended between macro fundamentals and regime labels")
    
    description: str = Field(default="", description="Human-readable regime description")


class VolatilityState(BaseModel):
    """Volatility regime classification."""
    current_state: str = Field(..., description="extremely_low | low | normal | elevated | extreme")
    vix_level: Optional[float] = Field(None, description="Current VIX level")
    realized_vol_percentile: float = Field(default=50.0, ge=0, le=100, description="Percentile rank of current vol")
    vol_trend: str = Field(default="stable", description="Volatility direction: increasing | decreasing | stable")
    vol_of_vol: str = Field(default="normal", description="Volatility stability: stable | normal | elevated | unstable")
    term_structure: str = Field(default="unknown", description="contango | backwardation | flat | unknown")


class MacroEnvironment(BaseModel):
    """Macroeconomic environment assessment."""
    macro_regime: str = Field(..., description="Overall macro regime")
    monetary_policy: str = Field(default="unknown", description="Monetary policy stance")
    inflation_state: str = Field(default="unknown", description="Inflation regime")
    growth_state: str = Field(default="unknown", description="Economic growth state")
    liquidity: str = Field(default="unknown", description="Liquidity conditions")
    composite_score: float = Field(default=0.0, ge=-1.0, le=1.0, description="Composite macro score (-1 to +1)")
    key_indicators: dict = Field(default_factory=dict, description="Key macro indicator values")
    yield_curve: dict = Field(default_factory=dict, description="Yield curve analysis")


class RiskSignals(BaseModel):
    """Individual risk signal scores."""
    correlation_convergence: float = Field(default=0.0, ge=0.0, le=1.0)
    vol_regime_break: float = Field(default=0.0, ge=0.0, le=1.0)
    yield_curve_inversion: Any = Field(default=False, description="Bool or float score")
    credit_stress: float = Field(default=0.0, ge=0.0, le=1.0)
    contagion_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SystemicRisk(BaseModel):
    """Systemic risk assessment."""
    overall_risk_level: float = Field(..., ge=0.0, le=1.0, description="Aggregate risk level")
    risk_category: str = Field(..., description="low | moderate | elevated | high | critical")
    risk_signals: dict = Field(default_factory=dict, description="Individual risk signal scores")
    risk_assessment: str = Field(default="", description="Human-readable risk narrative")
    recommended_caution: bool = Field(default=False, description="Whether to recommend caution")


class CrossAssetAnalysis(BaseModel):
    """Cross-asset correlation and risk appetite analysis."""
    correlation_state: str = Field(default="normal", description="increasing | normal | decreasing")
    median_correlation: float = Field(default=0.0, description="Median cross-asset correlation")
    risk_appetite_index: float = Field(default=0.5, ge=0.0, le=1.0, description="0=max fear, 1=max greed")
    key_correlations: dict = Field(default_factory=dict, description="Key asset pair correlations")


class LLMReasoning(BaseModel):
    """LLM-generated qualitative analysis."""
    market_narrative: str = Field(default="", description="Institutional-quality market narrative")
    key_risks: list = Field(default_factory=list, description="Key identified risks")
    opportunities: list = Field(default_factory=list, description="Identified opportunities")
    asset_class_outlook: dict = Field(default_factory=dict, description="Per-asset-class outlook")
    sector_implications: dict = Field(default_factory=dict, description="Sector over/underweight suggestions")
    risk_budget_suggestion: dict = Field(default_factory=dict, description="Risk budget recommendations")
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM confidence in analysis")
    uncertainty_factors: list = Field(default_factory=list, description="Factors creating uncertainty")


class AgentMetadata(BaseModel):
    """Agent execution metadata."""
    agent_id: str = Field(default="agent1_macro_intelligence")
    version: str = Field(default="1.0.0")
    execution_time_ms: float = Field(default=0.0, description="Total execution time")
    llm_calls: int = Field(default=0, description="Number of LLM API calls")
    llm_total_latency_ms: float = Field(default=0.0, description="Total LLM latency")
    models_used: list[str] = Field(default_factory=list, description="ML/LLM models used")
    data_sources: list[str] = Field(default_factory=list, description="Data providers used")


class Agent1Output(BaseModel):
    """
    Complete Agent 1 output schema.
    This is the contract between Agent 1 and downstream agents (2, 3, 4).
    """
    timestamp: str = Field(..., description="UTC timestamp of analysis")
    data_freshness: str = Field(default="unknown", description="Timestamp of underlying data")

    market_regime: MarketRegime
    volatility_state: VolatilityState
    macro_environment: MacroEnvironment
    systemic_risk: SystemicRisk
    cross_asset_analysis: CrossAssetAnalysis
    llm_reasoning: LLMReasoning
    agent_metadata: AgentMetadata

    low_confidence_warning: Optional[str] = Field(None, description="Warning when confidence < threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-02-24T10:00:00Z",
                "data_freshness": "2026-02-24T09:55:00Z",
                "market_regime": {
                    "primary_regime": "high_volatility_bearish",
                    "confidence": 0.84,
                    "hmm_regime": "bear_high_vol",
                    "rf_regime": "bear_high_vol",
                    "models_agree": True,
                    "regime_duration_days": 12,
                    "transition_probability": 0.23,
                    "description": "Bearish crisis regime with elevated volatility"
                }
            }
        }


def validate_output(output: dict) -> tuple[bool, Optional[str]]:
    """
    Validate Agent 1 output against the schema.

    Returns:
        (is_valid, error_message)
    """
    try:
        Agent1Output(**output)
        return True, None
    except Exception as e:
        return False, str(e)
