"""
Hybrid Intelligence Portfolio System -- Agent 3 Output Schema
================================================================
Pydantic models for the Strategic Allocation & Optimization Agent.

Agent 3 consumes Agent 1 (market context) + Agent 2 (investor profile)
and outputs a complete portfolio allocation with:
  - Per-asset weights, expected returns, risk contributions
  - Portfolio-level metrics (Sharpe, drawdown, VaR, CVaR)
  - Monte Carlo simulation results (10K scenarios)
  - LLM-generated institutional-quality explanation
"""

from typing import Optional
from pydantic import BaseModel, Field


# ================================================================
#  ALLOCATION DETAILS
# ================================================================

class AssetAllocation(BaseModel):
    """Single asset allocation within the portfolio."""
    ticker: str = Field(..., description="Asset ticker symbol")
    asset_class: str = Field(..., description="equity | bond | commodity | crypto | cash")
    weight: float = Field(..., ge=0.0, le=1.0, description="Portfolio weight (0-1)")
    expected_return: float = Field(
        default=0.0, description="Annualized expected return"
    )
    expected_volatility: float = Field(
        default=0.0, ge=0.0, description="Annualized expected volatility"
    )
    risk_contribution: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of total portfolio risk from this asset"
    )
    rationale: str = Field(
        default="", description="Why this weight was chosen"
    )


class PortfolioMetrics(BaseModel):
    """Portfolio-level risk/return metrics."""
    expected_annual_return: float = Field(
        ..., description="Annualized expected return"
    )
    expected_annual_volatility: float = Field(
        ..., ge=0.0, description="Annualized portfolio volatility"
    )
    sharpe_ratio: float = Field(
        ..., description="Risk-adjusted return (excess return / volatility)"
    )
    sortino_ratio: float = Field(
        default=0.0, description="Downside-risk-adjusted return"
    )
    max_drawdown_estimate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Estimated maximum drawdown (0-1)"
    )
    value_at_risk_95: float = Field(
        default=0.0, description="5% VaR (worst 5% loss threshold)"
    )
    cvar_95: float = Field(
        default=0.0, description="Conditional VaR (expected loss beyond VaR)"
    )
    risk_free_rate: float = Field(
        default=0.04, description="Risk-free rate used in calculations"
    )
    diversification_ratio: float = Field(
        default=1.0, ge=1.0,
        description="Weighted avg vol / portfolio vol (>1 = diversification benefit)"
    )


class MonteCarloResult(BaseModel):
    """Monte Carlo simulation summary (10K scenarios)."""
    num_simulations: int = Field(default=10000, description="Number of scenarios")
    time_horizon_days: int = Field(default=252, description="Simulation horizon")
    percentile_returns: dict = Field(
        default_factory=dict,
        description="Return distribution: p5, p25, p50, p75, p95"
    )
    probability_of_loss: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="P(return < 0) over time horizon"
    )
    probability_of_severe_loss: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="P(drawdown > max_acceptable_drawdown)"
    )
    median_max_drawdown: float = Field(
        default=0.0, ge=0.0,
        description="Median max drawdown across simulations"
    )
    worst_case_return: float = Field(
        default=0.0, description="Worst 1% scenario return"
    )
    best_case_return: float = Field(
        default=0.0, description="Best 1% scenario return"
    )


class OptimizationDetails(BaseModel):
    """Details of the optimization process."""
    method_used: str = Field(
        ...,
        description="Optimization method: mean_variance | risk_parity | cvar_constrained | blended"
    )
    strategy_type: str = Field(
        ...,
        description="defensive | defensive_growth | balanced | growth | aggressive_growth | max_growth"
    )
    risk_aversion_parameter: float = Field(
        default=1.0, ge=0.0,
        description="Risk aversion lambda (higher = more conservative)"
    )
    constraints_applied: list[str] = Field(
        default_factory=list,
        description="Constraints active during optimization"
    )
    convergence_achieved: bool = Field(
        default=True, description="Whether optimizer converged"
    )
    alternatives_considered: dict = Field(
        default_factory=dict,
        description="Summary of MV, RP, CVaR allocations for comparison"
    )


class EvolutionMetrics(BaseModel):
    """Execution layer metrics: drift, turnover, and transaction costs."""
    requires_rebalance: bool = Field(
        default=True,
        description="Whether a rebalance trade is mathematically justified"
    )
    max_drift_detected: float = Field(
        default=0.0, ge=0.0,
        description="Maximum absolute weight deviation from current to target"
    )
    portfolio_turnover: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Estimated portfolio turnover (sum of absolute changes / 2)"
    )
    estimated_transaction_costs: float = Field(
        default=0.0, ge=0.0,
        description="Simulated transaction costs (e.g., 10 bps per unit of turnover)"
    )


class LLMExplanation(BaseModel):
    """LLM-generated institutional-quality allocation explanation."""
    allocation_rationale: str = Field(
        default="",
        description="2-3 paragraph explanation of why this allocation was chosen"
    )
    regime_impact: str = Field(
        default="",
        description="How the current market regime influenced the allocation"
    )
    risk_profile_alignment: str = Field(
        default="",
        description="How the allocation aligns with the investor's behavioral profile"
    )
    trade_offs: list[str] = Field(
        default_factory=list,
        description="Key trade-offs in this allocation"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Caveats and limitations"
    )
    rebalancing_triggers: list[str] = Field(
        default_factory=list,
        description="Conditions that should trigger portfolio rebalancing"
    )


class Agent3Output(BaseModel):
    """
    Complete Agent 3 output schema.
    Optimal portfolio allocation with full quantitative analysis.
    """
    timestamp: str = Field(..., description="UTC timestamp")
    session_id: str = Field(default="", description="Links to Agent 2 session")

    # Core outputs
    allocation: list[AssetAllocation] = Field(
        ..., min_length=1,
        description="Per-asset allocation details"
    )
    portfolio_metrics: PortfolioMetrics
    monte_carlo: MonteCarloResult
    optimization: OptimizationDetails
    evolution_metrics: EvolutionMetrics
    llm_explanation: LLMExplanation

    # Context consumed
    market_regime: str = Field(default="unknown")
    investor_risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    investor_behavioral_type: str = Field(default="moderate_balanced")
    investor_max_drawdown: float = Field(default=0.15, ge=0.0, le=1.0)

    # Metadata
    agent_metadata: dict = Field(default_factory=dict)


# ── Validators ─────────────────────────────────────────

def validate_agent3_output(output: dict) -> tuple[bool, Optional[str]]:
    """Validate Agent 3 output against schema."""
    try:
        model = Agent3Output(**output)
        # Verify weights sum to ~1.0
        total_weight = sum(a.weight for a in model.allocation)
        if abs(total_weight - 1.0) > 0.02:
            return False, f"Allocation weights sum to {total_weight:.4f}, expected ~1.0"
        return True, None
    except Exception as e:
        return False, str(e)
