"""
Hybrid Intelligence Portfolio System -- Allocation Explainer
================================================================
LLM-powered explanation layer for Agent 3 portfolio allocations.

Translates quantitative optimization output into institutional-quality
human language, following the prompt:
  "You are a Quantitative Portfolio Strategist.
   Be precise. No vague motivational language.
   Explain trade-offs clearly. Do not overpromise performance."
"""

import json
import logging
from typing import Optional

from config.settings import APIKeys
from llm.gemini_client import BaseLLMClient, LLMFactory

logger = logging.getLogger(__name__)

# ================================================================
#  PROMPTS
# ================================================================

STRATEGIST_SYSTEM_PROMPT = """You are a Quantitative Portfolio Strategist operating within 
an institutional-grade multi-agent portfolio management system.

RULES:
1. Be precise. Reference specific numbers, percentages, and metrics.
2. No vague motivational language. No "exciting opportunities" or "promising outlook."
3. Explain trade-offs clearly: what was sacrificed for what benefit.
4. Do not overpromise performance. All returns are expectations, not guarantees.
5. Reference the specific market regime, risk profile, and correlation data.
6. Output strictly valid JSON. No markdown, no extra text."""


ALLOCATION_EXPLANATION_PROMPT = """Explain this portfolio allocation decision.

## PORTFOLIO ALLOCATION:
{allocation_data}

## PORTFOLIO METRICS:
{metrics_data}

## MONTE CARLO SIMULATION (10K scenarios):
{monte_carlo_data}

## OPTIMIZATION DETAILS:
{optimization_data}

## MARKET CONTEXT (from Agent 1):
{market_context}

## INVESTOR PROFILE (from Agent 2):
{investor_profile}

Provide your analysis in this JSON format:
{{
    "allocation_rationale": "2-3 paragraphs explaining WHY this allocation was chosen. Reference specific regime conditions, risk parameters, and correlation data. Explain the optimization methodology and its implications.",
    "regime_impact": "How the current market regime specifically affected each asset weight. Reference VIX, correlations, and regime indicators.",
    "risk_profile_alignment": "How this allocation maps to the investor's behavioral type, risk score, max drawdown tolerance, and liquidity preference.",
    "trade_offs": [
        "Specific trade-off 1 with numbers (e.g., 'Reducing BTC from 15% to 8% lowers expected return by 1.2% but reduces max drawdown by 4%')",
        "Specific trade-off 2"
    ],
    "caveats": [
        "Specific caveat about model limitations",
        "Caveat about market condition assumptions"
    ],
    "rebalancing_triggers": [
        "Specific condition that should trigger rebalancing (e.g., 'VIX crosses above 30')",
        "Another trigger with specific thresholds"
    ]
}}

Be precise with numbers. No motivational language. Explain trade-offs clearly."""


class AllocationExplainer:
    """
    LLM-powered allocation explanation generator.

    Translates quant output into institutional-quality narrative.
    Falls back to quantitative-only explanation if LLM unavailable.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client or LLMFactory.create()
        self._total_calls = 0
        self._total_latency_ms = 0.0

    @property
    def llm_calls(self) -> int:
        return self._total_calls

    @property
    def llm_latency_ms(self) -> float:
        return self._total_latency_ms

    def explain(
        self,
        allocation: list[dict],
        metrics: dict,
        monte_carlo: dict,
        optimization: dict,
        agent1_output: dict,
        agent2_output: dict,
    ) -> dict:
        """
        Generate institutional-quality allocation explanation.

        Args:
            allocation: Per-asset allocation list
            metrics: Portfolio metrics dict
            monte_carlo: Monte Carlo results dict
            optimization: Optimization details dict
            agent1_output: Agent 1 context
            agent2_output: Agent 2 profile

        Returns:
            LLMExplanation-compatible dict
        """
        # Try LLM first
        if self._llm.is_available():
            try:
                return self._explain_via_llm(
                    allocation, metrics, monte_carlo, optimization,
                    agent1_output, agent2_output
                )
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}")

        # Fallback: quantitative-only explanation
        return self._generate_fallback(
            allocation, metrics, monte_carlo, optimization,
            agent1_output, agent2_output
        )

    # ────────────────────────────────────────────────────
    #  LLM EXPLANATION
    # ────────────────────────────────────────────────────

    def _explain_via_llm(
        self,
        allocation: list[dict],
        metrics: dict,
        monte_carlo: dict,
        optimization: dict,
        agent1_output: dict,
        agent2_output: dict,
    ) -> dict:
        """Generate explanation via LLM."""
        # Format context
        market_ctx = self._format_market_context(agent1_output)
        investor_ctx = self._format_investor_profile(agent2_output)

        prompt = ALLOCATION_EXPLANATION_PROMPT.format(
            allocation_data=json.dumps(allocation, indent=2, default=str),
            metrics_data=json.dumps(metrics, indent=2, default=str),
            monte_carlo_data=json.dumps(monte_carlo, indent=2, default=str),
            optimization_data=json.dumps(optimization, indent=2, default=str),
            market_context=market_ctx,
            investor_profile=investor_ctx,
        )

        response = self._llm.generate(
            prompt=prompt,
            system_prompt=STRATEGIST_SYSTEM_PROMPT,
            json_mode=True,
        )
        self._total_calls += 1
        self._total_latency_ms += response.get("latency_ms", 0)

        return self._parse_json(response["content"])

    # ────────────────────────────────────────────────────
    #  FALLBACK EXPLANATION
    # ────────────────────────────────────────────────────

    def _generate_fallback(
        self,
        allocation: list[dict],
        metrics: dict,
        monte_carlo: dict,
        optimization: dict,
        agent1_output: dict,
        agent2_output: dict,
    ) -> dict:
        """Generate quantitative-only fallback explanation."""
        regime = agent1_output.get("market_regime", {}).get("primary_regime", "unknown")
        strategy = optimization.get("strategy_type", "unknown")
        method = optimization.get("method_used", "unknown")

        # Build allocation summary
        alloc_summary = ", ".join(
            f"{a.get('ticker', '?')}: {a.get('weight', 0):.0%}"
            for a in allocation
        )

        risk_score = agent2_output.get("risk_classification", agent2_output).get("risk_score", 0.5)
        beh_type = agent2_output.get("risk_classification", agent2_output).get("behavioral_type", "moderate")
        max_dd = agent2_output.get("risk_classification", agent2_output).get("max_acceptable_drawdown", 0.15)

        exp_ret = metrics.get("expected_annual_return", 0)
        exp_vol = metrics.get("expected_annual_volatility", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        mc_loss_prob = monte_carlo.get("probability_of_loss", 0)
        mc_median_dd = monte_carlo.get("median_max_drawdown", 0)

        rationale = (
            f"This {strategy} portfolio ({alloc_summary}) was constructed using "
            f"{method} optimization under a {regime} market regime. "
            f"The allocation targets an expected annual return of {exp_ret:.1%} "
            f"with {exp_vol:.1%} volatility (Sharpe ratio: {sharpe:.2f}). "
            f"Monte Carlo simulation across 10,000 scenarios projects a "
            f"{mc_loss_prob:.0%} probability of loss over the 1-year horizon, "
            f"with median maximum drawdown of {mc_median_dd:.1%}."
        )

        regime_impact = (
            f"The {regime} regime influenced asset weights by "
            f"{'increasing defensive allocations (bonds, gold, cash)' if 'bear' in regime else 'maintaining growth exposure (equity, crypto)'}"
            f". Correlation structure and volatility state were incorporated "
            f"into the covariance matrix used for optimization."
        )

        risk_alignment = (
            f"Allocation designed for a {beh_type} investor with "
            f"risk score {risk_score:.2f} and maximum acceptable drawdown of {max_dd:.0%}. "
            f"Simulated median max drawdown ({mc_median_dd:.1%}) {'aligns with' if mc_median_dd <= max_dd else 'exceeds'} "
            f"the investor's stated tolerance."
        )

        return {
            "allocation_rationale": rationale,
            "regime_impact": regime_impact,
            "risk_profile_alignment": risk_alignment,
            "trade_offs": [
                f"Higher equity ({allocation[0].get('weight', 0):.0%} SPY) increases expected return but raises drawdown risk",
                f"Cash allocation ({allocation[-1].get('weight', 0) if allocation else 0:.0%}) provides liquidity but reduces long-term compounding",
            ],
            "caveats": [
                "Expected returns are based on historical data and macro-adjusted estimates, not forecasts",
                "Covariance matrix assumes stationarity; regime changes may alter correlation structure",
                "Monte Carlo simulation uses Geometric Brownian Motion which may underestimate tail risks",
            ],
            "rebalancing_triggers": [
                f"Market regime changes from {regime} (transition probability monitored by Agent 1)",
                "Any single asset drifts more than 5% from target weight",
                "VIX crosses above 30 (elevated stress) or below 12 (complacency)",
            ],
        }

    # ────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────

    @staticmethod
    def _format_market_context(agent1: dict) -> str:
        regime = agent1.get("market_regime", {})
        vol = agent1.get("volatility_state", {})
        risk = agent1.get("systemic_risk", {})
        cross = agent1.get("cross_asset_analysis", {})
        macro = agent1.get("macro_environment", {})
        return (
            f"Regime: {regime.get('primary_regime', 'N/A')} (confidence: {regime.get('confidence', 0):.0%})\n"
            f"Volatility: {vol.get('current_state', 'N/A')} (VIX: {vol.get('vix_level', 'N/A')})\n"
            f"Systemic Risk: {risk.get('risk_category', 'N/A')} ({risk.get('overall_risk_level', 0):.2f})\n"
            f"Median Correlation: {cross.get('median_correlation', 0):.3f}\n"
            f"Key Correlations: {json.dumps(cross.get('key_correlations', {}))}\n"
            f"Macro Regime: {macro.get('macro_regime', 'N/A')}, "
            f"Monetary: {macro.get('monetary_policy', 'N/A')}, "
            f"Inflation: {macro.get('inflation_state', 'N/A')}"
        )

    @staticmethod
    def _format_investor_profile(agent2: dict) -> str:
        # Handle both full agent2 output and nested profile
        risk_cls = agent2.get("risk_classification", agent2)
        beh_prof = agent2.get("behavioral_profile", {})
        return (
            f"Risk Score: {risk_cls.get('risk_score', 'N/A')}\n"
            f"Behavioral Type: {risk_cls.get('behavioral_type', 'N/A')}\n"
            f"Max Acceptable Drawdown: {risk_cls.get('max_acceptable_drawdown', 'N/A')}\n"
            f"Liquidity Preference: {risk_cls.get('liquidity_preference', 'N/A')}\n"
            f"Time Horizon: {risk_cls.get('time_horizon', 'N/A')}\n"
            f"Consistency: {beh_prof.get('consistency_score', 'N/A')}\n"
            f"Emotional Stability: {beh_prof.get('emotional_stability', 'N/A')}\n"
            f"Stress Response: {beh_prof.get('stress_response_pattern', 'N/A')}"
        )

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse LLM JSON response safely."""
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return {"allocation_rationale": content}
