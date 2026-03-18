"""
Hybrid Intelligence Portfolio System — Context Builder
========================================================
Assembles the final structured JSON output from Agent 1.
Merges ML model outputs with LLM reasoning, validates against
Pydantic schema, and produces the intelligence context consumed
by downstream agents (Agent 2 & 3).
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from config.settings import SystemMeta, RiskConfig
from llm.gemini_client import BaseLLMClient, LLMFactory
from llm.prompts import (
    AGENT1_SYSTEM_PROMPT,
    REGIME_INTERPRETATION_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    CONTEXT_SYNTHESIS_PROMPT,
    format_regime_data,
    format_feature_summary,
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds the final Agent 1 intelligence context by:
      1. Passing ML outputs through LLM for interpretation
      2. Running multi-stage LLM reasoning (regime → risk → synthesis)
      3. Merging quantitative ML outputs with qualitative LLM reasoning
      4. Validating final output structure
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client or LLMFactory.create()
        self._total_llm_calls = 0
        self._total_latency_ms = 0.0

    def build_context(
        self,
        regime_result: dict,
        volatility_state: dict,
        macro_analysis: dict,
        risk_assessment: dict,
        features: dict,
        vix_data: Optional[dict] = None,
        market_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Build the complete Agent 1 output context.

        This is a multi-stage LLM reasoning pipeline:
          Stage 1: Regime interpretation (validate + explain ML regime)
          Stage 2: Risk assessment (synthesize risk signals)
          Stage 3: Context synthesis (produce final unified output)

        All stages feed into the final JSON that downstream agents consume.

        Args:
            regime_result: From EnsembleRegimeDetector
            volatility_state: From VolatilityClassifier
            macro_analysis: From MacroAnalyzer
            risk_assessment: From SystemicRiskDetector
            features: From FeatureEngine
            vix_data: VIX DataFrame
            market_metadata: Market data fetch metadata

        Returns:
            Complete Agent 1 output JSON dict
        """
        logger.info("═" * 60)
        logger.info("CONTEXT BUILDER — Multi-stage LLM reasoning pipeline")
        logger.info("═" * 60)

        start_time = time.time()

        # ── Stage 1: Regime Interpretation ───────────────
        logger.info("Stage 1: LLM regime interpretation...")
        llm_regime = self._interpret_regime(regime_result, features, vix_data)

        # ── Stage 2: Risk Assessment ─────────────────────
        logger.info("Stage 2: LLM risk assessment...")
        llm_risk = self._assess_risk(risk_assessment, macro_analysis, volatility_state)

        # ── Stage 3: Context Synthesis ───────────────────
        logger.info("Stage 3: LLM context synthesis...")
        llm_synthesis = self._synthesize_context(
            llm_regime, llm_risk, macro_analysis,
            volatility_state, features
        )

        # ── Assemble Final Output ────────────────────────
        total_time_ms = (time.time() - start_time) * 1000

        output = self._assemble_output(
            regime_result=regime_result,
            volatility_state=volatility_state,
            macro_analysis=macro_analysis,
            risk_assessment=risk_assessment,
            llm_regime=llm_regime,
            llm_risk=llm_risk,
            llm_synthesis=llm_synthesis,
            features=features,
            market_metadata=market_metadata,
            total_time_ms=total_time_ms,
        )

        logger.info(f"Context built in {total_time_ms:.0f}ms ({self._total_llm_calls} LLM calls)")
        return output

    # ─────────────────────────────────────────────────
    #  LLM REASONING STAGES
    # ─────────────────────────────────────────────────

    def _interpret_regime(self, regime_result: dict, features: dict, vix_data) -> dict:
        """Stage 1: LLM interprets and validates ML regime detection output."""
        try:
            # Format VIX summary
            vix_summary = "VIX data not available"
            if vix_data is not None and isinstance(vix_data, dict):
                vix_close = vix_data.get("close")
                if vix_close is not None:
                    vix_summary = f"Current VIX: {float(vix_close.iloc[-1]):.2f}"

            prompt = REGIME_INTERPRETATION_PROMPT.format(
                regime_data=format_regime_data(regime_result),
                feature_summary=format_feature_summary(features),
                vix_summary=vix_summary,
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt=AGENT1_SYSTEM_PROMPT,
                json_mode=True,
            )
            self._total_llm_calls += 1
            self._total_latency_ms += response.get("latency_ms", 0)

            return self._parse_json_response(response["content"], "regime_interpretation")

        except Exception as e:
            logger.error(f"LLM regime interpretation failed: {e}")
            return {
                "market_narrative": f"Quantitative models indicate {regime_result.get('primary_regime', 'unknown')} regime.",
                "confidence_level": regime_result.get("confidence", 0.0),
                "key_drivers": [],
                "error": str(e),
            }

    def _assess_risk(self, risk_assessment: dict, macro_analysis: dict, volatility_state: dict) -> dict:
        """Stage 2: LLM synthesizes risk signals into actionable assessment."""
        try:
            prompt = RISK_ASSESSMENT_PROMPT.format(
                risk_data=json.dumps(risk_assessment.get("risk_signals", {}), indent=2),
                macro_data=json.dumps({
                    "regime": macro_analysis.get("macro_regime"),
                    "composite_score": macro_analysis.get("composite_score"),
                    "risk_factors": macro_analysis.get("risk_factors", []),
                    "key_indicators": macro_analysis.get("key_indicators", {}),
                }, indent=2),
                vol_data=json.dumps({
                    "state": volatility_state.get("current_state"),
                    "vix": volatility_state.get("vix_level"),
                    "trend": volatility_state.get("vol_trend"),
                    "percentile": volatility_state.get("realized_vol_percentile"),
                }, indent=2),
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt=AGENT1_SYSTEM_PROMPT,
                json_mode=True,
            )
            self._total_llm_calls += 1
            self._total_latency_ms += response.get("latency_ms", 0)

            return self._parse_json_response(response["content"], "risk_assessment")

        except Exception as e:
            logger.error(f"LLM risk assessment failed: {e}")
            return {
                "risk_narrative": risk_assessment.get("risk_assessment", "Risk assessment unavailable"),
                "key_risks": [],
                "confidence_level": 0.5,
                "error": str(e),
            }

    def _synthesize_context(
        self,
        llm_regime: dict,
        llm_risk: dict,
        macro_analysis: dict,
        volatility_state: dict,
        features: dict,
    ) -> dict:
        """Stage 3: Final context synthesis combining all intelligence."""
        try:
            # Correlation summary
            correlations = features.get("correlations", {})
            corr_matrix = correlations.get("matrix")
            corr_summary = {"median_correlation": correlations.get("median_correlation", 0)}
            if corr_matrix is not None and hasattr(corr_matrix, 'to_dict') and not corr_matrix.empty:
                # Get top correlations
                import numpy as np
                mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
                pairs = corr_matrix.where(mask).stack()
                if len(pairs) > 0:
                    top_5 = pairs.abs().nlargest(5)
                    corr_summary["top_correlations"] = {
                        f"{i[0]}-{i[1]}": round(float(pairs.loc[i]), 3)
                        for i in top_5.index
                    }

            prompt = CONTEXT_SYNTHESIS_PROMPT.format(
                regime_analysis=json.dumps(llm_regime, indent=2, default=str),
                risk_assessment=json.dumps(llm_risk, indent=2, default=str),
                macro_environment=json.dumps({
                    "regime": macro_analysis.get("macro_regime"),
                    "monetary": macro_analysis.get("monetary_policy_state"),
                    "inflation": macro_analysis.get("inflation_state"),
                    "growth": macro_analysis.get("growth_state"),
                    "composite_score": macro_analysis.get("composite_score"),
                }, indent=2),
                volatility_state=json.dumps({
                    "state": volatility_state.get("current_state"),
                    "vix": volatility_state.get("vix_level"),
                    "trend": volatility_state.get("vol_trend"),
                    "vol_of_vol": volatility_state.get("vol_of_vol"),
                }, indent=2),
                correlations=json.dumps(corr_summary, indent=2, default=str),
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt=AGENT1_SYSTEM_PROMPT,
                json_mode=True,
            )
            self._total_llm_calls += 1
            self._total_latency_ms += response.get("latency_ms", 0)

            return self._parse_json_response(response["content"], "context_synthesis")

        except Exception as e:
            logger.error(f"LLM context synthesis failed: {e}")
            return {
                "market_narrative": "Context synthesis unavailable due to LLM error.",
                "confidence_level": 0.3,
                "error": str(e),
            }

    # ─────────────────────────────────────────────────
    #  OUTPUT ASSEMBLY
    # ─────────────────────────────────────────────────

    def _assemble_output(
        self,
        regime_result: dict,
        volatility_state: dict,
        macro_analysis: dict,
        risk_assessment: dict,
        llm_regime: dict,
        llm_risk: dict,
        llm_synthesis: dict,
        features: dict,
        market_metadata: Optional[dict],
        total_time_ms: float,
    ) -> dict:
        """Assemble the final Agent 1 output JSON."""

        # Cross-asset correlation analysis
        correlations = features.get("correlations", {})
        corr_matrix = correlations.get("matrix")
        key_correlations = {}
        if corr_matrix is not None and hasattr(corr_matrix, 'empty') and not corr_matrix.empty:
            # Try to extract specific asset class correlations
            for pair in [("SPY", "TLT"), ("SPY", "BTCUSDT"), ("SPY", "GLD"), ("BTCUSDT", "ETHUSDT")]:
                a, b = pair
                if a in corr_matrix.columns and b in corr_matrix.columns:
                    key_correlations[f"{a}_{b}"] = round(float(corr_matrix.loc[a, b]), 3)

        # Compute confidence caveats
        overall_confidence = self._compute_overall_confidence(
            regime_result, macro_analysis, risk_assessment, llm_synthesis
        )

        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_freshness": market_metadata.get("timestamp", "unknown") if market_metadata else "unknown",

            "market_regime": {
                "primary_regime": regime_result.get("primary_regime", "unknown"),
                "confidence": regime_result.get("confidence", 0.0),
                "hmm_regime": regime_result.get("hmm_result", {}).get("current_regime", "unknown"),
                "rf_regime": regime_result.get("rf_result", {}).get("current_regime", "unknown"),
                "models_agree": regime_result.get("models_agree", False),
                "regime_duration_days": regime_result.get("regime_duration_days", 0),
                "transition_probability": regime_result.get("transition_probability", 0.0),
                "observations_count": regime_result.get("observations_count", 0),
                "convergence_warning": regime_result.get("convergence_warning", False),
                "adjusted_confidence": regime_result.get("adjusted_confidence", 0.0),
                "effective_risk_state": regime_result.get("effective_risk_state", 0.5),
                "description": regime_result.get("description", ""),
            },

            "volatility_state": {
                "current_state": volatility_state.get("current_state", "unknown"),
                "vix_level": volatility_state.get("vix_level"),
                "realized_vol_percentile": volatility_state.get("realized_vol_percentile", 0),
                "vol_trend": volatility_state.get("vol_trend", "unknown"),
                "vol_of_vol": volatility_state.get("vol_of_vol", "unknown"),
                "term_structure": volatility_state.get("term_structure", "unknown"),
            },

            "macro_environment": {
                "macro_regime": macro_analysis.get("macro_regime", "unknown"),
                "monetary_policy": macro_analysis.get("monetary_policy_state", "unknown"),
                "inflation_state": macro_analysis.get("inflation_state", "unknown"),
                "growth_state": macro_analysis.get("growth_state", "unknown"),
                "liquidity": macro_analysis.get("liquidity_state", "unknown"),
                "composite_score": macro_analysis.get("composite_score", 0.0),
                "key_indicators": macro_analysis.get("key_indicators", {}),
                "yield_curve": macro_analysis.get("yield_curve", {}),
            },

            "systemic_risk": {
                "overall_risk_level": risk_assessment.get("overall_risk_level", 0.0),
                "risk_category": risk_assessment.get("risk_category", "unknown"),
                "risk_signals": risk_assessment.get("risk_signals", {}),
                "risk_assessment": risk_assessment.get("risk_assessment", ""),
                "recommended_caution": risk_assessment.get("recommended_caution", False),
            },

            "cross_asset_analysis": {
                "correlation_state": "increasing" if correlations.get("median_correlation", 0) > 0.5 else "normal",
                "median_correlation": correlations.get("median_correlation", 0.0),
                "risk_appetite_index": self._compute_risk_appetite(
                    regime_result, volatility_state, risk_assessment
                ),
                "key_correlations": key_correlations,
            },

            "llm_reasoning": {
                "market_narrative": llm_synthesis.get(
                    "market_narrative",
                    llm_regime.get("market_narrative", "Analysis unavailable")
                ),
                "key_risks": llm_risk.get("key_risks", []),
                "opportunities": llm_risk.get("opportunities", []),
                "asset_class_outlook": llm_synthesis.get("asset_class_outlook", {}),
                "sector_implications": llm_synthesis.get("sector_implications", {}),
                "risk_budget_suggestion": llm_synthesis.get("risk_budget_suggestion", {}),
                "confidence_level": overall_confidence,
                "uncertainty_factors": llm_synthesis.get("uncertainty_factors", []),
            },

            "agent_metadata": {
                "agent_id": SystemMeta.AGENT_ID,
                "version": SystemMeta.VERSION,
                "execution_time_ms": round(total_time_ms, 0),
                "llm_calls": self._total_llm_calls,
                "llm_total_latency_ms": round(self._total_latency_ms, 0),
                "models_used": [
                    "hmm_regime_v1",
                    "rf_regime_v1",
                    "vol_classifier_v1",
                    "macro_analyzer_v1",
                    "risk_detector_v1",
                    self._llm._model_name if hasattr(self._llm, '_model_name') else "llm",
                ],
                "data_sources": market_metadata.get("data_sources", []) if market_metadata else [],
            },
        }

        # Low confidence warning
        if overall_confidence < RiskConfig.CONFIDENCE_THRESHOLD:
            output["low_confidence_warning"] = (
                f"Overall confidence ({overall_confidence:.0%}) is below the "
                f"{RiskConfig.CONFIDENCE_THRESHOLD:.0%} threshold. Results should be "
                f"interpreted with caution."
            )

        return output

    # ─────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(content: str, stage: str) -> dict:
        """Safely parse LLM JSON response."""
        try:
            # Clean potential markdown wrapping
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Sometimes LLMs return extra text before/after JSON
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx+1]

            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON ({stage}): {e}")
            logger.debug(f"Raw content: {content}")
            return {"raw_content": content, "parse_error": str(e)}

    @staticmethod
    def _compute_overall_confidence(
        regime: dict, macro: dict, risk: dict, synthesis: dict
    ) -> float:
        """Compute weighted overall confidence score."""
        confidences = [
            regime.get("confidence", 0.0),
            macro.get("confidence", 0.0),
            risk.get("confidence", 0.0),
            synthesis.get("confidence_level", 0.0),
        ]
        valid = [c for c in confidences if c > 0]
        if not valid:
            return 0.3
        return round(float(sum(valid) / len(valid)), 4)

    @staticmethod
    def _compute_risk_appetite(regime: dict, vol: dict, risk: dict) -> float:
        """
        Compute risk appetite index (0 = max fear, 1 = max greed).
        Inverse of overall risk signals.
        """
        risk_level = risk.get("overall_risk_level", 0.5)
        vol_state = vol.get("current_state", "normal")

        # Base: inverse of risk
        appetite = 1.0 - risk_level

        # Vol adjustment
        vol_adjustments = {
            "extremely_low": 0.1,
            "low": 0.05,
            "normal": 0.0,
            "elevated": -0.1,
            "extreme": -0.2,
        }
        appetite += vol_adjustments.get(vol_state, 0.0)

        # Regime adjustment
        regime_name = regime.get("primary_regime", "")
        if "bull" in regime_name:
            appetite += 0.05
        elif "bear" in regime_name:
            appetite -= 0.05

        return round(max(0.0, min(1.0, appetite)), 4)
