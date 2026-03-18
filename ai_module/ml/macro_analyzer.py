"""
Hybrid Intelligence Portfolio System — Macro Environment Analyzer
==================================================================
Synthesizes macroeconomic data from FRED into a regime classification.
Evaluates monetary policy, inflation, growth, liquidity, and labor
to produce a composite macro environment score and state.
"""

import logging
from typing import Optional

import numpy as np

from config.settings import RiskConfig

logger = logging.getLogger(__name__)


class MacroAnalyzer:
    """
    Analyzes macroeconomic environment from FRED data snapshot.
    Produces a composite score and classification consumed by the
    LLM reasoning layer and downstream agents.

    Macro States:
      - risk_on_expansion: Growth strong, policy accommodative
      - stable_growth: Balanced macro environment
      - late_cycle: Growth peaking, policy tightening
      - stagflation: Inflation high, growth stalling
      - recession: Contraction with policy easing
      - recovery: Early cycle improvement
    """

    MACRO_STATES = [
        "risk_on_expansion",
        "stable_growth",
        "late_cycle",
        "stagflation",
        "recession",
        "recovery",
    ]

    @classmethod
    def analyze(cls, macro_snapshot: dict) -> dict:
        """
        Produce comprehensive macro environment analysis.

        Args:
            macro_snapshot: Output from MacroDataFetcher.compute_macro_snapshot()

        Returns:
            {
                "macro_regime": str,
                "monetary_policy_state": str,
                "inflation_state": str,
                "growth_state": str,
                "liquidity_state": str,
                "labor_state": str,
                "yield_curve": dict,
                "composite_score": float (-1 to +1, negative = contractionary),
                "key_indicators": dict,
                "risk_factors": list,
                "confidence": float,
            }
        """
        logger.info("Analyzing macroeconomic environment...")

        monetary = macro_snapshot.get("monetary_policy", {})
        inflation = macro_snapshot.get("inflation", {})
        growth = macro_snapshot.get("growth", {})
        liquidity = macro_snapshot.get("liquidity", {})
        labor = macro_snapshot.get("labor", {})
        yield_curve = macro_snapshot.get("derived_indicators", {}).get("yield_curve", {})
        current_values = macro_snapshot.get("current_values", {})

        # ── Extract States ───────────────────────────────
        monetary_state = monetary.get("state", "unknown")
        inflation_state = inflation.get("state", "unknown")
        growth_state = growth.get("state", "unknown")
        liquidity_state = liquidity.get("state", "unknown")
        labor_state = labor.get("state", "unknown")

        # ── Compute Composite Score ──────────────────────
        component_scores = cls._compute_component_scores(
            monetary_state, inflation_state, growth_state,
            liquidity_state, labor_state, yield_curve
        )
        composite = cls._compute_composite(component_scores)

        # ── Determine Macro Regime ───────────────────────
        macro_regime = cls._determine_regime(
            growth_state, inflation_state, monetary_state,
            liquidity_state, composite
        )

        # ── Identify Risk Factors ────────────────────────
        risk_factors = cls._identify_risk_factors(
            monetary, inflation, growth, liquidity, labor, yield_curve
        )

        # ── Extract Key Indicators ───────────────────────
        key_indicators = cls._extract_key_indicators(current_values)

        # ── Confidence Calculation ───────────────────────
        confidences = [
            monetary.get("confidence", 0),
            inflation.get("confidence", 0),
            growth.get("confidence", 0),
            liquidity.get("confidence", 0),
            labor.get("confidence", 0),
        ]
        valid_confs = [c for c in confidences if c > 0]
        avg_confidence = float(np.mean(valid_confs)) if valid_confs else 0.3

        # Flag low confidence
        if avg_confidence < RiskConfig.CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Macro analysis confidence ({avg_confidence:.2%}) below threshold "
                f"({RiskConfig.CONFIDENCE_THRESHOLD:.0%}). Results may be unreliable."
            )

        result = {
            "macro_regime": macro_regime,
            "monetary_policy_state": monetary_state,
            "inflation_state": inflation_state,
            "growth_state": growth_state,
            "liquidity_state": liquidity_state,
            "labor_state": labor_state,
            "yield_curve": yield_curve,
            "composite_score": round(composite, 4),
            "component_scores": component_scores,
            "key_indicators": key_indicators,
            "risk_factors": risk_factors,
            "confidence": round(avg_confidence, 4),
            "low_confidence_warning": avg_confidence < RiskConfig.CONFIDENCE_THRESHOLD,
        }

        logger.info(f"Macro regime: {macro_regime} (composite: {composite:+.2f}, confidence: {avg_confidence:.2%})")
        return result

    @classmethod
    def _compute_component_scores(
        cls,
        monetary: str,
        inflation: str,
        growth: str,
        liquidity: str,
        labor: str,
        yield_curve: dict,
    ) -> dict:
        """
        Score each macro component on a -1 to +1 scale.
        Negative = contractionary / risk-off.
        Positive = expansionary / risk-on.
        """
        scores = {}

        # Monetary policy score
        monetary_map = {
            "aggressive_easing": 0.9,
            "easing": 0.5,
            "neutral_hold": 0.0,
            "tightening": -0.5,
            "aggressive_tightening": -0.9,
        }
        scores["monetary"] = monetary_map.get(monetary, 0.0)

        # Inflation score (moderate is best for risk assets)
        inflation_map = {
            "deflation": -0.8,
            "low_inflation": 0.3,
            "target_range": 0.5,
            "above_target": -0.2,
            "elevated": -0.5,
            "high_inflation": -0.9,
        }
        scores["inflation"] = inflation_map.get(inflation, 0.0)

        # Growth score
        growth_map = {
            "recession": -0.9,
            "stagnation": -0.4,
            "moderate_growth": 0.5,
            "strong_growth": 0.8,
            "overheating": -0.2,
        }
        scores["growth"] = growth_map.get(growth, 0.0)

        # Liquidity score
        liquidity_map = {
            "contraction": -0.9,
            "tight": -0.4,
            "neutral": 0.1,
            "accommodative": 0.5,
            "flood": 0.3,  # Can signal overheating
        }
        scores["liquidity"] = liquidity_map.get(liquidity, 0.0)

        # Labor score
        labor_map = {
            "tight_labor_market": 0.5,
            "healthy": 0.4,
            "deteriorating": -0.5,
            "elevated_unemployment": -0.7,
        }
        scores["labor"] = labor_map.get(labor, 0.0)

        # Yield curve penalty
        if yield_curve.get("inverted", False):
            scores["yield_curve"] = -0.6
        elif yield_curve.get("signal") == "deepening_inversion":
            scores["yield_curve"] = -0.8
        elif yield_curve.get("signal") == "un-inverting":
            scores["yield_curve"] = 0.3
        else:
            scores["yield_curve"] = 0.2

        return scores

    @classmethod
    def _compute_composite(cls, component_scores: dict) -> float:
        """
        Compute weighted composite score.
        Weights reflect relative importance for market impact.
        """
        weights = {
            "monetary": 0.25,
            "inflation": 0.15,
            "growth": 0.25,
            "liquidity": 0.15,
            "labor": 0.10,
            "yield_curve": 0.10,
        }

        composite = 0.0
        total_weight = 0.0
        for component, score in component_scores.items():
            weight = weights.get(component, 0.1)
            composite += score * weight
            total_weight += weight

        if total_weight > 0:
            composite /= total_weight

        return max(-1.0, min(1.0, composite))

    @classmethod
    def _determine_regime(
        cls,
        growth: str,
        inflation: str,
        monetary: str,
        liquidity: str,
        composite: float,
    ) -> str:
        """Determine macro regime from component states."""

        # Recession
        if growth in {"recession"} or (growth == "stagnation" and composite < -0.4):
            return "recession"

        # Stagflation: stagnant growth + high inflation
        if growth in {"stagnation", "recession"} and inflation in {"elevated", "high_inflation"}:
            return "stagflation"

        # Late cycle: tightening + growth peaking
        if monetary in {"tightening", "aggressive_tightening"} and growth in {"strong_growth", "overheating"}:
            return "late_cycle"

        # Recovery: easing + growth improving
        if monetary in {"easing", "aggressive_easing"} and growth in {"stagnation", "moderate_growth"}:
            return "recovery"

        # Risk-on expansion
        if composite > 0.3 and growth in {"strong_growth", "moderate_growth"}:
            return "risk_on_expansion"

        # Default: stable growth
        return "stable_growth"

    @classmethod
    def _identify_risk_factors(cls, monetary, inflation, growth, liquidity, labor, yield_curve) -> list[str]:
        """Identify active macro risk factors."""
        risks = []

        if monetary.get("state") in {"aggressive_tightening", "tightening"}:
            risks.append("Monetary policy tightening — negative for risk assets and valuations")

        if inflation.get("state") in {"elevated", "high_inflation"}:
            risks.append(f"Elevated inflation ({inflation.get('details', {}).get('cpi_yoy_pct', 'N/A')}% YoY) — eroding purchasing power")

        if inflation.get("details", {}).get("inflation_momentum") == "accelerating":
            risks.append("Inflation momentum accelerating — may force more aggressive policy")

        if growth.get("state") in {"recession", "stagnation"}:
            risks.append("Economic growth weakness — earnings risk")

        if yield_curve.get("inverted"):
            spread = yield_curve.get("spreads", {}).get("10y_2y", "N/A")
            risks.append(f"Yield curve inverted (10Y-2Y: {spread}%) — historic recession predictor")

        if liquidity.get("state") in {"contraction", "tight"}:
            risks.append("Liquidity contraction — tightening financial conditions")

        if liquidity.get("details", {}).get("credit_stress"):
            risks.append("Credit spreads widening — financial stress signal")

        if labor.get("state") == "deteriorating":
            risks.append("Labor market deteriorating — consumer spending risk")

        return risks

    @classmethod
    def _extract_key_indicators(cls, current_values: dict) -> dict:
        """Extract current values of key macro indicators."""
        keys = [
            "fed_funds_rate", "treasury_10y", "treasury_2y",
            "cpi_yoy", "unemployment", "gdp_growth", "m2_money_supply",
            "consumer_sentiment", "credit_spread_hy",
        ]
        extracted = {}
        for key in keys:
            if key in current_values:
                extracted[key] = current_values[key].get("value")
        return extracted
