"""
Hybrid Intelligence Portfolio System — Systemic Risk Detector
===============================================================
Detects systemic risk signals across financial markets using:
  1. Correlation convergence (crisis indicator)
  2. Volatility regime breaks (sudden vol spikes)
  3. Yield curve inversion analysis
  4. Credit stress monitoring
  5. Cross-market contagion scoring

Produces an aggregate risk level (0.0 – 1.0) with component breakdown.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import RiskConfig

logger = logging.getLogger(__name__)


class SystemicRiskDetector:
    """
    Monitors multiple dimensions of systemic risk to provide
    an early warning system for portfolio-threatening events.

    Risk Level Scale:
      0.0 – 0.30: Low risk        — normal market conditions
      0.30 – 0.50: Moderate risk   — some stress signals, monitor
      0.50 – 0.70: Elevated risk   — active stress, reduce exposure
      0.70 – 0.85: High risk       — crisis conditions likely
      0.85 – 1.00: Critical risk   — systemic crisis, max defensiveness
    """

    @classmethod
    def detect(
        cls,
        features: dict,
        macro_analysis: dict,
        volatility_state: dict,
    ) -> dict:
        """
        Run full systemic risk assessment.

        Args:
            features: Output from FeatureEngine.build_features()
            macro_analysis: Output from MacroAnalyzer.analyze()
            volatility_state: Output from VolatilityClassifier.classify()

        Returns:
            {
                "overall_risk_level": float (0-1),
                "risk_category": str,
                "risk_signals": {
                    "correlation_convergence": float,
                    "vol_regime_break": float,
                    "yield_curve_inversion": bool,
                    "credit_stress": float,
                    "contagion_score": float,
                },
                "risk_assessment": str,
                "recommended_caution": bool,
                "component_details": dict,
                "confidence": float,
            }
        """
        logger.info("═" * 60)
        logger.info("SYSTEMIC RISK DETECTOR — Scanning for risk signals")
        logger.info("═" * 60)

        signals = {}
        details = {}

        # ── 1. Correlation Convergence ───────────────────
        corr_result = cls._assess_correlation_convergence(features)
        signals["correlation_convergence"] = corr_result["score"]
        details["correlation_convergence"] = corr_result

        # ── 2. Volatility Regime Break ───────────────────
        vol_result = cls._assess_volatility_break(features, volatility_state)
        signals["vol_regime_break"] = vol_result["score"]
        details["vol_regime_break"] = vol_result

        # ── 3. Yield Curve Inversion ─────────────────────
        yc_result = cls._assess_yield_curve(macro_analysis)
        signals["yield_curve_inversion"] = yc_result["inverted"]
        details["yield_curve"] = yc_result

        # ── 4. Credit Stress ─────────────────────────────
        credit_result = cls._assess_credit_stress(macro_analysis)
        signals["credit_stress"] = credit_result["score"]
        details["credit_stress"] = credit_result

        # ── 5. Contagion Score ───────────────────────────
        contagion_result = cls._assess_contagion(
            corr_result, vol_result, yc_result, credit_result
        )
        signals["contagion_score"] = contagion_result["score"]
        details["contagion"] = contagion_result

        # ── Aggregate Risk Level ─────────────────────────
        overall_risk = cls._compute_overall_risk(signals)

        # Risk category
        if overall_risk < RiskConfig.RISK_LEVEL_LOW:
            category = "low"
        elif overall_risk < RiskConfig.RISK_LEVEL_MODERATE:
            category = "moderate"
        elif overall_risk < RiskConfig.RISK_LEVEL_ELEVATED:
            category = "elevated"
        elif overall_risk < RiskConfig.RISK_LEVEL_CRITICAL:
            category = "high"
        else:
            category = "critical"

        # Risk assessment narrative
        assessment = cls._generate_assessment(signals, category, details)

        # Recommended caution
        recommended_caution = overall_risk >= RiskConfig.RISK_LEVEL_MODERATE

        # Confidence
        confidence = cls._compute_confidence(details)

        result = {
            "overall_risk_level": round(overall_risk, 4),
            "risk_category": category,
            "risk_signals": {k: round(v, 4) if isinstance(v, float) else v for k, v in signals.items()},
            "risk_assessment": assessment,
            "recommended_caution": recommended_caution,
            "component_details": details,
            "confidence": round(confidence, 4),
        }

        logger.info(f"Risk level: {overall_risk:.2f} ({category}) — Caution: {recommended_caution}")
        return result

    # ─────────────────────────────────────────────────
    #  RISK SIGNAL ASSESSORS
    # ─────────────────────────────────────────────────

    @classmethod
    def _assess_correlation_convergence(cls, features: dict) -> dict:
        """
        Detect correlation convergence: when all assets become highly correlated,
        it signals crisis conditions (diversification failure).
        """
        correlations = features.get("correlations", {})
        median_corr = correlations.get("median_correlation", 0.0)
        corr_matrix = correlations.get("matrix", pd.DataFrame())

        # Score: 0 if median corr is typical, 1 if extreme convergence
        threshold = RiskConfig.CORRELATION_CONVERGENCE_THRESHOLD
        if median_corr >= threshold:
            score = min(1.0, 0.5 + (median_corr - threshold) / (1 - threshold))
        elif median_corr >= threshold * 0.7:
            score = 0.3 + (median_corr - threshold * 0.7) / (threshold - threshold * 0.7) * 0.2
        else:
            score = max(0.0, median_corr / threshold * 0.3)

        # Check for specific high-risk cross-correlations
        high_corr_pairs = []
        if not corr_matrix.empty:
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.8:
                        high_corr_pairs.append({
                            "pair": f"{corr_matrix.index[i]}-{corr_matrix.columns[j]}",
                            "correlation": round(float(val), 3),
                        })

        return {
            "score": round(score, 4),
            "median_correlation": round(median_corr, 4),
            "threshold": threshold,
            "above_threshold": median_corr >= threshold,
            "high_correlation_pairs": high_corr_pairs[:10],  # Top 10
            "signal": "convergence_detected" if median_corr >= threshold else "normal_dispersion",
        }

    @classmethod
    def _assess_volatility_break(cls, features: dict, vol_state: dict) -> dict:
        """
        Detect sudden volatility regime breaks: sharp spikes that
        indicate market shock or panic.
        """
        vol_data = features.get("volatility", {})
        vol_zscore = vol_data.get("vol_zscore")

        current_zscore = 0.0
        if vol_zscore is not None and isinstance(vol_zscore, pd.Series) and not vol_zscore.empty:
            current_zscore = float(vol_zscore.iloc[-1])

        threshold = RiskConfig.VOL_BREAK_ZSCORE
        vix_level = vol_state.get("vix_level")

        # Score based on z-score
        if current_zscore >= threshold:
            score = min(1.0, 0.6 + (current_zscore - threshold) / 3)
        elif current_zscore >= threshold * 0.6:
            score = 0.2 + (current_zscore - threshold * 0.6) / (threshold - threshold * 0.6) * 0.4
        else:
            score = max(0.0, current_zscore / threshold * 0.2) if current_zscore > 0 else 0.0

        # VIX contribution
        if vix_level is not None:
            if vix_level > 35:
                score = max(score, 0.8)
            elif vix_level > 28:
                score = max(score, 0.5)

        vol_trend = vol_state.get("vol_trend", "stable")
        if vol_trend in {"sharply_increasing"} and score < 0.5:
            score = max(score, 0.4)

        return {
            "score": round(score, 4),
            "current_zscore": round(current_zscore, 3),
            "threshold": threshold,
            "break_detected": current_zscore >= threshold,
            "vix_level": vix_level,
            "vol_trend": vol_trend,
            "signal": "vol_break_detected" if current_zscore >= threshold else "vol_normal",
        }

    @classmethod
    def _assess_yield_curve(cls, macro_analysis: dict) -> dict:
        """Assess yield curve inversion risk."""
        yc = macro_analysis.get("yield_curve", {})
        inverted = yc.get("inverted", False)
        spreads = yc.get("spreads", {})
        signal = yc.get("signal", "normal")

        spread_10y2y = spreads.get("10y_2y")
        spread_10y3m = spreads.get("10y_3m")

        # Score
        if signal == "deepening_inversion":
            score = 0.9
        elif inverted:
            score = 0.7
        elif signal == "inverted_but_improving":
            score = 0.5
        elif signal == "un-inverting":
            score = 0.3
        else:
            score = 0.1

        return {
            "inverted": inverted,
            "score": round(score, 4),
            "spread_10y2y": spread_10y2y,
            "spread_10y3m": spread_10y3m,
            "signal": signal,
        }

    @classmethod
    def _assess_credit_stress(cls, macro_analysis: dict) -> dict:
        """Assess credit market stress from high-yield spreads."""
        key_indicators = macro_analysis.get("key_indicators", {})
        hy_spread = key_indicators.get("credit_spread_hy")

        if hy_spread is None:
            return {"score": 0.0, "hy_spread": None, "stress_level": "unknown"}

        threshold = RiskConfig.CREDIT_STRESS_THRESHOLD_BPS / 100  # Convert to percentage

        if hy_spread >= threshold:
            score = min(1.0, 0.7 + (hy_spread - threshold) / 5)
            stress_level = "severe"
        elif hy_spread >= threshold * 0.7:
            score = 0.4 + (hy_spread - threshold * 0.7) / (threshold - threshold * 0.7) * 0.3
            stress_level = "elevated"
        elif hy_spread >= threshold * 0.5:
            score = 0.15 + (hy_spread - threshold * 0.5) / (threshold - threshold * 0.5) * 0.25
            stress_level = "moderate"
        else:
            score = max(0.0, hy_spread / (threshold * 0.5) * 0.15)
            stress_level = "low"

        return {
            "score": round(score, 4),
            "hy_spread": round(hy_spread, 2) if hy_spread else None,
            "threshold": round(threshold, 2),
            "stress_level": stress_level,
        }

    @classmethod
    def _assess_contagion(
        cls,
        corr_result: dict,
        vol_result: dict,
        yc_result: dict,
        credit_result: dict,
    ) -> dict:
        """
        Compute contagion score: measures how multiple risk signals
        are amplifying each other (non-linear risk escalation).
        """
        component_scores = [
            corr_result["score"],
            vol_result["score"],
            yc_result["score"],
            credit_result["score"],
        ]

        # Count how many signals are elevated (> 0.5)
        elevated_count = sum(1 for s in component_scores if s > 0.5)
        avg_score = float(np.mean(component_scores))

        # Contagion is non-linear: multiple elevated signals = exponential risk
        if elevated_count >= 4:
            contagion = min(1.0, avg_score * 1.4)
        elif elevated_count >= 3:
            contagion = min(1.0, avg_score * 1.25)
        elif elevated_count >= 2:
            contagion = min(1.0, avg_score * 1.1)
        else:
            contagion = avg_score * 0.8

        return {
            "score": round(contagion, 4),
            "elevated_signal_count": elevated_count,
            "average_signal": round(avg_score, 4),
            "amplification_factor": round(contagion / max(avg_score, 0.01), 2) if avg_score > 0 else 1.0,
        }

    @classmethod
    def _compute_overall_risk(cls, signals: dict) -> float:
        """
        Compute overall systemic risk level from all signals.
        Uses weighted combination with non-linear risk escalation.
        """
        weights = {
            "correlation_convergence": 0.20,
            "vol_regime_break": 0.25,
            "yield_curve_inversion": 0.15,
            "credit_stress": 0.20,
            "contagion_score": 0.20,
        }

        weighted_sum = 0.0
        for signal, weight in weights.items():
            value = signals.get(signal, 0.0)
            if isinstance(value, bool):
                value = 0.8 if value else 0.1
            weighted_sum += value * weight

        # Non-linear escalation: if average is high, push higher
        if weighted_sum > 0.6:
            weighted_sum = weighted_sum ** 0.85  # Compress toward 1.0 for high values
        elif weighted_sum > 0.4:
            weighted_sum *= 1.05  # Slight amplification in moderate range

        return max(0.0, min(1.0, weighted_sum))

    @classmethod
    def _generate_assessment(cls, signals: dict, category: str, details: dict) -> str:
        """Generate human-readable risk assessment narrative."""
        parts = []

        if category == "critical":
            parts.append("CRITICAL: Multiple systemic risk signals firing simultaneously.")
        elif category == "high":
            parts.append("HIGH RISK: Significant systemic stress detected across markets.")
        elif category == "elevated":
            parts.append("ELEVATED: Active risk signals require attention.")
        elif category == "moderate":
            parts.append("MODERATE: Some stress indicators warrant monitoring.")
        else:
            parts.append("LOW: Markets showing normal risk characteristics.")

        # Specific signal callouts
        if signals.get("correlation_convergence", 0) > 0.5:
            parts.append("Cross-asset correlations converging — diversification benefits reduced.")

        if signals.get("vol_regime_break", 0) > 0.5:
            vix = details.get("vol_regime_break", {}).get("vix_level")
            vix_str = f" (VIX: {vix})" if vix else ""
            parts.append(f"Volatility regime break detected{vix_str} — potential market shock.")

        yc_inverted = signals.get("yield_curve_inversion", False)
        if yc_inverted:
            parts.append("Yield curve inverted — historic recession warning signal.")

        if signals.get("credit_stress", 0) > 0.5:
            parts.append("Credit spreads widening — financial stress in credit markets.")

        if signals.get("contagion_score", 0) > 0.6:
            parts.append("Contagion risk elevated — risk signals amplifying each other.")

        return " ".join(parts)

    @classmethod
    def _compute_confidence(cls, details: dict) -> float:
        """Compute confidence based on data availability and quality."""
        # Higher confidence when more signals have data
        has_data = 0
        total = 0

        for key, detail in details.items():
            total += 1
            if isinstance(detail, dict):
                score = detail.get("score", None)
                if score is not None:
                    has_data += 1

        if total == 0:
            return 0.3

        data_coverage = has_data / total
        return min(0.95, 0.5 + data_coverage * 0.4)
