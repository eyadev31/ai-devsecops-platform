"""
Hybrid Intelligence Portfolio System — Volatility State Classifier
====================================================================
Multi-dimensional volatility regime classification using statistical methods.
Analyzes realized vol, implied vol (VIX), vol-of-vol, and term structure.

Classifications: extremely_low | low | normal | elevated | extreme
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import MLConfig

logger = logging.getLogger(__name__)


class VolatilityClassifier:
    """
    Classifies the current volatility state along multiple dimensions:
      1. Realized volatility level (z-score based)
      2. Implied volatility (VIX) level
      3. Volatility-of-volatility (regime instability)
      4. Volatility term structure (contango vs backwardation)
      5. Volatility trend direction
    """

    VOL_STATES = ["extremely_low", "low", "normal", "elevated", "extreme"]

    @classmethod
    def classify(
        cls,
        features: dict,
        vix_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Produce comprehensive volatility state classification.

        Args:
            features: Output from FeatureEngine.build_features()
            vix_data: VIX OHLCV DataFrame (from TwelveData)

        Returns:
            {
                "current_state": str,
                "vix_level": float,
                "realized_vol_percentile": float,
                "vol_trend": str,
                "vol_of_vol": str,
                "term_structure": str,
                "components": dict,
                "confidence": float,
            }
        """
        logger.info("Classifying volatility state...")

        result = {
            "current_state": "normal",
            "vix_level": None,
            "realized_vol_percentile": 50.0,
            "vol_trend": "stable",
            "vol_of_vol": "normal",
            "term_structure": "unknown",
            "components": {},
            "confidence": 0.0,
        }

        component_scores = []

        # ── 1. Realized Volatility Classification ────────
        vol_data = features.get("volatility", {})
        realized_vol = vol_data.get("realized")
        if realized_vol is not None and isinstance(realized_vol, pd.DataFrame) and not realized_vol.empty:
            rv_result = cls._classify_realized_vol(realized_vol)
            result["components"]["realized_vol"] = rv_result
            component_scores.append(rv_result)

        # ── 2. Volatility Percentile ─────────────────────
        vol_percentile = vol_data.get("vol_percentile")
        if vol_percentile is not None and isinstance(vol_percentile, pd.Series) and not vol_percentile.empty:
            pct = float(vol_percentile.iloc[-1])
            result["realized_vol_percentile"] = round(pct, 1)

        # ── 3. VIX Level Classification ──────────────────
        if vix_data is not None and not vix_data.empty and "close" in vix_data.columns:
            vix_result = cls._classify_vix(vix_data["close"])
            result["components"]["vix"] = vix_result
            result["vix_level"] = vix_result.get("current_value")
            component_scores.append(vix_result)

        # ── 4. Vol-of-Vol Classification ─────────────────
        vov = vol_data.get("vol_of_vol")
        if vov is not None and isinstance(vov, pd.Series) and not vov.empty:
            vov_result = cls._classify_vol_of_vol(vov)
            result["components"]["vol_of_vol"] = vov_result
            result["vol_of_vol"] = vov_result.get("state", "normal")
            component_scores.append(vov_result)

        # ── 5. Volatility Trend ──────────────────────────
        vol_zscore = vol_data.get("vol_zscore")
        if vol_zscore is not None and isinstance(vol_zscore, pd.Series) and not vol_zscore.empty:
            trend_result = cls._classify_vol_trend(vol_zscore)
            result["vol_trend"] = trend_result.get("trend", "stable")
            result["components"]["vol_trend"] = trend_result

        # ── Aggregate State ──────────────────────────────
        if component_scores:
            aggregate = cls._aggregate_state(component_scores, result)
            result["current_state"] = aggregate["state"]
            result["confidence"] = aggregate["confidence"]
        else:
            result["confidence"] = 0.3

        logger.info(
            f"Volatility state: {result['current_state']} "
            f"(VIX: {result['vix_level']}, "
            f"percentile: {result['realized_vol_percentile']}, "
            f"trend: {result['vol_trend']})"
        )

        return result

    @classmethod
    def _classify_realized_vol(cls, realized_vol: pd.DataFrame) -> dict:
        """Classify realized volatility using z-score."""
        # Use 21-day vol as primary
        vol_col = "vol_21d" if "vol_21d" in realized_vol.columns else realized_vol.columns[0]
        vol_series = realized_vol[vol_col].dropna()

        if len(vol_series) < 30:
            return {"state": "normal", "zscore": 0.0, "current_value": 0.0, "confidence": 0.3}

        current_vol = float(vol_series.iloc[-1])
        mean_vol = float(vol_series.mean())
        std_vol = float(vol_series.std())

        if std_vol == 0:
            zscore = 0.0
        else:
            zscore = (current_vol - mean_vol) / std_vol

        state = cls._zscore_to_state(zscore)
        confidence = min(0.95, 0.5 + abs(zscore) * 0.15)

        return {
            "state": state,
            "zscore": round(zscore, 3),
            "current_value": round(current_vol * 100, 2),  # as percentage
            "mean": round(mean_vol * 100, 2),
            "confidence": round(confidence, 3),
        }

    @classmethod
    def _classify_vix(cls, vix_close: pd.Series) -> dict:
        """Classify VIX level using absolute thresholds and history."""
        if vix_close.empty:
            return {"state": "normal", "current_value": None, "confidence": 0.3}

        current_vix = float(vix_close.iloc[-1])

        # Historical context
        if len(vix_close) >= 252:
            percentile = float(stats.percentileofscore(vix_close.iloc[-252:], current_vix))
        else:
            percentile = 50.0

        # Absolute VIX thresholds (well-established in finance)
        if current_vix < 12:
            state = "extremely_low"
        elif current_vix < 16:
            state = "low"
        elif current_vix < 22:
            state = "normal"
        elif current_vix < 30:
            state = "elevated"
        else:
            state = "extreme"

        confidence = 0.85  # VIX is a direct measure

        return {
            "state": state,
            "current_value": round(current_vix, 2),
            "percentile": round(percentile, 1),
            "confidence": confidence,
        }

    @classmethod
    def _classify_vol_of_vol(cls, vov_series: pd.Series) -> dict:
        """Classify vol-of-vol (regime stability indicator)."""
        if vov_series.empty or len(vov_series) < 10:
            return {"state": "normal", "confidence": 0.3}

        current_vov = float(vov_series.iloc[-1])
        mean_vov = float(vov_series.mean())
        std_vov = float(vov_series.std())

        if std_vov == 0:
            zscore = 0.0
        else:
            zscore = (current_vov - mean_vov) / std_vov

        if zscore < -1:
            state = "stable"
        elif zscore < 0.5:
            state = "normal"
        elif zscore < 1.5:
            state = "elevated"
        else:
            state = "unstable"

        return {
            "state": state,
            "zscore": round(zscore, 3),
            "current": round(current_vov, 6),
            "confidence": min(0.85, 0.5 + abs(zscore) * 0.12),
        }

    @classmethod
    def _classify_vol_trend(cls, vol_zscore: pd.Series) -> dict:
        """Classify the direction of volatility movement."""
        if len(vol_zscore) < 10:
            return {"trend": "stable", "confidence": 0.3}

        current = float(vol_zscore.iloc[-1])
        prev_5d = float(vol_zscore.iloc[-5]) if len(vol_zscore) >= 5 else current
        prev_10d = float(vol_zscore.iloc[-10]) if len(vol_zscore) >= 10 else current

        change_5d = current - prev_5d
        change_10d = current - prev_10d

        if change_5d > 0.5 and change_10d > 0.3:
            trend = "sharply_increasing"
        elif change_5d > 0.2:
            trend = "increasing"
        elif change_5d < -0.5 and change_10d < -0.3:
            trend = "sharply_decreasing"
        elif change_5d < -0.2:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "current_zscore": round(current, 3),
            "5d_change": round(change_5d, 3),
            "10d_change": round(change_10d, 3),
            "confidence": min(0.85, 0.5 + abs(change_5d) * 0.15),
        }

    @classmethod
    def _zscore_to_state(cls, zscore: float) -> str:
        """Convert z-score to volatility state label."""
        thresholds = MLConfig.VOL_ZSCORE_THRESHOLDS
        if zscore <= thresholds["extremely_low"]:
            return "extremely_low"
        elif zscore <= thresholds["low"]:
            return "low"
        elif zscore <= thresholds["normal_high"]:
            return "normal"
        elif zscore <= thresholds["elevated"]:
            return "elevated"
        else:
            return "extreme"

    @classmethod
    def _aggregate_state(cls, components: list[dict], result: dict) -> dict:
        """
        Aggregate component classifications into overall state.
        Weighted by confidence of each component.
        """
        state_scores = {s: 0.0 for s in cls.VOL_STATES}
        total_weight = 0.0

        for comp in components:
            state = comp.get("state", "normal")
            conf = comp.get("confidence", 0.5)

            if state in state_scores:
                state_scores[state] += conf
                total_weight += conf

        if total_weight == 0:
            return {"state": "normal", "confidence": 0.3}

        # Normalize
        for s in state_scores:
            state_scores[s] /= total_weight

        best_state = max(state_scores, key=state_scores.get)
        best_score = state_scores[best_state]

        # Confidence = support for best state
        confidence = min(0.95, best_score + 0.1)

        return {"state": best_state, "confidence": round(confidence, 3)}
