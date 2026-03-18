"""
Hybrid Intelligence Portfolio System -- Question Calibration Engine
=====================================================================
Calibrates question difficulty and category selection based on
Agent 1's market context. This is the ML backbone that drives
the question generation pipeline.

The engine does NOT generate questions (that's the LLM's job).
Instead, it determines:
  1. How stressful the questions should be (stress_multiplier)
  2. Which behavioral biases to probe (category selection)
  3. What numeric parameters to inject (drawdown %, time pressure)
  4. What market data points to reference in scenarios
"""

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# ================================================================
#  BEHAVIORAL CATEGORIES
# ================================================================

BEHAVIORAL_CATEGORIES = {
    "loss_aversion": {
        "description": "Fear of losses outweighing equivalent gains",
        "stress_weight": 0.9,   # very relevant during market stress
        "question_triggers": ["drawdown", "portfolio_decline", "unrealized_loss"],
    },
    "herd_behavior": {
        "description": "Following the crowd regardless of fundamentals",
        "stress_weight": 0.8,
        "question_triggers": ["market_panic", "trending_asset", "media_sentiment"],
    },
    "anchoring": {
        "description": "Fixating on a reference point (purchase price, past high)",
        "stress_weight": 0.5,
        "question_triggers": ["purchase_price", "all_time_high", "round_number"],
    },
    "time_pressure": {
        "description": "Decision quality under urgency",
        "stress_weight": 1.0,   # peak relevance in crisis
        "question_triggers": ["flash_crash", "deadline", "opportunity_window"],
    },
    "regret_aversion": {
        "description": "Inaction from fear of making the wrong choice",
        "stress_weight": 0.7,
        "question_triggers": ["missed_rally", "wrong_timing", "opportunity_cost"],
    },
    "overconfidence": {
        "description": "Overestimating one's ability to predict markets",
        "stress_weight": 0.3,   # more relevant in bull markets
        "question_triggers": ["prediction_accuracy", "beating_market", "leverage"],
    },
    "recency_bias": {
        "description": "Overweighting recent events in projections",
        "stress_weight": 0.6,
        "question_triggers": ["recent_crash", "recent_rally", "news_cycle"],
    },
    "disposition_effect": {
        "description": "Selling winners too early, holding losers too long",
        "stress_weight": 0.7,
        "question_triggers": ["profit_taking", "averaging_down", "stop_loss"],
    },
    "mental_accounting": {
        "description": "Treating money differently based on source or label",
        "stress_weight": 0.3,
        "question_triggers": ["bonus_money", "retirement_fund", "play_money"],
    },
    "sunk_cost": {
        "description": "Continuing investment because of sunk costs",
        "stress_weight": 0.5,
        "question_triggers": ["losing_position", "additional_investment", "cut_losses"],
    },
}


class QuestionCalibrator:
    """
    Calibrates question generation parameters based on market context.

    This is a quantitative engine, not an LLM. It uses Agent 1's output
    to compute:
      - stress_multiplier: 0.0 (calm markets) to 1.0 (crisis)
      - category_weights: probability distribution over bias categories
      - scenario_params: numeric parameters for question scenarios
    """

    @classmethod
    def calibrate(
        cls,
        agent1_output: dict,
        user_history: Optional[list[dict]] = None,
        num_questions: int = 4,
    ) -> dict:
        """
        Compute calibration parameters from Agent 1 context.

        Args:
            agent1_output: Complete Agent 1 JSON output
            user_history: Previous session answers (for category dedup)
            num_questions: Number of questions to generate

        Returns:
            Calibration dictionary with stress level, categories, and scenario params
        """
        # Extract market context
        regime = agent1_output.get("market_regime", {})
        vol_state = agent1_output.get("volatility_state", {})
        risk = agent1_output.get("systemic_risk", {})
        macro = agent1_output.get("macro_environment", {})

        # ── Step 1: Compute stress multiplier ──────────────
        stress = cls._compute_stress_multiplier(regime, vol_state, risk)

        # ── Step 2: Select categories ──────────────────────
        categories = cls._select_categories(
            stress, num_questions, user_history
        )

        # ── Step 3: Compute scenario parameters ────────────
        scenario_params = cls._build_scenario_params(
            regime, vol_state, risk, macro, stress
        )

        # ── Step 4: Build calibration notes ────────────────
        calibration_notes = cls._build_calibration_notes(
            regime, vol_state, risk, stress
        )

        calibration = {
            "stress_multiplier": stress,
            "categories": categories,
            "difficulty_range": cls._difficulty_range(stress),
            "scenario_params": scenario_params,
            "regime_used": regime.get("primary_regime", "unknown"),
            "volatility_state": vol_state.get("current_state", "unknown"),
            "risk_level": risk.get("risk_category", "unknown"),
            "calibration_notes": calibration_notes,
        }

        logger.info(
            f"Calibrated: stress={stress:.2f}, "
            f"categories={[c['name'] for c in categories]}, "
            f"regime={regime.get('primary_regime', 'N/A')}"
        )

        return calibration

    # ────────────────────────────────────────────────────
    #  STRESS MULTIPLIER COMPUTATION
    # ────────────────────────────────────────────────────

    @classmethod
    def _compute_stress_multiplier(
        cls, regime: dict, vol_state: dict, risk: dict
    ) -> float:
        """
        Compute market stress multiplier (0.0-1.0).
        Higher stress = harder, more emotionally charged questions.

        Factors:
          - Regime type (bear > bull)
          - Volatility state (extreme > normal)
          - Systemic risk level
          - Vol trend (increasing adds stress)
          - Regime confidence (low confidence = uncertainty = stress)
        """
        stress = 0.0

        # Regime contribution (0.0 - 0.35)
        regime_name = regime.get("primary_regime", "").lower()
        regime_scores = {
            "bull_low_vol": 0.05,
            "bull_high_vol": 0.20,
            "bear_low_vol": 0.20,
            "bear_high_vol": 0.35,
        }
        # Match partial names for flexibility
        for key, score in regime_scores.items():
            if key in regime_name:
                stress += score
                break
        else:
            stress += 0.15  # unknown regime = moderate stress

        # Volatility contribution (0.0 - 0.25)
        vol_scores = {
            "extremely_low": 0.0,
            "low": 0.05,
            "normal": 0.10,
            "elevated": 0.20,
            "extreme": 0.25,
        }
        stress += vol_scores.get(
            vol_state.get("current_state", "normal"), 0.10
        )

        # Vol trend bonus
        vol_trend = vol_state.get("vol_trend", "stable")
        if vol_trend in ("sharply_increasing", "increasing"):
            stress += 0.05

        # Systemic risk contribution (0.0 - 0.25)
        risk_level = risk.get("overall_risk_level", 0.0)
        stress += min(0.25, risk_level * 0.5)

        # Regime uncertainty bonus (0.0 - 0.10)
        confidence = regime.get("confidence", 0.5)
        if confidence < 0.6:
            stress += 0.10
        elif confidence < 0.75:
            stress += 0.05

        # Models disagreeing = uncertainty
        if not regime.get("models_agree", True):
            stress += 0.05

        return round(min(1.0, max(0.0, stress)), 3)

    # ────────────────────────────────────────────────────
    #  CATEGORY SELECTION
    # ────────────────────────────────────────────────────

    @classmethod
    def _select_categories(
        cls,
        stress: float,
        num_questions: int,
        user_history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Select behavioral categories to probe, weighted by stress level.

        High stress -> loss_aversion, time_pressure, herd_behavior
        Low stress  -> overconfidence, mental_accounting, anchoring
        """
        # Compute weights for each category
        weighted = {}
        for name, info in BEHAVIORAL_CATEGORIES.items():
            # Base weight from stress relevance
            weight = info["stress_weight"] * stress + (1 - info["stress_weight"]) * (1 - stress)
            # Normalize to 0-1 range
            weight = max(0.1, weight)
            weighted[name] = weight

        # Reduce weight for recently used categories
        if user_history:
            recent_categories = set()
            for session in user_history[-3:]:  # last 3 sessions
                for q in session.get("questions", []):
                    recent_categories.add(q.get("category", ""))
            for cat in recent_categories:
                if cat in weighted:
                    weighted[cat] *= 0.3  # heavily penalize repeats

        # Weighted random selection without replacement
        all_cats = list(weighted.keys())
        all_weights = [weighted[c] for c in all_cats]

        selected = []
        remaining_cats = list(zip(all_cats, all_weights))

        for _ in range(min(num_questions, len(remaining_cats))):
            cats, weights = zip(*remaining_cats)
            total = sum(weights)
            probs = [w / total for w in weights]

            chosen_idx = random.choices(range(len(cats)), weights=probs, k=1)[0]
            chosen_cat = cats[chosen_idx]

            # Assign difficulty within the stress-calibrated range
            diff_low, diff_high = cls._difficulty_range(stress)
            difficulty = round(random.uniform(diff_low, diff_high), 2)

            selected.append({
                "name": chosen_cat,
                "difficulty": difficulty,
                "description": BEHAVIORAL_CATEGORIES[chosen_cat]["description"],
                "triggers": BEHAVIORAL_CATEGORIES[chosen_cat]["question_triggers"],
            })

            remaining_cats = [
                (c, w) for c, w in remaining_cats if c != chosen_cat
            ]
            if not remaining_cats:
                break

        return selected

    @staticmethod
    def _difficulty_range(stress: float) -> tuple[float, float]:
        """Map stress level to difficulty range."""
        if stress >= 0.7:
            return (0.6, 1.0)   # Hard to extreme
        elif stress >= 0.4:
            return (0.35, 0.75)  # Moderate to hard
        else:
            return (0.15, 0.50)  # Easy to moderate

    # ────────────────────────────────────────────────────
    #  SCENARIO PARAMETERS
    # ────────────────────────────────────────────────────

    @classmethod
    def _build_scenario_params(
        cls, regime: dict, vol: dict, risk: dict, macro: dict, stress: float
    ) -> dict:
        """
        Build numeric parameters for question scenarios.
        These get injected into the LLM prompt so questions reference
        realistic, current market numbers.
        """
        vix = vol.get("vix_level")
        risk_level = risk.get("overall_risk_level", 0.0)

        # Drawdown ranges calibrated to current vol
        if stress >= 0.7:
            drawdown_range = (-35, -15)
            recovery_months = (6, 24)
        elif stress >= 0.4:
            drawdown_range = (-20, -8)
            recovery_months = (3, 12)
        else:
            drawdown_range = (-12, -3)
            recovery_months = (1, 6)

        # Key macro indicators for scenarios
        indicators = macro.get("key_indicators", {})

        return {
            "drawdown_range_pct": drawdown_range,
            "recovery_timeframe_months": recovery_months,
            "current_vix": vix,
            "current_regime": regime.get("primary_regime", "unknown"),
            "regime_description": regime.get("description", ""),
            "risk_appetite_context": "low" if risk_level > 0.5 else "moderate" if risk_level > 0.25 else "high",
            "fed_rate": indicators.get("fed_funds_rate"),
            "unemployment": indicators.get("unemployment"),
            "yield_curve_inverted": macro.get("yield_curve", {}).get("inverted", False),
            "models_agree": regime.get("models_agree", True),
            "transition_probability": regime.get("transition_probability", 0.0),
        }

    @classmethod
    def _build_calibration_notes(
        cls, regime: dict, vol: dict, risk: dict, stress: float
    ) -> str:
        """Build human-readable calibration explanation."""
        parts = []

        regime_name = regime.get("primary_regime", "unknown")
        parts.append(f"Market regime: {regime_name}")

        if stress >= 0.7:
            parts.append("HIGH STRESS: Questions calibrated for crisis-level psychological testing")
        elif stress >= 0.4:
            parts.append("MODERATE STRESS: Questions probe intermediate risk scenarios")
        else:
            parts.append("LOW STRESS: Questions focus on preference discovery and subtle biases")

        vol_state = vol.get("current_state", "unknown")
        if vol_state in ("elevated", "extreme"):
            parts.append(f"Elevated volatility ({vol_state}) -- incorporating drawdown scenarios")

        risk_cat = risk.get("risk_category", "unknown")
        if risk_cat in ("elevated", "high", "critical"):
            parts.append(f"Systemic risk is {risk_cat} -- stress-testing panic responses")

        if not regime.get("models_agree", True):
            parts.append("ML models disagree on regime -- probing for ambiguity tolerance")

        return "; ".join(parts)
