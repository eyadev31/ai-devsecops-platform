"""
Hybrid Intelligence Portfolio System -- Adaptive Risk Classifier
===================================================================
ML engine that computes a continuous risk tolerance score (0.0-1.0)
and behavioral type classification from DAQ answer patterns.

Key features:
  1. Continuous risk scoring (not buckets -- a precise 0.0-1.0 value)
  2. Market-adjusted scoring (risk tolerance shifts with market stress)
  3. 8 behavioral type classifications
  4. Liquidity preference inference
  5. Time horizon detection
  6. Maximum acceptable drawdown estimation
  7. Confidence scoring based on answer quality
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# ================================================================
#  BEHAVIORAL TYPE DEFINITIONS
# ================================================================

BEHAVIORAL_TYPES = {
    "conservative_stable": {
        "risk_range": (0.0, 0.20),
        "stability": "stable",
        "description": "Risk-averse with consistent behavior across conditions",
        "typical_drawdown_tolerance": 0.05,
        "liquidity": "high",
        "time_horizon": "short",
    },
    "conservative_anxious": {
        "risk_range": (0.0, 0.25),
        "stability": "volatile",
        "description": "Risk-averse but emotionally unstable under stress",
        "typical_drawdown_tolerance": 0.03,
        "liquidity": "high",
        "time_horizon": "short",
    },
    "moderate_balanced": {
        "risk_range": (0.30, 0.55),
        "stability": "stable",
        "description": "Balanced risk-taker with consistent rational decisions",
        "typical_drawdown_tolerance": 0.12,
        "liquidity": "medium",
        "time_horizon": "medium",
    },
    "moderate_volatile": {
        "risk_range": (0.30, 0.55),
        "stability": "volatile",
        "description": "Moderate risk appetite but behavior shifts with market conditions",
        "typical_drawdown_tolerance": 0.10,
        "liquidity": "medium",
        "time_horizon": "medium",
    },
    "growth_seeker": {
        "risk_range": (0.55, 0.75),
        "stability": "stable",
        "description": "Growth-oriented investor with strong conviction",
        "typical_drawdown_tolerance": 0.20,
        "liquidity": "medium",
        "time_horizon": "long",
    },
    "growth_seeker_with_volatility_sensitivity": {
        "risk_range": (0.55, 0.75),
        "stability": "volatile",
        "description": "Wants growth but struggles psychologically with drawdowns",
        "typical_drawdown_tolerance": 0.15,
        "liquidity": "medium",
        "time_horizon": "medium",
    },
    "aggressive_speculator": {
        "risk_range": (0.75, 1.0),
        "stability": "any",
        "description": "High risk tolerance, often overconfident",
        "typical_drawdown_tolerance": 0.30,
        "liquidity": "low",
        "time_horizon": "long",
    },
    "aggressive_contrarian": {
        "risk_range": (0.75, 1.0),
        "stability": "stable",
        "description": "Buys fear, sells greed -- contrarian with discipline",
        "typical_drawdown_tolerance": 0.35,
        "liquidity": "low",
        "time_horizon": "long",
    },
}


class AdaptiveRiskClassifier:
    """
    Computes continuous risk score and behavioral type from DAQ answers.

    Pipeline:
      1. Extract feature vector from answers
      2. Compute raw risk score
      3. Apply market adjustment
      4. Classify behavioral type
      5. Infer liquidity preference & time horizon
      6. Estimate max acceptable drawdown
      7. Score confidence in classification
    """

    @classmethod
    def classify(
        cls,
        questions: list[dict],
        answers: list[dict],
        behavioral_profile: dict,
        market_stress: float = 0.5,
        agent1_output: Optional[dict] = None,
    ) -> dict:
        """
        Run full risk classification pipeline.

        Args:
            questions: Generated questions
            answers: User answers
            behavioral_profile: From BehavioralConsistencyAnalyzer
            market_stress: Stress multiplier (0-1)
            agent1_output: Agent 1 context for market adjustment

        Returns:
            RiskClassification dict ready for schema validation
        """
        logger.info("Running adaptive risk classification...")

        # Match answers to questions
        answer_map = cls._match_answers(questions, answers)

        # ── Step 1: Extract features ──────────────────────
        features = cls._extract_features(answer_map, behavioral_profile)

        # ── Step 2: Compute raw risk score ────────────────
        raw_score = cls._compute_raw_risk_score(features)

        # ── Step 3: Market adjustment ─────────────────────
        adjusted_score = cls._apply_market_adjustment(
            raw_score, market_stress, behavioral_profile, agent1_output
        )

        # ── Step 4: Behavioral type classification ────────
        behavioral_type = cls._classify_type(
            adjusted_score, behavioral_profile, features
        )

        # ── Step 5: Infer preferences ─────────────────────
        liquidity = cls._infer_liquidity(features, behavioral_type)
        time_horizon = cls._infer_time_horizon(features, behavioral_type)
        max_drawdown = cls._estimate_max_drawdown(
            adjusted_score, behavioral_profile, features
        )

        # ── Step 6: Confidence scoring ────────────────────
        confidence = cls._compute_confidence(
            features, behavioral_profile, answer_map
        )

        result = {
            "risk_score": adjusted_score,
            "risk_score_raw": raw_score,
            "market_adjusted": True,
            "behavioral_type": behavioral_type,
            "confidence": confidence,
            "liquidity_preference": liquidity,
            "time_horizon": time_horizon,
            "max_acceptable_drawdown": max_drawdown,
        }

        logger.info(
            f"Risk classification: score={adjusted_score:.2f} (raw={raw_score:.2f}), "
            f"type={behavioral_type}, confidence={confidence:.2f}, "
            f"max_dd={max_drawdown:.0%}"
        )

        return result

    # ────────────────────────────────────────────────────
    #  FEATURE EXTRACTION
    # ────────────────────────────────────────────────────

    @classmethod
    def _match_answers(cls, questions: list[dict], answers: list[dict]) -> list[dict]:
        """Match answers to questions with full choice data."""
        q_map = {q["question_id"]: q for q in questions}
        matched = []

        for ans in answers:
            question = q_map.get(ans.get("question_id", ""))
            if not question:
                continue

            selected = None
            for choice in question.get("choices", []):
                if choice.get("id") == ans.get("selected_choice_id"):
                    selected = choice
                    break

            matched.append({
                "question": question,
                "answer": ans,
                "selected_choice": selected,
                "risk_signal": selected.get("risk_signal", 0.5) if selected else 0.5,
                "category": question.get("category", "unknown"),
                "difficulty": question.get("difficulty", 0.5),
            })

        return matched

    @classmethod
    def _extract_features(
        cls, answer_map: list[dict], behavioral_profile: dict
    ) -> dict:
        """
        Extract a rich feature vector from answer patterns.

        Features capture:
          - Core risk tolerance signals
          - Behavioral dimensions
          - Interaction with difficulty levels
          - Consistency metrics
        """
        if not answer_map:
            return cls._empty_features()

        risk_signals = [a["risk_signal"] for a in answer_map]
        difficulties = [a["difficulty"] for a in answer_map]

        # Core statistics
        mean_risk = sum(risk_signals) / len(risk_signals)
        std_risk = math.sqrt(
            sum((s - mean_risk) ** 2 for s in risk_signals) / len(risk_signals)
        ) if len(risk_signals) > 1 else 0.0

        # Weighted mean (harder questions weighted more -- they reveal true risk tolerance)
        weighted_sum = sum(r * (0.5 + d) for r, d in zip(risk_signals, difficulties))
        weight_total = sum(0.5 + d for d in difficulties)
        weighted_mean = weighted_sum / weight_total if weight_total > 0 else mean_risk

        # Category-specific signals
        category_signals = {}
        for a in answer_map:
            cat = a["category"]
            if cat not in category_signals:
                category_signals[cat] = []
            category_signals[cat].append(a["risk_signal"])

        category_means = {
            cat: sum(sigs) / len(sigs)
            for cat, sigs in category_signals.items()
        }

        # Loss aversion signal (key behavioral indicator)
        loss_signal = category_means.get("loss_aversion", mean_risk)

        # High-difficulty performance
        high_diff = [a["risk_signal"] for a in answer_map if a["difficulty"] >= 0.6]
        high_diff_mean = sum(high_diff) / len(high_diff) if high_diff else mean_risk

        features = {
            "mean_risk_signal": round(mean_risk, 4),
            "std_risk_signal": round(std_risk, 4),
            "weighted_mean_risk": round(weighted_mean, 4),
            "max_risk_signal": max(risk_signals),
            "min_risk_signal": min(risk_signals),
            "risk_range": max(risk_signals) - min(risk_signals),
            "loss_aversion_signal": round(loss_signal, 4),
            "high_difficulty_signal": round(high_diff_mean, 4),
            "category_signals": category_means,
            "consistency_score": behavioral_profile.get("consistency_score", 0.5),
            "emotional_stability_score": behavioral_profile.get("emotional_stability_score", 0.5),
            "num_contradictions": len(behavioral_profile.get("contradiction_flags", [])),
            "num_biases": len(behavioral_profile.get("detected_biases", [])),
            "stress_response": behavioral_profile.get("stress_response_pattern", "unknown"),
            "n_answers": len(answer_map),
        }

        return features

    @staticmethod
    def _empty_features() -> dict:
        """Return empty feature dict when no answers available."""
        return {
            "mean_risk_signal": 0.5,
            "std_risk_signal": 0.0,
            "weighted_mean_risk": 0.5,
            "max_risk_signal": 0.5,
            "min_risk_signal": 0.5,
            "risk_range": 0.0,
            "loss_aversion_signal": 0.5,
            "high_difficulty_signal": 0.5,
            "category_signals": {},
            "consistency_score": 0.5,
            "emotional_stability_score": 0.5,
            "num_contradictions": 0,
            "num_biases": 0,
            "stress_response": "unknown",
            "n_answers": 0,
        }

    # ────────────────────────────────────────────────────
    #  RAW RISK SCORING
    # ────────────────────────────────────────────────────

    @classmethod
    def _compute_raw_risk_score(cls, features: dict) -> float:
        """
        Compute raw risk tolerance score (0.0-1.0).

        Uses a weighted combination of:
          - Weighted mean risk signal (primary: 45%)
          - High-difficulty response (stress-tested: 25%)
          - Loss aversion signal (behavioral floor: 15%)
          - Min risk signal (worst case tolerance: 15%)
        """
        weighted_mean = features["weighted_mean_risk"]
        high_diff = features["high_difficulty_signal"]
        loss_signal = features["loss_aversion_signal"]
        min_signal = features["min_risk_signal"]

        raw = (
            0.45 * weighted_mean +
            0.25 * high_diff +
            0.15 * loss_signal +
            0.15 * min_signal
        )

        return round(max(0.0, min(1.0, raw)), 4)

    # ────────────────────────────────────────────────────
    #  MARKET ADJUSTMENT
    # ────────────────────────────────────────────────────

    @classmethod
    def _apply_market_adjustment(
        cls,
        raw_score: float,
        market_stress: float,
        behavioral_profile: dict,
        agent1_output: Optional[dict] = None,
    ) -> float:
        """
        Adjust risk score based on market conditions.

        Rationale: A user who says they're aggressive in calm markets
        may not actually be aggressive. We slightly lower their score
        during low-stress questioning, and slightly raise it during
        high-stress (because they're being tested harder).

        Also accounts for behavioral stability: unstable users get
        a downward adjustment (they may panic when markets actually move).
        """
        adjusted = raw_score

        # ── Market stress adjustment ──────────────────────
        # During calm markets, expressed risk tolerance may be inflated
        if market_stress < 0.3:
            adjusted -= 0.05  # Deflate calm-market bravado
        elif market_stress > 0.7:
            adjusted += 0.03  # Credit for maintaining risk appetite under stress

        # ── Stability adjustment ──────────────────────────
        stability = behavioral_profile.get("emotional_stability", "moderate")
        stability_adjustments = {
            "stable": 0.0,
            "moderate": -0.02,
            "volatile": -0.05,
            "highly_volatile": -0.10,
        }
        adjusted += stability_adjustments.get(stability, -0.02)

        # ── Stress response adjustment ────────────────────
        stress_response = behavioral_profile.get("stress_response_pattern", "unknown")
        if stress_response == "flight":
            adjusted -= 0.05  # Likely to panic sell
        elif stress_response == "fight":
            adjusted += 0.03  # Contrarian, may handle drawdowns
        elif stress_response == "freeze":
            adjusted -= 0.03  # May fail to rebalance

        # ── Contradiction adjustment ──────────────────────
        contradictions = behavioral_profile.get("contradiction_flags", [])
        if len(contradictions) >= 2:
            adjusted -= 0.05  # Uncertain about true risk tolerance

        # ── Agent 1 Confidence Adjustment (MONSTER RULE) ──
        if agent1_output:
            a1_conf = agent1_output.get("market_regime", {}).get("confidence", 1.0)
            if a1_conf < 0.60:
                logger.warning(f"MONSTER RULE: Agent 1 Confidence ({a1_conf}) < 0.60. Reducing risk score.")
                adjusted -= 0.08  # Force defensive behavioral profiling

        return round(max(0.0, min(1.0, adjusted)), 4)

    # ────────────────────────────────────────────────────
    #  BEHAVIORAL TYPE CLASSIFICATION
    # ────────────────────────────────────────────────────

    @classmethod
    def _classify_type(
        cls, risk_score: float, behavioral_profile: dict, features: dict
    ) -> str:
        """
        Classify into one of 8 behavioral types.

        Uses risk score + stability to determine the precise type.
        """
        stability = behavioral_profile.get("emotional_stability", "moderate")
        stress_response = behavioral_profile.get("stress_response_pattern", "unknown")
        is_stable = stability in ("stable",)
        is_volatile = stability in ("volatile", "highly_volatile")

        # ── Conservative zone (0.0 - 0.30) ────────────────
        if risk_score <= 0.30:
            if is_stable:
                return "conservative_stable"
            else:
                return "conservative_anxious"

        # ── Moderate zone (0.30 - 0.55) ───────────────────
        elif risk_score <= 0.55:
            if is_stable:
                return "moderate_balanced"
            else:
                return "moderate_volatile"

        # ── Growth zone (0.55 - 0.75) ─────────────────────
        elif risk_score <= 0.75:
            # Check for volatility sensitivity specifically
            loss_signal = features.get("loss_aversion_signal", 0.5)
            if is_volatile or (loss_signal < risk_score - 0.15):
                return "growth_seeker_with_volatility_sensitivity"
            else:
                return "growth_seeker"

        # ── Aggressive zone (0.75 - 1.0) ──────────────────
        else:
            # Contrarian = aggressive + stable + fights under stress
            if is_stable and stress_response == "fight":
                return "aggressive_contrarian"
            else:
                return "aggressive_speculator"

    # ────────────────────────────────────────────────────
    #  PREFERENCE INFERENCE
    # ────────────────────────────────────────────────────

    @classmethod
    def _infer_liquidity(cls, features: dict, behavioral_type: str) -> str:
        """Infer liquidity preference from behavioral type and risk features."""
        type_info = BEHAVIORAL_TYPES.get(behavioral_type, {})
        base_liquidity = type_info.get("liquidity", "medium")

        # Override if stress response suggests liquidity need
        stress = features.get("stress_response", "unknown")
        if stress == "flight" and base_liquidity != "high":
            return "high"  # Flight responders need liquid assets

        return base_liquidity

    @classmethod
    def _infer_time_horizon(cls, features: dict, behavioral_type: str) -> str:
        """Infer investment time horizon."""
        type_info = BEHAVIORAL_TYPES.get(behavioral_type, {})
        return type_info.get("time_horizon", "medium")

    @classmethod
    def _estimate_max_drawdown(
        cls,
        risk_score: float,
        behavioral_profile: dict,
        features: dict,
    ) -> float:
        """
        Estimate the maximum drawdown the user can psychologically handle.

        Uses a calibrated mapping from risk score, tempered by
        behavioral stability and loss aversion signals.
        """
        # Base drawdown from risk score (linear mapping)
        # 0.0 risk -> 3% DD, 1.0 risk -> 40% DD
        base_dd = 0.03 + risk_score * 0.37

        # Loss aversion adjustment
        loss_signal = features.get("loss_aversion_signal", 0.5)
        if loss_signal < 0.3:
            base_dd *= 0.7  # Very loss-averse: lower DD tolerance
        elif loss_signal > 0.7:
            base_dd *= 1.1  # Loss-tolerant: slightly higher DD

        # Stability adjustment
        stability = behavioral_profile.get("emotional_stability_score", 0.5)
        if stability < 0.4:
            base_dd *= 0.8  # Emotionally unstable: reduce DD
        elif stability > 0.75:
            base_dd *= 1.05  # Very stable: slight increase

        return round(max(0.02, min(0.50, base_dd)), 2)

    # ────────────────────────────────────────────────────
    #  CONFIDENCE SCORING
    # ────────────────────────────────────────────────────

    @classmethod
    def _compute_confidence(
        cls,
        features: dict,
        behavioral_profile: dict,
        answer_map: list[dict],
    ) -> float:
        """
        Compute confidence in the risk classification.

        Factors:
          - Number of answers (more = higher confidence)
          - Consistency score (high = confident)
          - Low contradictions (fewer = confident)
          - Answer diversity (not all same choice = confident)
        """
        confidence = 0.5

        # More answers = more data = higher confidence
        n = features.get("n_answers", 0)
        if n >= 4:
            confidence += 0.15
        elif n >= 3:
            confidence += 0.10
        elif n >= 2:
            confidence += 0.05

        # Consistency bonus
        consistency = features.get("consistency_score", 0.5)
        confidence += (consistency - 0.5) * 0.3  # -0.15 to +0.15

        # Contradiction penalty
        n_contradictions = features.get("num_contradictions", 0)
        confidence -= min(0.20, n_contradictions * 0.08)

        # Risk range: extremely narrow = possibly disengaged, very wide = inconsistent
        risk_range = features.get("risk_range", 0.0)
        if risk_range < 0.05 and n >= 3:
            confidence -= 0.10  # Possibly not paying attention
        elif risk_range > 0.6:
            confidence -= 0.10  # Very inconsistent

        return round(max(0.2, min(0.98, confidence)), 2)
