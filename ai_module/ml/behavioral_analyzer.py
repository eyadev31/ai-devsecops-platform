"""
Hybrid Intelligence Portfolio System -- Behavioral Consistency Analyzer
=========================================================================
ML engine that analyzes user answer patterns to detect:
  1. Internal contradictions (saying one thing, doing another)
  2. Emotional stability across market conditions
  3. Cognitive biases revealed by answer patterns
  4. Stress response patterns (fight/flight/freeze/adapt)
  5. Decision speed profiling

This module works WITHOUT an LLM. It uses purely quantitative
signal processing on structured answer data.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class BehavioralConsistencyAnalyzer:
    """
    Analyzes behavioral consistency from DAQ answer patterns.

    This is a stateless analyzer -- it takes in a complete set of
    questions + answers and produces a behavioral profile.
    """

    # ── Risk signal thresholds for contradiction detection ──
    CONTRADICTION_THRESHOLD = 0.35  # min difference to flag as contradiction
    STRONG_CONTRADICTION = 0.50     # severe contradiction
    STABILITY_WINDOW = 3            # min sessions for stability analysis

    @classmethod
    def analyze(
        cls,
        questions: list[dict],
        answers: list[dict],
        market_stress: float = 0.5,
        historical_sessions: Optional[list[dict]] = None,
    ) -> dict:
        """
        Run full behavioral consistency analysis.

        Args:
            questions: List of GeneratedQuestion dicts
            answers: List of UserAnswer dicts
            market_stress: Current stress_multiplier (0-1)
            historical_sessions: Previous DAQ sessions for temporal analysis

        Returns:
            BehavioralProfile dict ready for Pydantic validation
        """
        logger.info(f"Analyzing behavioral consistency ({len(answers)} answers)...")

        # Match answers to their questions
        answer_map = cls._match_answers(questions, answers)

        # ── Step 1: Extract risk signals from answers ──────
        risk_signals = cls._extract_risk_signals(answer_map)

        # ── Step 2: Detect contradictions ──────────────────
        contradictions = cls._detect_contradictions(answer_map, risk_signals)

        # ── Step 3: Detect cognitive biases ────────────────
        biases = cls._detect_biases(answer_map, risk_signals, market_stress)

        # ── Step 4: Compute consistency score ──────────────
        consistency = cls._compute_consistency(risk_signals, contradictions)

        # ── Step 5: Assess emotional stability ─────────────
        stability = cls._assess_emotional_stability(
            risk_signals, contradictions, answer_map, historical_sessions
        )

        # ── Step 6: Determine stress response pattern ──────
        stress_pattern = cls._determine_stress_response(
            answer_map, risk_signals, market_stress
        )

        # ── Step 7: Profile decision speed ─────────────────
        speed_profile = cls._profile_decision_speed(answers)

        result = {
            "consistency_score": consistency,
            "emotional_stability": stability["label"],
            "emotional_stability_score": stability["score"],
            "contradiction_flags": contradictions,
            "detected_biases": biases,
            "stress_response_pattern": stress_pattern,
            "decision_speed_profile": speed_profile,
        }

        logger.info(
            f"Behavioral analysis: consistency={consistency:.2f}, "
            f"stability={stability['label']}, "
            f"contradictions={len(contradictions)}, "
            f"biases={len(biases)}"
        )

        return result

    # ────────────────────────────────────────────────────
    #  ANSWER MATCHING
    # ────────────────────────────────────────────────────

    @classmethod
    def _match_answers(cls, questions: list[dict], answers: list[dict]) -> list[dict]:
        """Match each answer to its question and selected choice."""
        q_map = {q["question_id"]: q for q in questions}
        matched = []

        for ans in answers:
            qid = ans.get("question_id", "")
            question = q_map.get(qid)
            if not question:
                continue

            # Find the selected choice
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
                "response_time": ans.get("response_time_seconds"),
                "changed_answer": ans.get("changed_answer", False),
            })

        return matched

    # ────────────────────────────────────────────────────
    #  RISK SIGNAL EXTRACTION
    # ────────────────────────────────────────────────────

    @classmethod
    def _extract_risk_signals(cls, answer_map: list[dict]) -> list[float]:
        """Extract the risk signal from each answered question."""
        return [item["risk_signal"] for item in answer_map]

    # ────────────────────────────────────────────────────
    #  CONTRADICTION DETECTION
    # ────────────────────────────────────────────────────

    @classmethod
    def _detect_contradictions(
        cls, answer_map: list[dict], risk_signals: list[float]
    ) -> list[dict]:
        """
        Detect contradictions in user responses.

        Types:
          - cross_question: Two answers in the same session conflict
          - stress_induced: High-stress questions produce different risk signal
                            than low-stress ones
          - anchoring: User clings to specific numbers regardless of scenario
        """
        contradictions = []

        if len(answer_map) < 2:
            return contradictions

        # ── Cross-question contradictions ──────────────────
        # Compare all pairs of answers
        for i in range(len(answer_map)):
            for j in range(i + 1, len(answer_map)):
                a, b = answer_map[i], answer_map[j]
                signal_diff = abs(a["risk_signal"] - b["risk_signal"])

                # Are these similar categories? If so, contradiction is more meaningful
                same_family = cls._categories_related(a["category"], b["category"])

                threshold = cls.CONTRADICTION_THRESHOLD if same_family else cls.STRONG_CONTRADICTION

                if signal_diff >= threshold:
                    severity = min(1.0, signal_diff / 0.8)
                    contradictions.append({
                        "type": "cross_question",
                        "severity": round(severity, 2),
                        "description": (
                            f"User showed {cls._signal_label(a['risk_signal'])} "
                            f"behavior in {a['category']} scenario but "
                            f"{cls._signal_label(b['risk_signal'])} in {b['category']} -- "
                            f"risk signal gap of {signal_diff:.0%}"
                        ),
                        "question_ids": [
                            a["question"]["question_id"],
                            b["question"]["question_id"],
                        ],
                    })

        # ── Stress-induced contradictions ──────────────────
        # Compare high-difficulty vs low-difficulty answers
        high_stress = [a for a in answer_map if a["difficulty"] >= 0.6]
        low_stress = [a for a in answer_map if a["difficulty"] < 0.4]

        if high_stress and low_stress:
            avg_high = sum(a["risk_signal"] for a in high_stress) / len(high_stress)
            avg_low = sum(a["risk_signal"] for a in low_stress) / len(low_stress)
            stress_gap = abs(avg_high - avg_low)

            if stress_gap >= cls.CONTRADICTION_THRESHOLD:
                direction = "more conservative" if avg_high < avg_low else "more aggressive"
                contradictions.append({
                    "type": "stress_induced",
                    "severity": round(min(1.0, stress_gap / 0.6), 2),
                    "description": (
                        f"User becomes {direction} under stress scenarios "
                        f"(risk signal shift: {stress_gap:.0%})"
                    ),
                    "question_ids": [
                        a["question"]["question_id"]
                        for a in high_stress + low_stress
                    ],
                })

        # ── Anchoring detection ────────────────────────────
        # Check if user always picks the same position (always A, always C, etc.)
        if len(answer_map) >= 3:
            choice_positions = [
                a["answer"].get("selected_choice_id", "") for a in answer_map
            ]
            from collections import Counter
            counter = Counter(choice_positions)
            most_common_id, most_common_count = counter.most_common(1)[0]
            if most_common_count >= len(answer_map) * 0.75 and len(answer_map) >= 3:
                contradictions.append({
                    "type": "anchoring",
                    "severity": 0.6,
                    "description": (
                        f"User consistently selected choice '{most_common_id}' "
                        f"({most_common_count}/{len(answer_map)} times) -- "
                        f"possible anchoring or satisficing behavior"
                    ),
                    "question_ids": [
                        a["question"]["question_id"] for a in answer_map
                    ],
                })

        return contradictions

    # ────────────────────────────────────────────────────
    #  BIAS DETECTION
    # ────────────────────────────────────────────────────

    @classmethod
    def _detect_biases(
        cls,
        answer_map: list[dict],
        risk_signals: list[float],
        market_stress: float,
    ) -> list[dict]:
        """Detect cognitive biases from answer patterns."""
        biases = []

        if not answer_map:
            return biases

        avg_risk = sum(risk_signals) / len(risk_signals)

        # ── Loss Aversion ──────────────────────────────────
        loss_answers = [
            a for a in answer_map if a["category"] == "loss_aversion"
        ]
        if loss_answers:
            loss_risk = sum(a["risk_signal"] for a in loss_answers) / len(loss_answers)
            if loss_risk < avg_risk - 0.15:
                biases.append({
                    "bias_type": "loss_aversion",
                    "strength": round(min(1.0, (avg_risk - loss_risk) / 0.4), 2),
                    "evidence": (
                        f"Risk tolerance drops significantly in loss scenarios "
                        f"(avg signal: {loss_risk:.2f} vs overall: {avg_risk:.2f})"
                    ),
                })

        # ── Overconfidence ─────────────────────────────────
        overconf_answers = [
            a for a in answer_map if a["category"] == "overconfidence"
        ]
        if overconf_answers:
            overconf_risk = sum(a["risk_signal"] for a in overconf_answers) / len(overconf_answers)
            if overconf_risk > 0.75:
                biases.append({
                    "bias_type": "overconfidence",
                    "strength": round(min(1.0, (overconf_risk - 0.5) / 0.4), 2),
                    "evidence": (
                        f"User shows high confidence in market predictions "
                        f"(risk signal: {overconf_risk:.2f} in overconfidence scenarios)"
                    ),
                })

        # ── Herd Behavior ──────────────────────────────────
        herd_answers = [
            a for a in answer_map if a["category"] == "herd_behavior"
        ]
        if herd_answers:
            herd_risk = sum(a["risk_signal"] for a in herd_answers) / len(herd_answers)
            # During high stress, following the herd = low risk signal (panic selling)
            if market_stress > 0.5 and herd_risk < 0.3:
                biases.append({
                    "bias_type": "herd_behavior",
                    "strength": round(min(1.0, (0.5 - herd_risk) / 0.4), 2),
                    "evidence": (
                        f"User follows crowd behavior during market stress "
                        f"(herd risk signal: {herd_risk:.2f} under stress={market_stress:.2f})"
                    ),
                })

        # ── Recency Bias ──────────────────────────────────
        recency_answers = [
            a for a in answer_map if a["category"] == "recency_bias"
        ]
        if recency_answers:
            recency_risk = sum(a["risk_signal"] for a in recency_answers) / len(recency_answers)
            # Strong reaction to recent events
            if abs(recency_risk - 0.5) > 0.25:
                biases.append({
                    "bias_type": "recency_bias",
                    "strength": round(min(1.0, abs(recency_risk - 0.5) / 0.4), 2),
                    "evidence": (
                        f"User heavily weights recent market events in decisions "
                        f"(recency risk signal: {recency_risk:.2f})"
                    ),
                })

        # ── Disposition Effect ─────────────────────────────
        disposition_answers = [
            a for a in answer_map if a["category"] == "disposition_effect"
        ]
        if disposition_answers:
            disp_risk = sum(a["risk_signal"] for a in disposition_answers) / len(disposition_answers)
            if disp_risk > 0.6:
                biases.append({
                    "bias_type": "disposition_effect",
                    "strength": round(min(1.0, (disp_risk - 0.5) / 0.3), 2),
                    "evidence": (
                        f"User tends to hold losing positions and sell winners early "
                        f"(disposition signal: {disp_risk:.2f})"
                    ),
                })

        return biases

    # ────────────────────────────────────────────────────
    #  CONSISTENCY SCORING
    # ────────────────────────────────────────────────────

    @classmethod
    def _compute_consistency(
        cls, risk_signals: list[float], contradictions: list[dict]
    ) -> float:
        """
        Compute internal consistency score (0-1).

        Factors:
          - Variance of risk signals (low variance = consistent)
          - Number and severity of contradictions
        """
        if not risk_signals:
            return 0.5

        # Variance-based consistency
        mean = sum(risk_signals) / len(risk_signals)
        variance = sum((s - mean) ** 2 for s in risk_signals) / len(risk_signals)
        std_dev = math.sqrt(variance)

        # Map std_dev to consistency: 0 std = 1.0 consistency, 0.5 std = 0.0
        variance_consistency = max(0.0, 1.0 - (std_dev / 0.35))

        # Contradiction penalty
        severity_sum = sum(c.get("severity", 0.5) for c in contradictions)
        contradiction_penalty = min(0.4, severity_sum * 0.15)

        consistency = max(0.0, min(1.0, variance_consistency - contradiction_penalty))
        return round(consistency, 2)

    # ────────────────────────────────────────────────────
    #  EMOTIONAL STABILITY
    # ────────────────────────────────────────────────────

    @classmethod
    def _assess_emotional_stability(
        cls,
        risk_signals: list[float],
        contradictions: list[dict],
        answer_map: list[dict],
        historical: Optional[list[dict]] = None,
    ) -> dict:
        """
        Assess emotional stability from response patterns.

        Stability factors:
          - Risk signal variance
          - Contradiction severity
          - Answer changes (indicating hesitation)
          - Response time variance
        """
        if not risk_signals:
            return {"label": "unknown", "score": 0.5}

        # Base score from risk signal variance
        mean = sum(risk_signals) / len(risk_signals)
        variance = sum((s - mean) ** 2 for s in risk_signals) / len(risk_signals)

        # Changed answers indicate emotional volatility
        changes = sum(1 for a in answer_map if a.get("changed_answer", False))
        change_ratio = changes / max(1, len(answer_map))

        # Response time variance (if available)
        times = [
            a.get("response_time") for a in answer_map
            if a.get("response_time") is not None
        ]
        time_variance = 0.0
        if len(times) >= 2:
            mean_time = sum(times) / len(times)
            time_variance = sum((t - mean_time) ** 2 for t in times) / len(times)
            time_variance = min(1.0, time_variance / 100)  # normalize

        # Combine factors
        score = 1.0
        score -= min(0.35, variance * 2)        # risk variance penalty
        score -= min(0.25, change_ratio * 0.5)   # answer change penalty
        score -= min(0.15, time_variance * 0.3)  # time variance penalty
        score -= min(0.15, len(contradictions) * 0.08)  # contradiction penalty

        score = round(max(0.0, min(1.0, score)), 2)

        # Map to label
        if score >= 0.75:
            label = "stable"
        elif score >= 0.50:
            label = "moderate"
        elif score >= 0.30:
            label = "volatile"
        else:
            label = "highly_volatile"

        return {"label": label, "score": score}

    # ────────────────────────────────────────────────────
    #  STRESS RESPONSE PATTERN
    # ────────────────────────────────────────────────────

    @classmethod
    def _determine_stress_response(
        cls, answer_map: list[dict], risk_signals: list[float], market_stress: float
    ) -> str:
        """
        Determine how the user responds under stress.

        Patterns:
          - flight: Becomes very conservative under stress (sell everything)
          - fight: Becomes more aggressive under stress (buy the dip)
          - freeze: Takes no action, picks neutral options
          - adapt: Adjusts rationally based on scenario specifics
        """
        if not answer_map or len(answer_map) < 2:
            return "unknown"

        high_diff = [a for a in answer_map if a["difficulty"] >= 0.6]
        low_diff = [a for a in answer_map if a["difficulty"] < 0.4]

        if not high_diff:
            # Use market_stress as proxy when all questions have similar difficulty
            if market_stress > 0.5:
                avg_signal = sum(risk_signals) / len(risk_signals)
                if avg_signal < 0.3:
                    return "flight"
                elif avg_signal > 0.7:
                    return "fight"
                elif 0.45 <= avg_signal <= 0.55:
                    return "freeze"
                else:
                    return "adapt"
            return "adapt"

        avg_high = sum(a["risk_signal"] for a in high_diff) / len(high_diff)
        avg_low = sum(a["risk_signal"] for a in low_diff) / len(low_diff) if low_diff else 0.5

        diff = avg_high - avg_low

        if diff < -0.20:
            return "flight"     # Much more conservative under stress
        elif diff > 0.20:
            return "fight"      # More aggressive under stress
        elif abs(diff) < 0.05 and 0.4 <= avg_high <= 0.6:
            return "freeze"     # No change, stuck in middle
        else:
            return "adapt"      # Reasonable adjustment

    # ────────────────────────────────────────────────────
    #  DECISION SPEED PROFILING
    # ────────────────────────────────────────────────────

    @classmethod
    def _profile_decision_speed(cls, answers: list[dict]) -> str:
        """
        Profile decision-making speed from response times.

        Categories:
          - deliberate: Avg > 30s (thinks carefully)
          - moderate: 10-30s (balanced)
          - impulsive: < 10s (gut reactions)
        """
        times = [
            a.get("response_time_seconds")
            for a in answers
            if a.get("response_time_seconds") is not None
        ]

        if not times or len(times) < 2:
            return "unknown"

        avg_time = sum(times) / len(times)

        if avg_time > 30:
            return "deliberate"
        elif avg_time > 10:
            return "moderate"
        else:
            return "impulsive"

    # ────────────────────────────────────────────────────
    #  UTILITIES
    # ────────────────────────────────────────────────────

    @staticmethod
    def _categories_related(cat1: str, cat2: str) -> bool:
        """Check if two categories probe similar behavioral dimensions."""
        related_groups = [
            {"loss_aversion", "regret_aversion", "disposition_effect"},
            {"overconfidence", "anchoring", "recency_bias"},
            {"herd_behavior", "time_pressure"},
            {"mental_accounting", "sunk_cost"},
        ]
        for group in related_groups:
            if cat1 in group and cat2 in group:
                return True
        return False

    @staticmethod
    def _signal_label(signal: float) -> str:
        """Convert risk signal to human label."""
        if signal < 0.25:
            return "very conservative"
        elif signal < 0.40:
            return "conservative"
        elif signal < 0.60:
            return "moderate"
        elif signal < 0.75:
            return "growth-oriented"
        else:
            return "aggressive"
