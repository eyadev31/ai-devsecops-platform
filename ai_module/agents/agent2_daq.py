"""
Hybrid Intelligence Portfolio System -- Agent 2 Orchestrator
===============================================================
Cognitive & Behavioral Profiling Agent (DAQ System)

This agent runs a multi-phase pipeline:
  Phase 1: Question Generation
    - Consume Agent 1 output
    - Calibrate difficulty (QuestionCalibrator)
    - Generate 4 adaptive questions (LLM or fallback bank)

  Phase 2: Answer Processing & Profiling
    - Accept user answers
    - Behavioral consistency analysis (BehavioralConsistencyAnalyzer)
    - Risk classification (AdaptiveRiskClassifier)
    - LLM narrative generation

Both phases can run independently or together (mock mode).
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from ml.question_engine import QuestionCalibrator
from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
from ml.risk_classifier import AdaptiveRiskClassifier
from llm.question_generator import DynamicQuestionGenerator

logger = logging.getLogger(__name__)


class Agent2BehavioralIntelligence:
    """
    Agent 2 -- Cognitive & Behavioral Profiling Agent.

    Produces two outputs:
      1. QuestionSetOutput: Adaptive questions for the user
      2. Agent2Output: Complete behavioral profile after answers
    """

    def __init__(self):
        self._generator = DynamicQuestionGenerator()
        self._execution_log = []

    # ════════════════════════════════════════════════════
    #  PHASE 1: QUESTION GENERATION
    # ════════════════════════════════════════════════════

    def generate_questions(
        self,
        agent1_output: dict,
        user_history: Optional[list[dict]] = None,
        num_questions: int = 4,
    ) -> dict:
        """
        Phase 1: Generate adaptive behavioral questions.

        Args:
            agent1_output: Complete Agent 1 JSON output
            user_history: Previous DAQ sessions
            num_questions: Number of questions to generate

        Returns:
            QuestionSetOutput-compatible dict
        """
        self._log_banner("PHASE 1 -- QUESTION GENERATION")
        start = time.time()

        # ── Step 1: Calibrate ─────────────────────────────
        self._log_step(1, 3, "Calibrating question parameters...")
        calibration = QuestionCalibrator.calibrate(
            agent1_output=agent1_output,
            user_history=user_history,
            num_questions=num_questions,
        )
        self._log_execution("calibration", start)

        # ── Step 2: Generate Questions ────────────────────
        step2_start = time.time()
        self._log_step(2, 3, "Generating adaptive questions via LLM...")
        question_output = self._generator.generate_questions(
            calibration=calibration,
            agent1_output=agent1_output,
            user_history=user_history,
            num_questions=num_questions,
        )
        self._log_execution("question_generation", step2_start)

        # ── Step 3: Validate Output ───────────────────────
        step3_start = time.time()
        self._log_step(3, 3, "Validating question output...")
        n_questions = len(question_output.get("questions", []))
        method = question_output.get("generation_method", "unknown")
        logger.info(
            f"Generated {n_questions} questions via {method} | "
            f"Stress: {calibration.get('stress_multiplier', 0):.0%} | "
            f"Regime: {calibration.get('regime_used', 'N/A')}"
        )
        self._log_execution("validation", step3_start)

        total_ms = (time.time() - start) * 1000
        question_output.setdefault("agent_metadata", {})["phase1_time_ms"] = round(total_ms)
        question_output["agent_metadata"]["execution_log"] = self._execution_log.copy()

        return question_output

    # ════════════════════════════════════════════════════
    #  PHASE 2: ANSWER PROCESSING & PROFILING
    # ════════════════════════════════════════════════════

    def process_answers(
        self,
        questions: list[dict],
        answers: list[dict],
        agent1_output: dict,
        session_id: str = "",
        market_stress: float = 0.5,
        bypass_llm: bool = False,
    ) -> dict:
        """
        Phase 2: Process user answers and generate behavioral profile.

        Args:
            questions: Generated questions from Phase 1
            answers: User answer dicts
            agent1_output: Agent 1 context
            session_id: Links to Phase 1
            market_stress: Stress multiplier

        Returns:
            Agent2Output-compatible dict
        """
        self._log_banner("PHASE 2 -- BEHAVIORAL PROFILING")
        self._execution_log.clear()
        start = time.time()

        # ── Step 1: Behavioral Consistency Analysis ───────
        self._log_step(1, 4, "Analyzing behavioral consistency...")
        step1_start = time.time()
        behavioral_profile = BehavioralConsistencyAnalyzer.analyze(
            questions=questions,
            answers=answers,
            market_stress=market_stress,
        )
        self._log_execution("behavioral_analysis", step1_start)

        # ── Step 2: Risk Classification ───────────────────
        self._log_step(2, 4, "Classifying risk tolerance...")
        step2_start = time.time()
        risk_classification = AdaptiveRiskClassifier.classify(
            questions=questions,
            answers=answers,
            behavioral_profile=behavioral_profile,
            market_stress=market_stress,
            agent1_output=agent1_output,
        )
        self._log_execution("risk_classification", step2_start)

        # ── Step 3: LLM Narrative ─────────────────────────
        self._log_step(3, 4, "Generating behavioral narrative via LLM...")
        step3_start = time.time()

        regime = agent1_output.get("market_regime", {}).get("primary_regime", "unknown")
        
        if bypass_llm:
            llm_narrative = {
                "investor_profile_summary": "MOCK NARRATIVE (LLM BYPASSED)",
                "key_behavioral_traits": ["MOCK"],
                "risk_blind_spots": ["MOCK"],
                "regime_specific_advice": "MOCK",
                "communication_style_preference": "MOCK"
            }
        else:
            market_context = (
                f"Regime: {regime}, "
                f"Vol: {agent1_output.get('volatility_state', {}).get('current_state', 'N/A')}, "
                f"Risk: {agent1_output.get('systemic_risk', {}).get('risk_category', 'N/A')}"
            )

            llm_narrative = self._generator.generate_narrative(
                risk_classification=risk_classification,
                behavioral_profile=behavioral_profile,
                market_context=market_context,
            )
        self._log_execution("llm_narrative", step3_start)

        # ── Step 4: Assemble Output ───────────────────────
        self._log_step(4, 4, "Assembling final profile...")
        total_ms = (time.time() - start) * 1000

        output = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "risk_classification": risk_classification,
            "behavioral_profile": behavioral_profile,
            "llm_narrative": llm_narrative,
            "market_regime_at_assessment": regime,
            "questions_asked": len(questions),
            "answers_processed": len(answers),
            "agent_metadata": {
                "agent_id": "agent2_behavioral_intelligence",
                "version": "1.0.0",
                "execution_time_ms": round(total_ms),
                "llm_calls": self._generator._total_calls,
                "llm_total_latency_ms": round(self._generator._total_latency_ms),
                "models_used": [
                    "question_calibrator_v1",
                    "behavioral_consistency_analyzer_v1",
                    "adaptive_risk_classifier_v1",
                    "llm_core_model",
                ],
                "execution_log": self._execution_log.copy(),
            },
        }

        # Log summary
        logger.info(
            f"{'='*60}\n"
            f"AGENT 2 COMPLETE\n"
            f"  Risk Score:    {risk_classification.get('risk_score', 'N/A')}\n"
            f"  Behavioral:    {risk_classification.get('behavioral_type', 'N/A')}\n"
            f"  Confidence:    {risk_classification.get('confidence', 'N/A')}\n"
            f"  Stability:     {behavioral_profile.get('emotional_stability', 'N/A')}\n"
            f"  Liquidity:     {risk_classification.get('liquidity_preference', 'N/A')}\n"
            f"  Max Drawdown:  {risk_classification.get('max_acceptable_drawdown', 'N/A'):.0%}\n"
            f"  Exec Time:     {total_ms:.0f}ms\n"
            f"{'='*60}"
        )

        return output

    # ════════════════════════════════════════════════════
    #  MOCK MODE -- FULL PIPELINE SIMULATION
    # ════════════════════════════════════════════════════

    def run_mock(self, agent1_output: Optional[dict] = None, bypass_llm: bool = False) -> dict:
        """
        Run full Agent 2 pipeline with mock data.

        This simulates a complete session:
          1. Generate questions from Agent 1 context
          2. Simulate user answers (synthetic investor profile)
          3. Process answers and generate behavioral profile
        """
        self._log_banner("AGENT 2 -- MOCK MODE")

        # Use mock Agent 1 output if not provided
        if agent1_output is None:
            agent1_output = self._get_mock_agent1_output()

        # ── Phase 1: Generate Questions ───────────────────
        question_output = self.generate_questions(
            agent1_output=agent1_output,
            num_questions=4,
        )
        questions = question_output.get("questions", [])
        session_id = question_output.get("session_id", "mock")

        # ── Simulate User Answers ─────────────────────────
        logger.info("Simulating user answers (growth-oriented investor)...")
        answers = self._simulate_answers(questions, profile="growth_seeker")

        # ── Phase 2: Process Answers ──────────────────────
        stress = question_output.get("market_calibration", {}).get("stress_multiplier", 0.5)
        profile_output = self.process_answers(
            questions=questions,
            answers=answers,
            agent1_output=agent1_output,
            session_id=session_id,
            market_stress=stress,
            bypass_llm=bypass_llm,
        )

        # Combine both phases
        combined = {
            "phase1_questions": question_output,
            "phase2_profile": profile_output,
        }

        return combined

    # ────────────────────────────────────────────────────
    #  MOCK HELPERS
    # ────────────────────────────────────────────────────

    @staticmethod
    def _simulate_answers(
        questions: list[dict], profile: str = "moderate"
    ) -> list[dict]:
        """
        Simulate user answers based on an investor profile.

        Profiles:
          - conservative: Tends to pick low risk_signal choices
          - moderate: Picks middle-ground choices
          - growth_seeker: Tends to pick higher risk_signal choices
          - aggressive: Picks highest risk_signal choices
        """
        import random

        profile_targets = {
            "conservative": (0.1, 0.3),
            "moderate": (0.35, 0.55),
            "growth_seeker": (0.55, 0.80),
            "aggressive": (0.75, 0.95),
        }
        target_low, target_high = profile_targets.get(profile, (0.35, 0.55))

        answers = []
        for q in questions:
            choices = q.get("choices", [])
            if not choices:
                continue

            # Find the choice closest to our target range
            target = random.uniform(target_low, target_high)
            best_choice = min(
                choices,
                key=lambda c: abs(c.get("risk_signal", 0.5) - target)
            )

            # Sometimes pick a slightly different choice for realism
            if random.random() < 0.15 and len(choices) > 1:
                alt_choices = [c for c in choices if c["id"] != best_choice["id"]]
                best_choice = random.choice(alt_choices)

            answers.append({
                "question_id": q["question_id"],
                "selected_choice_id": best_choice["id"],
                "response_time_seconds": round(random.uniform(8, 45), 1),
                "changed_answer": random.random() < 0.1,  # 10% chance
            })

        return answers

    @staticmethod
    def _get_mock_agent1_output() -> dict:
        """Return a realistic mock Agent 1 output."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_freshness": datetime.utcnow().isoformat() + "Z",
            "market_regime": {
                "primary_regime": "bull_low_vol",
                "confidence": 0.85,
                "hmm_regime": "bull_low_vol",
                "rf_regime": "bull_low_vol",
                "models_agree": True,
                "regime_duration_days": 45,
                "transition_probability": 0.15,
                "description": "Bullish regime with contained volatility",
            },
            "volatility_state": {
                "current_state": "normal",
                "vix_level": 18.5,
                "realized_vol_percentile": 40.0,
                "vol_trend": "stable",
                "vol_of_vol": "normal",
                "term_structure": "contango",
            },
            "macro_environment": {
                "macro_regime": "stable_growth",
                "monetary_policy": "neutral",
                "inflation_state": "target_range",
                "growth_state": "moderate_growth",
                "liquidity": "neutral",
                "composite_score": 0.3,
                "key_indicators": {
                    "fed_funds_rate": 4.5,
                    "treasury_10y": 4.2,
                    "unemployment": 3.8,
                    "gdp_growth": 2.1,
                },
                "yield_curve": {
                    "inverted": False,
                    "signal": "normal",
                },
            },
            "systemic_risk": {
                "overall_risk_level": 0.15,
                "risk_category": "low",
                "risk_signals": {},
                "risk_assessment": "Low systemic risk environment",
                "recommended_caution": False,
            },
            "cross_asset_analysis": {
                "correlation_state": "normal",
                "median_correlation": 0.25,
                "risk_appetite_index": 0.75,
                "key_correlations": {},
            },
        }

    # ────────────────────────────────────────────────────
    #  LOGGING
    # ────────────────────────────────────────────────────

    def _log_banner(self, title: str):
        logger.info(f"\n{'+'*60}")
        logger.info(f"|  {title:<56}|")
        logger.info(f"|  Hybrid Intelligence Portfolio System v1.0.0{' '*11}|")
        logger.info(f"{'+'*60}")

    @staticmethod
    def _log_step(step: int, total: int, msg: str):
        logger.info(f"STEP {step}/{total} -- {msg}")

    def _log_execution(self, step_name: str, start_time: float):
        duration = (time.time() - start_time) * 1000
        self._execution_log.append({
            "step": step_name,
            "status": "success",
            "duration_ms": round(duration),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
