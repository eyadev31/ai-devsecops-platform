"""
Hybrid Intelligence Portfolio System -- Dynamic Question Generator
====================================================================
LLM-powered adaptive question generation for behavioral profiling.

This module:
  1. Takes calibration from QuestionCalibrator + Agent 1 context
  2. Calls LLM to generate 4 scenario-based behavioral questions
  3. Validates and parses the structured JSON output
  4. Falls back to a pre-built question bank if LLM fails
"""

import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Optional

from config.settings import APIKeys
from llm.gemini_client import BaseLLMClient, LLMFactory

logger = logging.getLogger(__name__)


# ================================================================
#  AGENT 2 PROMPTS
# ================================================================

AGENT2_SYSTEM_PROMPT = """You are a Senior Behavioral Finance Intelligence Agent, 
operating within an institutional-grade portfolio management system.

Your specialty is behavioral finance and investor psychology. You design 
psychologically calibrated assessment questions that reveal true risk 
tolerance, cognitive biases, and emotional stability under market stress.

CRITICAL RULES:
1. You are NOT a survey bot. Your questions must be scenario-based, 
   specific, and psychologically revealing.
2. Every question must include realistic numbers (drawdowns, returns, 
   timeframes) based on the current market context provided.
3. Each answer choice must have a hidden risk_signal (0.0-1.0) that 
   maps to the user's true risk tolerance.
4. Questions must be personalized when user history is available.
5. NEVER ask generic survey questions like "How do you feel about risk?"
6. Always output valid JSON. No markdown, no explanations outside JSON."""


DAQ_QUESTION_GENERATION_PROMPT = """Generate exactly {num_questions} behavioral finance assessment questions.

## CURRENT MARKET CONTEXT (from Agent 1):
{market_context}

## CALIBRATION PARAMETERS:
- Stress Level: {stress_multiplier:.0%} (0%=calm, 100%=crisis)
- Difficulty Range: {difficulty_low:.0%} to {difficulty_high:.0%}
- Categories to probe: {categories}

## SCENARIO PARAMETERS (use these real numbers):
{scenario_params}

{user_history_section}

## REQUIREMENTS FOR EACH QUESTION:
1. Rich scenario: 2-3 sentences describing a specific market situation with REAL numbers
2. Question text: Direct question about what the user would DO (not feel)
3. Exactly 4 answer options (A, B, C, D)
4. Each option must have:
   - Concrete action (not vague)
   - risk_signal: float 0.0-1.0 (0=most conservative, 1=most aggressive)
   - behavioral_tag: which bias this choice reveals

## OUTPUT FORMAT (strict JSON):
{{
  "questions": [
    {{
      "question_id": "q_{session_prefix}_1",
      "category": "loss_aversion",
      "difficulty": 0.65,
      "scenario": "Your diversified portfolio has dropped 18% in the last 3 weeks as the S&P 500 enters bear territory. VIX is at 32 and rising. Your financial advisor calls to discuss options.",
      "question_text": "What do you instruct your advisor to do?",
      "options": [
        {{
          "value": "A",
          "text": "Sell everything immediately and move to cash until markets stabilize",
          "risk_signal": 0.05,
          "behavioral_tag": "loss_aversion"
        }},
        {{
          "value": "B",
          "text": "Reduce equity exposure by 30% and shift to defensive sectors",
          "risk_signal": 0.30,
          "behavioral_tag": "moderate_risk"
        }},
        {{
          "value": "C",
          "text": "Hold current positions but stop checking the portfolio daily",
          "risk_signal": 0.60,
          "behavioral_tag": "neutral"
        }},
        {{
          "value": "D",
          "text": "Add to positions -- this is a buying opportunity at a discount",
          "risk_signal": 0.90,
          "behavioral_tag": "overconfidence"
        }}
      ],
      "market_context_used": "VIX at 32 with bear regime detected",
      "behavioral_insight": "Tests loss aversion: whether user panics during drawdowns or maintains strategy"
    }}
  ]
}}

Generate exactly {num_questions} questions. Each must probe a DIFFERENT category from the list above.
Use the market data provided -- DO NOT invent fake numbers when real ones are given."""


BEHAVIORAL_INTERPRETATION_PROMPT = """Analyze this investor's behavioral profile and provide an institutional-quality assessment.

## RISK CLASSIFICATION:
{risk_data}

## BEHAVIORAL PROFILE:
{behavioral_data}

## MARKET CONTEXT DURING ASSESSMENT:
{market_context}

Provide your analysis in this JSON format:
{{
    "investor_narrative": "2-3 sentence institutional-quality summary of this investor's psychology, referencing specific behavioral signals",
    "key_insights": [
        "Specific insight 1 with data reference",
        "Specific insight 2 with data reference"
    ],
    "risk_warnings": [
        "Warning about a specific behavioral pitfall"
    ],
    "recommended_guardrails": [
        "Specific portfolio guardrail to protect against behavioral weaknesses"
    ]
}}

Be precise. Reference specific numbers. No motivational language."""


class DynamicQuestionGenerator:
    """
    Generates adaptive behavioral finance questions using LLM.

    Pipeline:
      1. Format market context + calibration into prompt
      2. Call LLM for question generation
      3. Parse and validate JSON output
      4. Fallback to pre-built bank if LLM fails
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client or LLMFactory.create()
        self._total_calls = 0
        self._total_latency_ms = 0.0

    def generate_questions(
        self,
        calibration: dict,
        agent1_output: dict,
        user_history: Optional[list[dict]] = None,
        num_questions: int = 4,
    ) -> dict:
        """
        Generate adaptive questions.

        Args:
            calibration: From QuestionCalibrator.calibrate()
            agent1_output: Complete Agent 1 output
            user_history: Previous session data
            num_questions: Number of questions to generate

        Returns:
            QuestionSetOutput-compatible dict
        """
        session_id = f"daq_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Try LLM generation first
        questions = None
        generation_method = "llm"

        if self._llm.is_available():
            try:
                logger.info("Generating questions via LLM...")
                questions = self._generate_via_llm(
                    calibration, agent1_output, user_history,
                    num_questions, session_id
                )
            except Exception as e:
                logger.warning(f"LLM question generation failed: {e}")

        # Fallback to pre-built question bank
        if not questions:
            logger.info("Using fallback question bank...")
            questions = self._generate_fallback(
                calibration, num_questions, session_id
            )
            generation_method = "fallback_bank"

        output = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "questions": questions,
            "market_calibration": {
                "stress_multiplier": calibration.get("stress_multiplier", 0.5),
                "regime_used": calibration.get("regime_used", "unknown"),
                "volatility_state": calibration.get("volatility_state", "unknown"),
                "risk_level": calibration.get("risk_level", "unknown"),
                "calibration_notes": calibration.get("calibration_notes", ""),
            },
            "generation_method": generation_method,
            "agent_metadata": {
                "agent_id": "agent2_behavioral_intelligence",
                "llm_calls": self._total_calls,
                "llm_latency_ms": self._total_latency_ms,
            },
        }

        return output

    def generate_narrative(
        self,
        risk_classification: dict,
        behavioral_profile: dict,
        market_context: str = "",
    ) -> dict:
        """
        Generate LLM behavioral narrative from ML analysis.

        Returns:
            LLMBehavioralNarrative-compatible dict
        """
        try:
            prompt = BEHAVIORAL_INTERPRETATION_PROMPT.format(
                risk_data=json.dumps(risk_classification, indent=2, default=str),
                behavioral_data=json.dumps(behavioral_profile, indent=2, default=str),
                market_context=market_context,
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt=AGENT2_SYSTEM_PROMPT,
                json_mode=True,
            )
            self._total_calls += 1
            self._total_latency_ms += response.get("latency_ms", 0)

            return self._parse_json(response["content"], "behavioral_narrative")

        except Exception as e:
            logger.error(f"Behavioral narrative generation failed: {e}")
            return {
                "investor_narrative": (
                    f"Quantitative analysis indicates a {risk_classification.get('behavioral_type', 'moderate')} "
                    f"investor profile with risk score {risk_classification.get('risk_score', 0.5):.2f}."
                ),
                "key_insights": [],
                "risk_warnings": [],
                "recommended_guardrails": [],
            }

    # ────────────────────────────────────────────────────
    #  LLM GENERATION
    # ────────────────────────────────────────────────────

    def _generate_via_llm(
        self,
        calibration: dict,
        agent1_output: dict,
        user_history: Optional[list[dict]],
        num_questions: int,
        session_id: str,
    ) -> list[dict]:
        """Generate questions via LLM."""
        # Format market context
        market_context = self._format_market_context(agent1_output)

        # Format categories
        categories_str = "\n".join([
            f"  - {c['name']}: {c['description']} (difficulty: {c['difficulty']:.0%})"
            for c in calibration.get("categories", [])
        ])

        # Format scenario params
        params = calibration.get("scenario_params", {})
        scenario_str = json.dumps(params, indent=2, default=str)

        # User history section
        history_section = ""
        if user_history:
            history_section = self._format_user_history(user_history)

        # Difficulty range
        diff_range = calibration.get("difficulty_range", (0.3, 0.7))
        
        # Format categories string
        cats = calibration.get("categories", [])
        categories_str = ", ".join([c.get("name", "") for c in cats]) if cats else "loss_aversion, herd_behavior"

        prompt = DAQ_QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            market_context=market_context,
            stress_multiplier=calibration.get("stress_multiplier", 0.5),
            difficulty_low=diff_range[0],
            difficulty_high=diff_range[1],
            categories=categories_str,
            scenario_params=scenario_str,
            user_history_section=history_section,
            session_prefix=session_id[-6:],
        )

        response = self._llm.generate(
            prompt=prompt,
            system_prompt=AGENT2_SYSTEM_PROMPT,
            json_mode=True,
        )
        self._total_calls += 1
        self._total_latency_ms += response.get("latency_ms", 0)

        parsed = self._parse_json(response["content"], "question_generation")
        questions = parsed.get("questions", [])

        # Validate each question
        validated = []
        for q in questions:
            if self._validate_question(q):
                validated.append(q)

        if not validated:
            logger.warning("No valid questions from LLM, falling back")
            return None

        return validated

    # ────────────────────────────────────────────────────
    #  FALLBACK QUESTION BANK
    # ────────────────────────────────────────────────────

    @classmethod
    def _generate_fallback(
        cls,
        calibration: dict,
        num_questions: int,
        session_id: str,
    ) -> list[dict]:
        """
        Pre-built question bank for when LLM is unavailable.
        These are high-quality, scenario-based questions designed
        by behavioral finance principles.
        """
        stress = calibration.get("stress_multiplier", 0.5)
        categories = calibration.get("categories", [])
        params = calibration.get("scenario_params", {})

        dd_low, dd_high = params.get("drawdown_range_pct", (-20, -8))
        regime = params.get("current_regime", "unknown")

        bank = [
            {
                "category": "loss_aversion",
                "scenario": (
                    f"Markets are in a {regime} regime. Your portfolio has declined "
                    f"{abs(dd_high)}% over the past month, and analysts predict a further "
                    f"{abs(dd_low)}% decline is possible before recovery."
                ),
                "question_text": "What action do you take with your portfolio right now?",
                "options": [
                    {"value": "A", "text": "Liquidate all equity positions immediately and move to money market funds", "risk_signal": 0.05, "behavioral_tag": "loss_aversion"},
                    {"value": "B", "text": "Reduce equity by 40% and add Treasury bonds for safety", "risk_signal": 0.25, "behavioral_tag": "moderate_risk"},
                    {"value": "C", "text": "Maintain current allocation but set stop-losses at 10% below current levels", "risk_signal": 0.55, "behavioral_tag": "neutral"},
                    {"value": "D", "text": "Increase equity exposure by 15% -- market fear creates opportunity", "risk_signal": 0.85, "behavioral_tag": "contrarian"},
                ],
                "behavioral_insight": "Core loss aversion test: measures panic threshold during drawdowns",
            },
            {
                "category": "herd_behavior",
                "scenario": (
                    f"A viral social media trend is driving a 40% rally in a speculative "
                    f"asset class. Your colleagues and friends are all investing. Financial "
                    f"news covers it 24/7. The asset has no fundamental backing."
                ),
                "question_text": "How do you respond to this market frenzy?",
                "options": [
                    {"value": "A", "text": "Ignore it completely -- speculative assets don't belong in a serious portfolio", "risk_signal": 0.10, "behavioral_tag": "disciplined"},
                    {"value": "B", "text": "Research it thoroughly, then allocate a small 2-3% position if fundamentals support it", "risk_signal": 0.40, "behavioral_tag": "moderate_risk"},
                    {"value": "C", "text": "Invest 10% of your portfolio -- there's clearly momentum here", "risk_signal": 0.70, "behavioral_tag": "herd_behavior"},
                    {"value": "D", "text": "Go all-in on the trend -- you don't want to miss the next big thing", "risk_signal": 0.95, "behavioral_tag": "FOMO"},
                ],
                "behavioral_insight": "Tests susceptibility to herd behavior and FOMO under social pressure",
            },
            {
                "category": "time_pressure",
                "scenario": (
                    f"Breaking news: A major geopolitical event causes markets to drop 5% "
                    f"in 30 minutes. Your broker calls and says you have 10 minutes to "
                    f"decide before the next circuit breaker halts trading."
                ),
                "question_text": "With only 10 minutes to decide, what do you do?",
                "options": [
                    {"value": "A", "text": "Sell everything before the circuit breaker hits -- protect what's left", "risk_signal": 0.05, "behavioral_tag": "panic"},
                    {"value": "B", "text": "Sell half your positions to reduce exposure while keeping some upside", "risk_signal": 0.35, "behavioral_tag": "moderate_risk"},
                    {"value": "C", "text": "Do nothing -- refuse to make decisions under time pressure", "risk_signal": 0.55, "behavioral_tag": "freeze"},
                    {"value": "D", "text": "Place limit buy orders at 8% below current prices to catch the dip", "risk_signal": 0.90, "behavioral_tag": "contrarian"},
                ],
                "behavioral_insight": "Tests decision quality under extreme time pressure and panic conditions",
            },
            {
                "category": "regret_aversion",
                "scenario": (
                    f"Last month you sold a stock at $50 that has since risen to $75. "
                    f"You originally bought it at $30. A new analysis suggests it could "
                    f"reach $100, but there's also a 30% chance it drops back to $50."
                ),
                "question_text": "Do you buy back into the stock?",
                "options": [
                    {"value": "A", "text": "No -- buying at $75 what you sold at $50 is psychologically impossible", "risk_signal": 0.15, "behavioral_tag": "regret_aversion"},
                    {"value": "B", "text": "Buy a small position -- you'd regret missing $100 more than losing on $50", "risk_signal": 0.50, "behavioral_tag": "moderate_risk"},
                    {"value": "C", "text": "Buy the full position back -- the original sale was wrong, admit it and move on", "risk_signal": 0.75, "behavioral_tag": "rational"},
                    {"value": "D", "text": "Buy even more than you originally held -- double down on the thesis", "risk_signal": 0.90, "behavioral_tag": "overconfidence"},
                ],
                "behavioral_insight": "Tests regret aversion: can investor separate past decisions from current opportunities",
            },
            {
                "category": "overconfidence",
                "scenario": (
                    f"You've successfully timed the market 3 times this year, beating the S&P 500 "
                    f"by 8%. A friend suggests your 60/40 allocation is too conservative and you "
                    f"should go 90/10 given your track record."
                ),
                "question_text": "How do you adjust your strategy based on this year's success?",
                "options": [
                    {"value": "A", "text": "Keep the 60/40 -- past performance doesn't predict future results", "risk_signal": 0.20, "behavioral_tag": "disciplined"},
                    {"value": "B", "text": "Slightly increase to 70/30 -- you've earned some extra risk", "risk_signal": 0.50, "behavioral_tag": "moderate_risk"},
                    {"value": "C", "text": "Go 80/20 -- your skill is clearly above average", "risk_signal": 0.75, "behavioral_tag": "overconfidence"},
                    {"value": "D", "text": "Go 90/10 with leverage -- maximize your edge while it lasts", "risk_signal": 0.95, "behavioral_tag": "extreme_overconfidence"},
                ],
                "behavioral_insight": "Tests overconfidence bias: does investor mistake luck for skill",
            },
            {
                "category": "anchoring",
                "scenario": (
                    f"You bought shares at $100. The stock dropped to $60, then recovered to $85. "
                    f"Fundamentals suggest fair value is $80. Your original $100 purchase price "
                    f"keeps influencing your thinking."
                ),
                "question_text": "At what price would you sell the stock?",
                "options": [
                    {"value": "A", "text": "Sell now at $85 -- take the loss and move on to better opportunities", "risk_signal": 0.40, "behavioral_tag": "rational"},
                    {"value": "B", "text": "Hold until $100 -- you need to at least break even", "risk_signal": 0.30, "behavioral_tag": "anchoring"},
                    {"value": "C", "text": "Hold until $120 -- you deserve a profit for enduring the drawdown", "risk_signal": 0.60, "behavioral_tag": "anchoring"},
                    {"value": "D", "text": "Buy more at $85 since fair value is $80 -- the anchoring to $100 is irrelevant", "risk_signal": 0.80, "behavioral_tag": "contrarian"},
                ],
                "behavioral_insight": "Tests anchoring to purchase price: can investor evaluate based on current fundamentals",
            },
            {
                "category": "recency_bias",
                "scenario": (
                    f"Markets have been rallying for 6 months with the S&P 500 up 25%. "
                    f"Valuations are stretched at 22x earnings. Your portfolio is heavily "
                    f"in tech stocks that drove most of the gains."
                ),
                "question_text": "How do you position your portfolio for the next 12 months?",
                "options": [
                    {"value": "A", "text": "Rotate entirely into value stocks and bonds -- this rally can't continue", "risk_signal": 0.15, "behavioral_tag": "fearful"},
                    {"value": "B", "text": "Rebalance to reduce tech concentration and add defensive sectors", "risk_signal": 0.40, "behavioral_tag": "moderate_risk"},
                    {"value": "C", "text": "Hold current positions -- momentum is your friend and trends persist", "risk_signal": 0.70, "behavioral_tag": "recency_bias"},
                    {"value": "D", "text": "Add more to the winners -- the best performers will keep outperforming", "risk_signal": 0.90, "behavioral_tag": "momentum_chasing"},
                ],
                "behavioral_insight": "Tests recency bias: overweighting recent performance in forward-looking decisions",
            },
            {
                "category": "disposition_effect",
                "scenario": (
                    f"You have two positions: Stock A is up 40% and Stock B is down 25%. "
                    f"You need to raise cash for an emergency. Analysts rate both stocks "
                    f"as 'Hold' with similar forward prospects."
                ),
                "question_text": "Which stock do you sell to raise the cash?",
                "options": [
                    {"value": "A", "text": "Sell Stock A (winner) -- lock in the guaranteed profit", "risk_signal": 0.30, "behavioral_tag": "disposition_effect"},
                    {"value": "B", "text": "Sell half of each -- spread the decision across both", "risk_signal": 0.50, "behavioral_tag": "compromise"},
                    {"value": "C", "text": "Sell Stock B (loser) -- cut losses and let winners run", "risk_signal": 0.70, "behavioral_tag": "rational"},
                    {"value": "D", "text": "Sell neither -- find cash elsewhere, even if it means borrowing", "risk_signal": 0.60, "behavioral_tag": "sunk_cost"},
                ],
                "behavioral_insight": "Tests disposition effect: tendency to sell winners and hold losers",
            },
        ]

        # Select questions matching the requested categories
        selected = []
        target_cats = [c["name"] for c in categories] if categories else []

        # Match by category first
        for cat_name in target_cats:
            for q in bank:
                if q["category"] == cat_name and q not in selected:
                    selected.append(q)
                    break

        # Fill remaining slots randomly
        remaining = [q for q in bank if q not in selected]
        random.shuffle(remaining)
        while len(selected) < num_questions and remaining:
            selected.append(remaining.pop(0))

        # Add question IDs and difficulty
        for i, q in enumerate(selected):
            cat_info = next(
                (c for c in categories if c["name"] == q["category"]),
                None
            )
            q["question_id"] = f"q_{session_id[-6:]}_{i+1}"
            q["difficulty"] = cat_info["difficulty"] if cat_info else round(0.3 + stress * 0.4, 2)
            q["market_context_used"] = f"Market regime: {regime}"

        return selected[:num_questions]

    # ────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────

    @staticmethod
    def _format_market_context(agent1_output: dict) -> str:
        """Format Agent 1 output for question generation prompt."""
        regime = agent1_output.get("market_regime", {})
        vol = agent1_output.get("volatility_state", {})
        risk = agent1_output.get("systemic_risk", {})
        macro = agent1_output.get("macro_environment", {})

        return (
            f"Market Regime: {regime.get('primary_regime', 'unknown')} "
            f"(confidence: {regime.get('confidence', 0):.0%}, "
            f"models agree: {regime.get('models_agree', 'N/A')})\n"
            f"Volatility: {vol.get('current_state', 'unknown')} "
            f"(VIX: {vol.get('vix_level', 'N/A')}, trend: {vol.get('vol_trend', 'N/A')})\n"
            f"Systemic Risk: {risk.get('risk_category', 'unknown')} "
            f"(level: {risk.get('overall_risk_level', 0):.2f})\n"
            f"Macro Regime: {macro.get('macro_regime', 'unknown')} "
            f"(monetary: {macro.get('monetary_policy', 'N/A')}, "
            f"inflation: {macro.get('inflation_state', 'N/A')})\n"
            f"Yield Curve: {'INVERTED' if macro.get('yield_curve', {}).get('inverted') else 'Normal'}"
        )

    @staticmethod
    def _format_user_history(user_history: list[dict]) -> str:
        """Format user history for prompt."""
        if not user_history:
            return ""

        section = "\n## PREVIOUS USER DATA:\n"
        for i, session in enumerate(user_history[-2:]):  # Last 2 sessions
            risk_score = session.get("risk_score", "N/A")
            beh_type = session.get("behavioral_type", "N/A")
            section += f"Session {i+1}: risk_score={risk_score}, type={beh_type}\n"

        section += "\nPersonalize questions based on this history. Avoid repeating similar scenarios.\n"
        return section

    @staticmethod
    def _validate_question(q: dict) -> bool:
        """Validate a generated question has required fields."""
        required = ["question_id", "category", "scenario", "question_text", "options"]
        if not all(k in q for k in required):
            return False

        if not isinstance(q.get("options"), list) or len(q["options"]) < 3:
            return False

        for choice in q["options"]:
            if not all(k in choice for k in ["value", "text", "risk_signal"]):
                return False

        return True

    @staticmethod
    def _parse_json(content: str, stage: str) -> dict:
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

            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx + 1]

            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON ({stage}): {e}")
            return {}
