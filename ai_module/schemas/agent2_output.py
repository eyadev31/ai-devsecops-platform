"""
Hybrid Intelligence Portfolio System -- Agent 2 Output Schema
================================================================
Pydantic models for the Cognitive & Behavioral Profiling Agent.

Agent 2 has TWO output phases:
  Phase 1: Question Generation  -> QuestionSetOutput
  Phase 2: Profile Assessment   -> Agent2Output (full behavioral profile)

These schemas serve as the contract between Agent 2 and Agent 3.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ================================================================
#  PHASE 1: QUESTION GENERATION OUTPUT
# ================================================================

class QuestionChoice(BaseModel):
    """A single answer choice within a question."""
    id: str = Field(..., description="Choice identifier (A, B, C, D)")
    text: str = Field(..., description="Choice text -- scenario-based, not generic")
    risk_signal: float = Field(
        ..., ge=0.0, le=1.0,
        description="Hidden risk signal: 0.0=extremely conservative, 1.0=extremely aggressive"
    )
    behavioral_tag: str = Field(
        default="neutral",
        description="Behavioral bias this choice maps to (e.g., loss_aversion, overconfidence)"
    )


class GeneratedQuestion(BaseModel):
    """A single dynamically generated behavioral finance question."""
    question_id: str = Field(..., description="Unique question identifier")
    category: str = Field(
        ...,
        description="Behavioral category: loss_aversion | herd_behavior | anchoring | "
                    "time_pressure | regret_aversion | overconfidence | recency_bias | "
                    "disposition_effect | mental_accounting | sunk_cost"
    )
    difficulty: float = Field(
        ..., ge=0.0, le=1.0,
        description="Difficulty level: 0.0=easy, 1.0=extreme stress test"
    )
    scenario: str = Field(..., description="Rich scenario description with specific numbers")
    question_text: str = Field(..., description="The actual question to ask the user")
    choices: list[QuestionChoice] = Field(
        ..., min_length=3, max_length=5,
        description="Answer choices (3-5 options)"
    )
    market_context_used: str = Field(
        default="",
        description="Which market data point drove this question's calibration"
    )
    behavioral_insight: str = Field(
        default="",
        description="What this question is designed to reveal (hidden from user)"
    )


class MarketCalibration(BaseModel):
    """How market conditions influenced question generation."""
    stress_multiplier: float = Field(
        ..., ge=0.0, le=1.0,
        description="Market stress level used for calibration"
    )
    regime_used: str = Field(default="unknown", description="Market regime from Agent 1")
    volatility_state: str = Field(default="unknown", description="Vol state from Agent 1")
    risk_level: str = Field(default="unknown", description="Systemic risk from Agent 1")
    calibration_notes: str = Field(
        default="",
        description="Human-readable explanation of how market state shaped questions"
    )


class QuestionSetOutput(BaseModel):
    """Phase 1 output: A set of dynamically generated questions."""
    session_id: str = Field(..., description="Unique session identifier")
    timestamp: str = Field(..., description="UTC timestamp of generation")
    questions: list[GeneratedQuestion] = Field(
        ..., min_length=1, max_length=6,
        description="Generated questions (typically 4)"
    )
    market_calibration: MarketCalibration
    generation_method: str = Field(
        default="llm",
        description="How questions were generated: llm | fallback_bank | hybrid"
    )
    agent_metadata: dict = Field(default_factory=dict)


# ================================================================
#  PHASE 2: BEHAVIORAL PROFILE OUTPUT
# ================================================================

class UserAnswer(BaseModel):
    """A user's answer to a generated question."""
    question_id: str = Field(..., description="References GeneratedQuestion.question_id")
    selected_choice_id: str = Field(..., description="The choice ID the user selected")
    response_time_seconds: Optional[float] = Field(
        None, ge=0.0,
        description="How long the user took to answer (hesitation signal)"
    )
    changed_answer: bool = Field(
        default=False,
        description="Whether the user changed their initial selection"
    )


class ContradictionFlag(BaseModel):
    """A detected contradiction in user behavior."""
    type: str = Field(
        ...,
        description="Contradiction type: cross_question | temporal | stress_induced | anchoring"
    )
    severity: float = Field(
        ..., ge=0.0, le=1.0,
        description="How significant the contradiction is"
    )
    description: str = Field(..., description="Human-readable explanation")
    question_ids: list[str] = Field(
        default_factory=list,
        description="Question IDs involved in the contradiction"
    )


class DetectedBias(BaseModel):
    """A behavioral bias detected from answer patterns."""
    bias_type: str = Field(
        ...,
        description="loss_aversion | overconfidence | anchoring | herd_behavior | "
                    "recency_bias | disposition_effect | mental_accounting | sunk_cost"
    )
    strength: float = Field(
        ..., ge=0.0, le=1.0,
        description="How strongly this bias was detected"
    )
    evidence: str = Field(..., description="Specific answer pattern that revealed this bias")


class BehavioralProfile(BaseModel):
    """ML-generated behavioral consistency analysis."""
    consistency_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How internally consistent the user's answers are (1.0 = perfectly consistent)"
    )
    emotional_stability: str = Field(
        ...,
        description="stable | moderate | volatile | highly_volatile"
    )
    emotional_stability_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Numeric stability score"
    )
    contradiction_flags: list[ContradictionFlag] = Field(
        default_factory=list,
        description="Detected contradictions in answers"
    )
    detected_biases: list[DetectedBias] = Field(
        default_factory=list,
        description="Behavioral biases identified"
    )
    stress_response_pattern: str = Field(
        default="unknown",
        description="How user responds under stress: flight | fight | freeze | adapt"
    )
    decision_speed_profile: str = Field(
        default="unknown",
        description="deliberate | moderate | impulsive"
    )


class RiskClassification(BaseModel):
    """ML-generated risk classification."""
    risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Continuous risk tolerance: 0.0=ultra-conservative, 1.0=ultra-aggressive"
    )
    risk_score_raw: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw risk score before market adjustment"
    )
    market_adjusted: bool = Field(
        default=True,
        description="Whether risk score was adjusted for current market conditions"
    )
    behavioral_type: str = Field(
        ...,
        description="Behavioral classification: conservative_stable | conservative_anxious | "
                    "moderate_balanced | moderate_volatile | growth_seeker | "
                    "growth_seeker_with_volatility_sensitivity | aggressive_speculator | "
                    "aggressive_contrarian"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the classification"
    )
    liquidity_preference: str = Field(
        ...,
        description="high | medium | low -- inferred need for liquid assets"
    )
    time_horizon: str = Field(
        default="medium",
        description="short (<1y) | medium (1-5y) | long (5y+) -- inferred investment horizon"
    )
    max_acceptable_drawdown: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Maximum drawdown the user can psychologically tolerate"
    )


class LLMBehavioralNarrative(BaseModel):
    """LLM-generated interpretation of behavioral analysis."""
    investor_narrative: str = Field(
        default="",
        description="2-3 sentence institutional-quality description of investor psychology"
    )
    key_insights: list[str] = Field(
        default_factory=list,
        description="Key behavioral insights discovered"
    )
    risk_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about potential behavioral pitfalls"
    )
    recommended_guardrails: list[str] = Field(
        default_factory=list,
        description="Suggestions for portfolio guardrails based on behavior"
    )


class Agent2Output(BaseModel):
    """
    Complete Agent 2 output schema.
    Contract between Agent 2 and Agent 3 (Strategic Allocation).
    """
    session_id: str = Field(..., description="Links to QuestionSetOutput.session_id")
    timestamp: str = Field(..., description="UTC timestamp of assessment")

    # Core outputs
    risk_classification: RiskClassification
    behavioral_profile: BehavioralProfile
    llm_narrative: LLMBehavioralNarrative

    # Context
    market_regime_at_assessment: str = Field(
        default="unknown",
        description="Market regime active during this assessment"
    )
    questions_asked: int = Field(default=4, ge=1, description="Number of questions in session")
    answers_processed: int = Field(default=0, ge=0, description="Number of answers processed")

    # Metadata
    agent_metadata: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "daq_20260224_001",
                "timestamp": "2026-02-24T22:00:00Z",
                "risk_classification": {
                    "risk_score": 0.71,
                    "risk_score_raw": 0.75,
                    "market_adjusted": True,
                    "behavioral_type": "growth_seeker_with_volatility_sensitivity",
                    "confidence": 0.88,
                    "liquidity_preference": "medium",
                    "time_horizon": "medium",
                    "max_acceptable_drawdown": 0.20,
                },
                "behavioral_profile": {
                    "consistency_score": 0.82,
                    "emotional_stability": "moderate",
                    "emotional_stability_score": 0.65,
                    "stress_response_pattern": "adapt",
                    "decision_speed_profile": "deliberate",
                },
            }
        }


def validate_question_output(output: dict) -> tuple[bool, Optional[str]]:
    """Validate Phase 1 output."""
    try:
        QuestionSetOutput(**output)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_profile_output(output: dict) -> tuple[bool, Optional[str]]:
    """Validate Phase 2 output."""
    try:
        Agent2Output(**output)
        return True, None
    except Exception as e:
        return False, str(e)
