"""
Hybrid Intelligence Portfolio System -- Agent 4 Output Schema
================================================================
Pydantic models for the Meta-Risk & Supervision Agent.

Agent 4 consumes Agent 1 + 2 + 3 and issues a final verdict:
  - approved: Allocation passes all risk checks
  - approved_with_adjustments: Allocation modified to pass guardrails
  - rejected: Allocation fundamentally unsafe, must be reconstructed
"""

from typing import Optional
from pydantic import BaseModel, Field


class RiskAuditResult(BaseModel):
    """Result of a single risk audit check."""
    audit_name: str = Field(..., description="Name of the audit")
    verdict: str = Field(
        ..., description="pass | warning | fail"
    )
    severity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Severity of finding (0=benign, 1=catastrophic)"
    )
    finding: str = Field(default="", description="What was found")
    recommendation: str = Field(default="", description="What should be done")
    details: dict = Field(default_factory=dict, description="Supporting data")


class AdjustedAllocation(BaseModel):
    """Adjusted portfolio allocation after risk intervention."""
    ticker: str = Field(...)
    original_weight: float = Field(..., ge=0.0, le=1.0)
    adjusted_weight: float = Field(..., ge=0.0, le=1.0)
    change: float = Field(default=0.0, description="Adjustment delta")
    reason: str = Field(default="")


class RiskVerdict(BaseModel):
    """LLM-generated Chief Risk Officer verdict."""
    decision: str = Field(
        ..., description="approved | approved_with_adjustments | rejected"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="CRO confidence in this verdict"
    )
    reasoning: str = Field(
        default="", description="Multi-paragraph institutional reasoning"
    )
    critical_risks: list[str] = Field(
        default_factory=list,
        description="Critical risk factors identified"
    )
    mitigations_applied: list[str] = Field(
        default_factory=list,
        description="Risk mitigations applied or recommended"
    )
    residual_risks: list[str] = Field(
        default_factory=list,
        description="Risks that remain even after adjustments"
    )


class Agent4Output(BaseModel):
    """
    Complete Agent 4 output schema.
    Final risk oversight verdict on the proposed portfolio.
    """
    timestamp: str = Field(..., description="UTC timestamp")
    session_id: str = Field(default="")

    # Core verdict
    validation_status: str = Field(
        ...,
        description="approved | approved_with_adjustments | rejected"
    )
    overall_risk_level: str = Field(
        default="moderate",
        description="low | moderate | elevated | high | critical"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall confidence in verdict"
    )

    # Audit trail
    risk_audits: list[RiskAuditResult] = Field(
        default_factory=list,
        description="Results of all 5 independent risk audits"
    )
    total_audits: int = Field(default=5)
    audits_passed: int = Field(default=0)
    audits_warned: int = Field(default=0)
    audits_failed: int = Field(default=0)

    # Adjustments (if any)
    adjusted_allocation: list[AdjustedAllocation] = Field(
        default_factory=list,
        description="Adjusted weights if status is approved_with_adjustments"
    )
    original_allocation_preserved: bool = Field(
        default=True,
        description="Whether the original allocation was kept as-is"
    )

    # LLM verdict
    risk_verdict: RiskVerdict = Field(
        default_factory=lambda: RiskVerdict(decision="approved")
    )

    # Context consumed
    market_regime: str = Field(default="unknown")
    investor_risk_score: float = Field(default=0.5)
    proposed_strategy: str = Field(default="unknown")

    # Metadata
    agent_metadata: dict = Field(default_factory=dict)


def validate_agent4_output(output: dict) -> tuple[bool, Optional[str]]:
    """Validate Agent 4 output against schema."""
    try:
        model = Agent4Output(**output)
        if model.validation_status not in ("approved", "approved_with_adjustments", "rejected"):
            return False, f"Invalid status: {model.validation_status}"
        return True, None
    except Exception as e:
        return False, str(e)
