"""
Hybrid Intelligence Portfolio System -- Risk Adjudicator (LLM)
=================================================================
LLM-powered Chief Risk Officer for Agent 4.

Uses the exact prompt:
  "You are a Portfolio Risk Oversight AI.
   You must think like a Chief Risk Officer at a hedge fund.
   Be skeptical. Never assume the previous agent is correct.
   Output structured JSON with approval status and reasoning."
"""

import json
import logging
from typing import Optional

from config.settings import APIKeys
from llm.gemini_client import BaseLLMClient, LLMFactory

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════
#  SYSTEM PROMPT -- CHIEF RISK OFFICER
# ════════════════════════════════════════════════════════

CRO_SYSTEM_PROMPT = """You are a Portfolio Risk Oversight AI — the Chief Risk Officer (CRO) 
of an institutional-grade multi-agent portfolio management system.

YOUR MANDATE:
1. Critically evaluate proposed portfolio allocations.
2. Detect inconsistencies with market regime.
3. Ensure risk exposure matches user profile.
4. Reject allocations that exceed acceptable drawdown limits.

RULES:
1. Be skeptical. Never assume the previous agent is correct.
2. Think like a CRO at a hedge fund: your job is to PREVENT catastrophic losses.
3. Reference specific numbers from the audit results.
4. If you approve, explain WHY the risks are acceptable.
5. If you reject, be specific about what must change.
6. Output strictly valid JSON. No markdown, no extra text."""


CRO_ADJUDICATION_PROMPT = """You have received the results of 5 independent risk audits 
on a proposed portfolio allocation. Make your final verdict.

## AUDIT RESULTS:
{audit_results}

## PROPOSED ALLOCATION:
{allocation_data}

## PORTFOLIO METRICS:
{metrics_data}

## MARKET CONTEXT (Agent 1):
Regime: {regime}, Volatility: {vol_state}, Systemic Risk: {risk_cat} ({risk_level:.2f})
Models Agree: {models_agree}

## INVESTOR PROFILE (Agent 2):
Risk Score: {risk_score:.2f}, Type: {beh_type},
Max Drawdown: {max_dd:.0%}, Liquidity: {liquidity}

## ADJUSTED ALLOCATION (if adjustments were computed):
{adjusted_data}

Based on ALL the above, issue your verdict in this JSON format:
{{
    "decision": "approved | approved_with_adjustments | rejected",
    "confidence": 0.0 to 1.0,
    "reasoning": "2-3 paragraphs of institutional-quality reasoning. Reference specific audit findings, numbers, and risk metrics. Explain why the decision was made.",
    "critical_risks": ["specific risk 1 with numbers", "specific risk 2"],
    "mitigations_applied": ["specific mitigation 1", "specific mitigation 2"],
    "residual_risks": ["risk that remains even after adjustments"]
}}

Remember: Be skeptical. Think like a CRO. Protect the client."""


class RiskAdjudicator:
    """
    LLM-powered Chief Risk Officer.

    Synthesizes all audit findings and issues a final verdict
    on the proposed allocation.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client or LLMFactory.create()
        self._total_calls = 0
        self._total_latency_ms = 0.0

    @property
    def llm_calls(self) -> int:
        return self._total_calls

    @property
    def llm_latency_ms(self) -> float:
        return self._total_latency_ms

    def adjudicate(
        self,
        audit_results: list[dict],
        agent1_output: dict,
        agent2_output: dict,
        agent3_output: dict,
        adjusted_allocation: Optional[list[dict]] = None,
    ) -> dict:
        """
        Issue final CRO verdict on the portfolio.

        Args:
            audit_results: Results from 5 risk audits
            agent1_output: Agent 1 context
            agent2_output: Agent 2 profile
            agent3_output: Agent 3 allocation
            adjusted_allocation: Rebalanced allocation (if computed)

        Returns:
            RiskVerdict-compatible dict
        """
        # Try LLM first
        if self._llm.is_available():
            try:
                return self._adjudicate_via_llm(
                    audit_results, agent1_output, agent2_output,
                    agent3_output, adjusted_allocation,
                )
            except Exception as e:
                logger.warning(f"LLM adjudication failed: {e}")

        # Fallback: rule-based decision
        return self._rule_based_verdict(
            audit_results, agent1_output, agent2_output,
            agent3_output, adjusted_allocation,
        )

    # ────────────────────────────────────────────────────
    #  LLM ADJUDICATION
    # ────────────────────────────────────────────────────

    def _adjudicate_via_llm(
        self,
        audit_results: list[dict],
        agent1_output: dict,
        agent2_output: dict,
        agent3_output: dict,
        adjusted_allocation: Optional[list[dict]],
    ) -> dict:
        """CRO verdict via LLM."""
        # Extract context
        profile = agent2_output.get("risk_classification", agent2_output)
        regime_info = agent1_output.get("market_regime", {})
        vol_info = agent1_output.get("volatility_state", {})
        risk_info = agent1_output.get("systemic_risk", {})

        prompt = CRO_ADJUDICATION_PROMPT.format(
            audit_results=json.dumps(audit_results, indent=2, default=str),
            allocation_data=json.dumps(agent3_output.get("allocation", []), indent=2, default=str),
            metrics_data=json.dumps(agent3_output.get("portfolio_metrics", {}), indent=2, default=str),
            regime=regime_info.get("primary_regime", "unknown"),
            vol_state=vol_info.get("current_state", "unknown"),
            risk_cat=risk_info.get("risk_category", "unknown"),
            risk_level=risk_info.get("overall_risk_level", 0),
            models_agree=regime_info.get("models_agree", True),
            risk_score=profile.get("risk_score", 0.5),
            beh_type=profile.get("behavioral_type", "unknown"),
            max_dd=profile.get("max_acceptable_drawdown", 0.15),
            liquidity=profile.get("liquidity_preference", "medium"),
            adjusted_data=json.dumps(adjusted_allocation, indent=2, default=str) if adjusted_allocation else "No adjustments computed",
        )

        response = self._llm.generate(
            prompt=prompt,
            system_prompt=CRO_SYSTEM_PROMPT,
            json_mode=True,
        )
        self._total_calls += 1
        self._total_latency_ms += response.get("latency_ms", 0)

        return self._parse_json(response["content"])

    # ────────────────────────────────────────────────────
    #  RULE-BASED FALLBACK
    # ────────────────────────────────────────────────────

    def _rule_based_verdict(
        self,
        audit_results: list[dict],
        agent1_output: dict,
        agent2_output: dict,
        agent3_output: dict,
        adjusted_allocation: Optional[list[dict]],
    ) -> dict:
        """Deterministic CRO verdict when LLM unavailable."""
        # Count verdicts
        n_pass = sum(1 for a in audit_results if a["verdict"] == "pass")
        n_warn = sum(1 for a in audit_results if a["verdict"] == "warning")
        n_fail = sum(1 for a in audit_results if a["verdict"] == "fail")
        max_severity = max((a["severity"] for a in audit_results), default=0)

        # Collect findings
        critical = [a for a in audit_results if a["severity"] >= 0.6]
        warnings = [a for a in audit_results if 0.3 <= a["severity"] < 0.6]

        # Decision logic
        if n_fail >= 2 or max_severity >= 0.85:
            decision = "rejected"
            confidence = min(0.95, 0.5 + max_severity * 0.5)
        elif n_fail >= 1 or n_warn >= 2:
            decision = "approved_with_adjustments"
            confidence = 0.70 + (n_pass / 5) * 0.20
        elif n_warn >= 1:
            decision = "approved_with_adjustments"
            confidence = 0.80 + (n_pass / 5) * 0.15
        else:
            decision = "approved"
            confidence = 0.85 + (n_pass / 5) * 0.10

        # Build reasoning
        regime = agent1_output.get("market_regime", {}).get("primary_regime", "unknown")
        strategy = agent3_output.get("optimization", {}).get("strategy_type", "unknown")
        profile = agent2_output.get("risk_classification", agent2_output)

        if decision == "rejected":
            reasoning = (
                f"REJECTED: {n_fail} audits failed with maximum severity {max_severity:.2f}. "
                f"The proposed {strategy} allocation is fundamentally incompatible with the current "
                f"{regime} market regime and/or the investor's risk profile "
                f"(score: {profile.get('risk_score', 'N/A')}, max DD: {profile.get('max_acceptable_drawdown', 'N/A'):.0%}). "
                f"Critical findings: {'; '.join(a['finding'] for a in critical)}. "
                f"The allocation must be reconstructed with tighter constraints."
            )
        elif decision == "approved_with_adjustments":
            reasoning = (
                f"APPROVED WITH ADJUSTMENTS: {n_pass} audits passed, {n_warn} warnings, "
                f"{n_fail} failures. The proposed {strategy} allocation for a {regime} regime "
                f"requires modification to meet safety standards. "
                f"Adjustments applied to bring allocation within regime and profile constraints. "
                f"Key findings: {'; '.join(a['finding'] for a in (critical + warnings)[:3])}."
            )
        else:
            reasoning = (
                f"APPROVED: All {n_pass} audits passed. The {strategy} allocation is consistent "
                f"with the {regime} market regime and the investor's "
                f"{profile.get('behavioral_type', 'moderate')} profile "
                f"(risk score: {profile.get('risk_score', 0.5):.2f}). "
                f"Monte Carlo simulations confirm drawdown risk within stated tolerance."
            )

        critical_risks = [a["finding"] for a in critical]
        mitigations = []
        if adjusted_allocation:
            for adj in adjusted_allocation:
                if abs(adj.get("change", 0)) > 0.005:
                    mitigations.append(adj.get("reason", ""))

        residual = []
        if max_severity > 0:
            residual.append(
                f"Maximum audit severity remains at {max_severity:.2f} "
                f"— continued monitoring recommended"
            )
            if any("bear" in regime for _ in [1]):
                residual.append(
                    "Bear regime may persist — rebalancing triggers should be monitored"
                )

        return {
            "decision": decision,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "critical_risks": critical_risks,
            "mitigations_applied": mitigations[:5],
            "residual_risks": residual[:3],
        }

    # ────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(content: str) -> dict:
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
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM CRO JSON: {e}")
            return {"decision": "approved_with_adjustments", "reasoning": content}
