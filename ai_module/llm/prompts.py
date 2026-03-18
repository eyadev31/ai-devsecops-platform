"""
Hybrid Intelligence Portfolio System — Prompt Engineering
===========================================================
Production-grade prompt templates for Agent 1 (Macro Intelligence).
Designed for structured, analytical JSON output with quantified
uncertainty and zero hallucination.

All prompts enforce:
  - Strict JSON output format
  - Quantified confidence scores
  - Explicit uncertainty flagging
  - No speculation beyond available data
"""

# ══════════════════════════════════════════════════════
#  SYSTEM PROMPT — Agent 1 Identity
# ══════════════════════════════════════════════════════
AGENT1_SYSTEM_PROMPT = """You are a Senior Global Macro Intelligence System.

You analyze multi-asset financial markets including equities, ETFs, forex, commodities, and crypto.

Your job is to:
1. Identify the current market regime.
2. Detect volatility state.
3. Evaluate macroeconomic environment.
4. Detect systemic risk signals.
5. Output a structured JSON context for downstream AI agents.

You must:
- Be analytical and structured.
- Avoid generic statements — every claim must be supported by the data provided.
- Quantify uncertainty when possible (use confidence scores 0.0-1.0).
- Never hallucinate unknown data. If data is missing, say "data_unavailable".
- If confidence < 60%, explicitly flag it with "low_confidence_warning": true.
- Distinguish between facts (from data) and interpretations (your analysis).
- Consider second-order effects and cross-asset implications.

Output strictly in JSON format."""


# ══════════════════════════════════════════════════════
#  MARKET REGIME INTERPRETATION PROMPT
# ══════════════════════════════════════════════════════
REGIME_INTERPRETATION_PROMPT = """## Market Regime Analysis Task

You are given the output of two quantitative regime detection models (HMM and Random Forest) 
along with raw feature data. Your task is to:

1. **Validate** the ML model outputs against the raw data
2. **Synthesize** a unified market regime assessment
3. **Explain** the key drivers of the current regime
4. **Identify** regime transition risks

### ML Model Outputs:
{regime_data}

### Key Market Features:
{feature_summary}

### VIX Data:
{vix_summary}

Respond in this exact JSON structure:
{{
    "regime_validation": {{
        "ml_output_consistent_with_data": true/false,
        "validation_notes": "string"
    }},
    "market_narrative": "A 2-3 sentence institutional-quality description of the current market regime, referencing specific data points.",
    "key_drivers": [
        "driver 1 with specific metric",
        "driver 2 with specific metric"
    ],
    "regime_transition_risks": [
        {{
            "scenario": "description",
            "probability": 0.0-1.0,
            "trigger": "what would cause this"
        }}
    ],
    "cross_asset_implications": {{
        "equities": "impact assessment",
        "bonds": "impact assessment",
        "crypto": "impact assessment",
        "commodities": "impact assessment",
        "forex": "impact assessment"
    }},
    "confidence_level": 0.0-1.0,
    "uncertainty_factors": ["factor 1", "factor 2"]
}}"""


# ══════════════════════════════════════════════════════
#  RISK ASSESSMENT PROMPT
# ══════════════════════════════════════════════════════
RISK_ASSESSMENT_PROMPT = """## Systemic Risk Assessment Task

Analyze the following risk signals and macro environment to produce
a comprehensive risk assessment for portfolio management.

### Risk Detector Output:
{risk_data}

### Macro Environment:
{macro_data}

### Volatility State:
{vol_data}

Respond in this exact JSON structure:
{{
    "risk_narrative": "2-3 sentence summary of the overall risk environment. Be specific, reference data points.",
    "key_risks": [
        {{
            "risk": "description",
            "severity": "low|moderate|elevated|high|critical",
            "probability": 0.0-1.0,
            "time_horizon": "immediate|short_term|medium_term"
        }}
    ],
    "opportunities": [
        {{
            "opportunity": "description",
            "rationale": "why this exists given current conditions"
        }}
    ],
    "recommended_actions": [
        "action 1",
        "action 2"
    ],
    "hedging_considerations": "brief suggestions for risk mitigation",
    "confidence_level": 0.0-1.0,
    "data_limitations": ["limitation 1"]
}}"""


# ══════════════════════════════════════════════════════
#  CONTEXT SYNTHESIS PROMPT
# ══════════════════════════════════════════════════════
CONTEXT_SYNTHESIS_PROMPT = """## Final Context Synthesis Task

You are producing the final structured intelligence output that will be consumed
by downstream AI agents (Dynamic Question Generator, Behavioral Profiler, Portfolio Optimizer).

This output must be COMPLETE, ACTIONABLE, and QUANTIFIED.

### Regime Analysis:
{regime_analysis}

### Risk Assessment:
{risk_assessment}

### Macro Environment:
{macro_environment}

### Volatility State:
{volatility_state}

### Cross-Asset Correlations:
{correlations}

Produce the FINAL intelligence context in this JSON structure:
{{
    "market_narrative": "3-4 sentence institutional-quality market assessment. Reference specific metrics.",
    "key_risks": ["risk 1 with quantification", "risk 2"],
    "opportunities": ["opportunity 1", "opportunity 2"],
    "sector_implications": {{
        "overweight": ["sectors to overweight and why"],
        "underweight": ["sectors to underweight and why"],
        "neutral": ["sectors to maintain"]
    }},
    "asset_class_outlook": {{
        "equities": {{"outlook": "bullish|neutral|bearish", "conviction": 0.0-1.0}},
        "bonds": {{"outlook": "string", "conviction": 0.0-1.0}},
        "crypto": {{"outlook": "string", "conviction": 0.0-1.0}},
        "commodities": {{"outlook": "string", "conviction": 0.0-1.0}},
        "cash": {{"outlook": "string", "conviction": 0.0-1.0}}
    }},
    "risk_budget_suggestion": {{
        "max_portfolio_volatility": "percentage",
        "max_single_asset_weight": "percentage",
        "recommended_cash_buffer": "percentage"
    }},
    "confidence_level": 0.0-1.0,
    "uncertainty_factors": ["factor 1", "factor 2"],
    "data_quality_notes": "any caveats about data freshness or completeness"
}}"""


# ══════════════════════════════════════════════════════
#  HELPER: FORMAT DATA FOR PROMPTS
# ══════════════════════════════════════════════════════
def format_regime_data(regime_result: dict) -> str:
    """Format regime detection results for LLM prompt injection."""
    import json
    # Extract key fields only (avoid dumping raw model internals)
    formatted = {
        "primary_regime": regime_result.get("primary_regime"),
        "confidence": regime_result.get("confidence"),
        "models_agree": regime_result.get("models_agree"),
        "ensemble_method": regime_result.get("ensemble_method"),
        "regime_duration_days": regime_result.get("regime_duration_days"),
        "transition_probability": regime_result.get("transition_probability"),
        "hmm_regime": regime_result.get("hmm_result", {}).get("current_regime"),
        "hmm_confidence": regime_result.get("hmm_result", {}).get("confidence"),
        "rf_regime": regime_result.get("rf_result", {}).get("current_regime"),
        "rf_confidence": regime_result.get("rf_result", {}).get("confidence"),
        "rf_top_features": regime_result.get("rf_result", {}).get("feature_importances", {}),
    }
    return json.dumps(formatted, indent=2)


def format_feature_summary(features: dict) -> str:
    """Format feature data summary for LLM prompt."""
    import json
    summary = {}

    # Volatility summary
    vol = features.get("volatility", {})
    realized = vol.get("realized")
    if realized is not None and hasattr(realized, 'iloc') and not realized.empty:
        summary["realized_volatility"] = {
            col: round(float(realized[col].iloc[-1]), 4) for col in realized.columns if not realized[col].empty
        }

    vol_pct = vol.get("vol_percentile")
    if vol_pct is not None and hasattr(vol_pct, 'iloc') and not vol_pct.empty:
        summary["vol_percentile"] = round(float(vol_pct.iloc[-1]), 1)

    # Momentum summary
    mom = features.get("momentum", {})
    rsi = mom.get("rsi")
    if rsi is not None and hasattr(rsi, 'iloc') and not rsi.empty:
        summary["rsi_14d"] = round(float(rsi.iloc[-1]), 2)

    macd = mom.get("macd")
    if macd is not None and hasattr(macd, 'iloc') and not macd.empty:
        summary["macd_histogram"] = round(float(macd["histogram"].iloc[-1]), 4)

    # Drawdown
    dd = features.get("drawdown", {})
    current_dd = dd.get("current")
    if current_dd is not None and hasattr(current_dd, 'iloc') and not current_dd.empty:
        summary["current_drawdown_pct"] = round(float(current_dd["drawdown"].iloc[-1]) * 100, 2)

    # Correlations
    corr = features.get("correlations", {})
    summary["median_cross_asset_correlation"] = round(corr.get("median_correlation", 0), 3)

    return json.dumps(summary, indent=2)
