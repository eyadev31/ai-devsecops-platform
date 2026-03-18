"""
Hybrid Intelligence Portfolio System — News Analyst (LLM)
==========================================================
Agent 5: LLM-powered market signal synthesis and narrative generation.

Uses the existing Gemini/Groq LLM infrastructure to:
  1. Synthesize processed articles into a market signal
  2. Generate institutional-quality news intelligence briefings
  3. Assess risk events and their portfolio implications
  4. Produce per-asset-class allocation bias recommendations
  5. Identify contrarian signals that contradict consensus
"""

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


class NewsAnalyst:
    """
    LLM-powered news market signal synthesizer.
    
    Takes processed, sentiment-scored, impact-assessed articles
    and produces final structured market signals with detailed
    institutional-quality analysis narrative.
    """

    def __init__(self):
        self._llm_client = None
        self._llm_available = None
        self._llm_calls = 0
        self._llm_latency_ms = 0.0

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    @property
    def llm_latency_ms(self) -> float:
        return self._llm_latency_ms

    def analyze(
        self,
        articles: list[dict],
        temporal_sentiment: dict,
        event_detection: dict,
        agent1_context: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """
        Generate final market signal and LLM analysis.
        
        Args:
            articles: Top-impact processed articles
            temporal_sentiment: Temporal aggregation results
            event_detection: Event detection results
            agent1_context: Optional Agent 1 market context
            
        Returns:
            (market_signal_dict, llm_analysis_dict)
        """
        # Build the quantitative signal first (always available)
        quant_signal = self._build_quantitative_signal(articles, temporal_sentiment, event_detection)

        # Try LLM enhancement
        llm_analysis = self._generate_llm_analysis(
            articles, temporal_sentiment, event_detection, agent1_context
        )

        # Merge LLM insights into market signal if available
        if llm_analysis.get("confidence_level", 0) > 0:
            quant_signal["narrative"] = llm_analysis.get("market_narrative", "")
            if llm_analysis.get("key_themes"):
                quant_signal["primary_driver"] = llm_analysis["key_themes"][0]

        return quant_signal, llm_analysis

    def _build_quantitative_signal(
        self,
        articles: list[dict],
        temporal: dict,
        events: dict,
    ) -> dict:
        """
        Build market signal from quantitative data only (no LLM needed).
        This ensures the agent always produces output even without LLM.
        """
        # Overall sentiment direction
        sentiment_1h = temporal.get("overall_1h", 0.0)
        sentiment_24h = temporal.get("overall_24h", 0.0)
        sentiment_7d = temporal.get("overall_7d", 0.0)

        # Weighted composite
        composite = (sentiment_1h * 0.4) + (sentiment_24h * 0.35) + (sentiment_7d * 0.25)

        # Determine signal type
        if composite > 0.15:
            signal_type = "bullish"
        elif composite < -0.15:
            signal_type = "bearish"
        else:
            signal_type = "neutral"

        # Override for critical events
        has_critical = events.get("has_critical_event", False)
        if has_critical:
            signal_type = "bearish"
            composite = min(composite, -0.3)

        # Signal strength
        abs_composite = abs(composite)
        if abs_composite >= 0.6:
            strength = "extreme"
        elif abs_composite >= 0.4:
            strength = "strong"
        elif abs_composite >= 0.2:
            strength = "moderate"
        else:
            strength = "weak"

        # ── GUARD: Never allow extreme/strong without high-impact news ──
        high_impact_count = sum(
            1 for a in articles
            if a.get("impact", {}).get("impact_score", 0) > 0.70
        )
        if high_impact_count == 0 and strength in ("extreme", "strong"):
            strength = "moderate"
            logger.info(
                f"Signal strength capped to 'moderate' — no high-impact articles "
                f"(0 articles with impact_score > 0.70)"
            )

        # Confidence based on article volume and agreement
        n_articles = len(articles)
        if n_articles >= 10:
            volume_conf = 0.8
        elif n_articles >= 5:
            volume_conf = 0.6
        else:
            volume_conf = 0.4

        # Agreement factor: what fraction of articles agree with the direction
        agreeing = 0
        for a in articles:
            s = a.get("sentiment", {}).get("score", 0)
            if (composite > 0 and s > 0) or (composite < 0 and s < 0) or (composite == 0):
                agreeing += 1
        agreement_ratio = agreeing / max(n_articles, 1)
        
        confidence = min((volume_conf * 0.5) + (agreement_ratio * 0.5), 1.0)

        # Affected asset classes
        asset_class_counts = {}
        for a in articles:
            for ac in a.get("asset_classes", []):
                asset_class_counts[ac] = asset_class_counts.get(ac, 0) + 1

        affected_assets = sorted(asset_class_counts, key=asset_class_counts.get, reverse=True)

        # Per-asset-class bias
        recommended_bias = {}
        for ac in ["crypto", "equities", "bonds", "commodities", "forex"]:
            ac_sentiment = temporal.get(f"{ac}_sentiment", {})
            if isinstance(ac_sentiment, dict) and ac_sentiment:
                # Use short-term sentiment as bias
                short = ac_sentiment.get("1h", ac_sentiment.get("6h", 0.0))
                bias = short * 0.15  # Scale down to reasonable allocation adjustment
                recommended_bias[ac] = round(bias, 4)
            else:
                recommended_bias[ac] = 0.0

        # Primary driver: highest impact article topic
        primary_driver = ""
        if articles:
            top_article = max(articles, key=lambda a: a.get("impact", {}).get("impact_score", 0))
            primary_driver = top_article.get("title", "")[:100]

        # Risk events
        risk_events = [
            e.get("description", "") for e in events.get("events_detected", [])
        ]

        return {
            "signal_type": signal_type,
            "confidence": round(confidence, 4),
            "affected_assets": affected_assets[:5],
            "signal_strength": strength,
            "primary_driver": primary_driver,
            "recommended_bias": recommended_bias,
            "narrative": "",  # Filled by LLM if available
            "risk_events": risk_events[:5],
        }

    def _generate_llm_analysis(
        self,
        articles: list[dict],
        temporal: dict,
        events: dict,
        agent1_context: Optional[dict],
    ) -> dict:
        """Generate LLM-powered qualitative analysis."""
        # Default result (used when LLM unavailable)
        default_analysis = {
            "market_narrative": self._build_fallback_narrative(articles, temporal, events),
            "key_themes": self._extract_key_themes(articles),
            "risk_assessment": events.get("risk_alert", "No significant risks detected."),
            "allocation_implications": "",
            "confidence_level": 0.4,
            "contrarian_signals": [],
        }

        # Try LLM
        client = self._get_llm_client()
        if client is None:
            return default_analysis

        try:
            prompt = self._build_analysis_prompt(articles, temporal, events, agent1_context)
            start = time.time()
            response = client.generate(prompt)
            self._llm_calls += 1
            self._llm_latency_ms += (time.time() - start) * 1000

            if response:
                parsed = self._parse_llm_response(response.get("content", ""))
                if parsed:
                    return parsed

        except Exception as e:
            logger.warning(f"LLM news analysis failed: {e}")

        return default_analysis

    def _build_analysis_prompt(
        self,
        articles: list[dict],
        temporal: dict,
        events: dict,
        agent1_context: Optional[dict],
    ) -> str:
        """Build the analysis prompt for the LLM."""
        # Summarize top articles
        article_summaries = []
        for a in articles[:10]:
            sentiment = a.get("sentiment", {})
            impact = a.get("impact", {})
            article_summaries.append(
                f"- [{a.get('source', 'unknown')}] {a.get('title', '')} | "
                f"Sentiment: {sentiment.get('label', 'neutral')} ({sentiment.get('score', 0):.2f}) | "
                f"Impact: {impact.get('impact_score', 0):.2f}"
            )

        articles_text = "\n".join(article_summaries) if article_summaries else "No articles available."

        # Market context
        context_text = ""
        if agent1_context:
            regime = agent1_context.get("market_regime", {}).get("primary_regime", "unknown")
            risk = agent1_context.get("systemic_risk", {}).get("risk_category", "unknown")
            context_text = f"\nCurrent Market Regime: {regime}\nSystemic Risk: {risk}\n"

        # Events
        event_list = events.get("events_detected", [])
        events_text = ""
        if event_list:
            events_text = "\nDetected Events:\n" + "\n".join(
                f"- [{e.get('severity', 'unknown')}] {e.get('description', '')}"
                for e in event_list[:5]
            )

        prompt = f"""You are a senior hedge fund market analyst producing an institutional-quality news intelligence briefing.

PROCESSED NEWS DATA:
{articles_text}
{context_text}
TEMPORAL SENTIMENT:
  1h: {temporal.get('overall_1h', 0):.2f}
  24h: {temporal.get('overall_24h', 0):.2f}
  7d: {temporal.get('overall_7d', 0):.2f}
  Momentum: {temporal.get('sentiment_momentum', 'stable')}
  Regime: {temporal.get('regime_sentiment', 'neutral')}
{events_text}

Produce a JSON response with EXACTLY this structure:
{{
  "market_narrative": "<2-3 paragraph institutional-quality briefing>",
  "key_themes": ["<theme1>", "<theme2>", "<theme3>"],
  "risk_assessment": "<1-2 sentences on risk environment>",
  "allocation_implications": "<1-2 sentences on how to adjust portfolio>",
  "confidence_level": <float 0.0-1.0>,
  "contrarian_signals": ["<signal1>", "<signal2>"]
}}

Rules:
- Be specific and actionable. Reference actual data points.
- Write like a Goldman Sachs morning brief, not a blog post.
- Confidence should reflect data quality and consistency.
- Contrarian signals are instances where consensus may be wrong.

JSON Response:"""

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[dict]:
        """Parse LLM response into structured analysis dict."""
        try:
            # First, strip markdown code blocks if present (```json ... ```)
            clean_resp = response.strip()
            if clean_resp.startswith("```"):
                # Remove first line (e.g., ```json)
                clean_resp = clean_resp.split("\n", 1)[-1]
                # Remove trailing ```
                if clean_resp.endswith("```"):
                    clean_resp = clean_resp[:-3]
                    
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', clean_resp)
            if json_match:
                clean_json = re.sub(r'[\x00-\x1F]+', ' ', json_match.group())
                data = json.loads(clean_json)

                return {
                    "market_narrative": str(data.get("market_narrative", "")),
                    "key_themes": list(data.get("key_themes", [])),
                    "risk_assessment": str(data.get("risk_assessment", "")),
                    "allocation_implications": str(data.get("allocation_implications", "")),
                    "confidence_level": float(data.get("confidence_level", 0.5)),
                    "contrarian_signals": list(data.get("contrarian_signals", [])),
                }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM analysis response: {e}")

        return None

    def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_available is False:
            return None

        if self._llm_client is not None:
            return self._llm_client

        try:
            from llm.gemini_client import LLMFactory
            self._llm_client = LLMFactory.create()
            self._llm_available = True
            return self._llm_client
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            self._llm_available = False
            return None

    # ═════════════════════════════════════════════════════
    #  FALLBACK HELPERS (LLM unavailable)
    # ═════════════════════════════════════════════════════

    @staticmethod
    def _build_fallback_narrative(
        articles: list[dict],
        temporal: dict,
        events: dict,
    ) -> str:
        """Build a basic narrative when LLM is unavailable."""
        regime = temporal.get("regime_sentiment", "neutral")
        momentum = temporal.get("sentiment_momentum", "stable")
        n_articles = len(articles)
        n_events = events.get("event_count", 0)

        narrative = (
            f"News sentiment analysis based on {n_articles} financial articles indicates "
            f"a {regime} market sentiment regime with {momentum} momentum. "
        )

        if n_events > 0:
            severity = events.get("highest_severity", "routine")
            narrative += (
                f"{n_events} market event(s) detected at {severity} severity level. "
                f"Recommended action: {events.get('recommended_action', 'monitor')}. "
            )

        # Top article headlines
        if articles:
            top_3 = articles[:3]
            headlines = ", ".join(f'"{a.get("title", "")[:60]}"' for a in top_3)
            narrative += f"Key headlines: {headlines}."

        return narrative

    @staticmethod
    def _extract_key_themes(articles: list[dict]) -> list[str]:
        """Extract key themes from article topics."""
        topic_counts = {}
        for a in articles:
            topic = a.get("topic", "general_market")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Top 5 themes
        sorted_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)
        return sorted_topics[:5]
