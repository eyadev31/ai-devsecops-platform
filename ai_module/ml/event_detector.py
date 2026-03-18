"""
Hybrid Intelligence Portfolio System — Event Detector
=======================================================
Agent 5: Black swan and macro event detection AI.

Detects market-moving events from processed news articles:
  - Fed meetings / rate decisions
  - ETF approvals / launches
  - Regulatory actions
  - Black swan events (crashes, defaults, bank runs)
  - Geopolitical crises (wars, sanctions, embargoes)
  - Macro shocks (recession, stagflation)

Severity classification: routine → significant → major → critical → black_swan

Each event generates:
  - Type and description
  - Severity level
  - Expected market impact direction
  - Affected asset classes
  - Recommended portfolio action
"""

import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  EVENT IMPACT PROFILES
# ═══════════════════════════════════════════════════════

EVENT_IMPACT_PROFILES = {
    "fed_meeting": {
        "description": "Federal Reserve / FOMC policy event",
        "typical_severity": "significant",
        "impact_direction": {
            "bonds": "high",
            "equities": "high",
            "crypto": "moderate",
            "forex": "high",
            "commodities": "moderate",
        },
        "recommended_action": "review",
    },
    "etf_approval": {
        "description": "ETF approval or launch",
        "typical_severity": "significant",
        "impact_direction": {
            "crypto": "high",
            "equities": "moderate",
        },
        "recommended_action": "review",
    },
    "regulation": {
        "description": "Regulatory or enforcement action",
        "typical_severity": "significant",
        "impact_direction": {
            "crypto": "high",
            "equities": "moderate",
        },
        "recommended_action": "review",
    },
    "black_swan": {
        "description": "Potential black swan / market crisis event",
        "typical_severity": "critical",
        "impact_direction": {
            "equities": "extreme",
            "crypto": "extreme",
            "bonds": "high",
            "commodities": "high",
            "forex": "high",
        },
        "recommended_action": "derisk",
    },
    "geopolitical": {
        "description": "Geopolitical crisis or conflict",
        "typical_severity": "major",
        "impact_direction": {
            "equities": "high",
            "commodities": "high",
            "forex": "high",
            "bonds": "moderate",
            "crypto": "moderate",
        },
        "recommended_action": "hedge",
    },
    "macro_shock": {
        "description": "Macroeconomic disruption",
        "typical_severity": "major",
        "impact_direction": {
            "equities": "high",
            "bonds": "high",
            "commodities": "moderate",
            "crypto": "moderate",
            "forex": "moderate",
        },
        "recommended_action": "derisk",
    },
}

# Severity ranking (higher = more severe)
SEVERITY_RANK = {
    "none": 0,
    "routine": 1,
    "significant": 2,
    "major": 3,
    "critical": 4,
    "black_swan": 5,
}


class EventDetector:
    """
    Market event detection engine.
    
    Scans news articles for patterns indicating significant
    market-moving events and classifies their potential impact.
    """

    def __init__(self):
        from config.settings import NewsConfig
        self._config = NewsConfig
        self._detection_stats = {
            "articles_scanned": 0,
            "events_detected": 0,
            "by_type": {},
        }

    @property
    def stats(self) -> dict:
        return self._detection_stats.copy()

    def detect_events(self, articles: list[dict]) -> dict:
        """
        Scan articles for market-moving events.
        
        Args:
            articles: Processed articles with sentiment and entities
            
        Returns:
            Event detection result dict (compatible with EventDetection schema)
        """
        self._detection_stats = {
            "articles_scanned": len(articles),
            "events_detected": 0,
            "by_type": {},
        }

        # Step 1: Assign articles to event types based on keyword matches
        event_candidates = {}
        
        for article in articles:
            text = article.get("combined_text", article.get("title", "")).lower()
            title = article.get("title", "").lower()
            
            for event_type, keywords in self._config.EVENT_KEYWORDS.items():
                matches = []
                for keyword in keywords:
                    if keyword in title:
                        matches.append({"keyword": keyword, "in_title": True})
                    elif keyword in text:
                        matches.append({"keyword": keyword, "in_title": False})
                if matches:
                    if event_type not in event_candidates:
                        event_candidates[event_type] = []
                    event_candidates[event_type].append({
                        "article": article,
                        "matches": matches
                    })

        # Step 2: Evaluate severity for each event type based on aggregate data
        unique_events = []
        for event_type, candidates in event_candidates.items():
            event = self._evaluate_event_cluster(event_type, candidates)
            if event:
                unique_events.append(event)
                self._detection_stats["by_type"][event_type] = len(candidates)

        self._detection_stats["events_detected"] = len(unique_events)

        # Determine overall severity
        highest_severity = "none"
        for event in unique_events:
            if SEVERITY_RANK.get(event["severity"], 0) > SEVERITY_RANK.get(highest_severity, 0):
                highest_severity = event["severity"]

        # Determine recommended action
        has_critical = highest_severity in ("critical", "black_swan")
        recommended_action = "monitor"
        if has_critical:
            recommended_action = "derisk"
        elif highest_severity == "major":
            recommended_action = "hedge"
        elif highest_severity == "significant":
            recommended_action = "review"

        # Build risk alert message
        risk_alert = self._build_risk_alert(unique_events, highest_severity)

        result = {
            "events_detected": unique_events,
            "has_critical_event": has_critical,
            "highest_severity": highest_severity,
            "event_count": len(unique_events),
            "risk_alert": risk_alert,
            "recommended_action": recommended_action,
        }

        if unique_events:
            logger.info(
                f"Event Detection: {len(unique_events)} events detected | "
                f"Highest severity: {highest_severity} | "
                f"Action: {recommended_action}"
            )
        else:
            logger.info("Event Detection: No significant events detected.")

        return result

    def _evaluate_event_cluster(self, event_type: str, candidates: list[dict]) -> dict:
        """Evaluate an event type using aggregated article data to enforce strict validation."""
        from config.settings import NewsConfig

        profile = EVENT_IMPACT_PROFILES.get(event_type, {})
        base_severity = profile.get("typical_severity", "routine")
        base_rank = SEVERITY_RANK.get(base_severity, 1)

        sources = set()
        asset_classes = set()
        max_impact = 0.0
        max_sentiment = 0.0
        total_matches = 0
        title_matches = 0
        
        best_article = None
        best_sentiment_label = "neutral"
        all_keywords = set()

        for c in candidates:
            article = c["article"]
            matches = c["matches"]
            
            source = article.get("source", "unknown")
            sources.add(source)
            
            for ac in article.get("asset_classes", []):
                asset_classes.add(ac)
                
            impact = article.get("impact", {}).get("impact_score", 0.0)
            if impact > max_impact:
                max_impact = impact
                best_article = article
                
            sentiment_score = abs(article.get("sentiment", {}).get("score", 0.0))
            if sentiment_score > max_sentiment:
                max_sentiment = sentiment_score
                best_sentiment_label = article.get("sentiment", {}).get("label", "neutral")
                
            total_matches += len(matches)
            title_matches += sum(1 for m in matches if m.get("in_title", False))
            
            for m in matches:
                all_keywords.add(m["keyword"])

        if best_article is None:
            best_article = candidates[0]["article"]

        # Calculate initial severity score similarly to before
        match_bonus = min(total_matches * 0.1 + title_matches * 0.3, 2.0)
        
        sentiment_bonus = 0
        if max_sentiment > 0.7:
            sentiment_bonus = 1.0
        elif max_sentiment > 0.5:
            sentiment_bonus = 0.5

        # Source credibility bonus
        source_bonus = min(sum(NewsConfig.SOURCE_WEIGHTS.get(s.lower(), 0.4) for s in sources) * 0.2, 0.5)

        severity_score = base_rank + match_bonus + sentiment_bonus + source_bonus

        # Strict validation for black_swan and critical events
        is_multiple_sources = len(sources) > 1
        is_cross_asset = len(asset_classes) > 1
        is_high_impact = max_impact > 0.70  # User threshold requirement

        # Upgrade/Downgrade logic based on strict constraints
        if severity_score >= 4.5:
            if not (is_high_impact and is_multiple_sources and is_cross_asset):
                severity_score = 3.4  # Downgrade to major if it lacks validation
                
        if severity_score >= 3.5:
            if not (is_multiple_sources and is_high_impact):
                severity_score = 2.4  # Downgrade to significant if it lacks validation
                
        # Final mapping
        if severity_score >= 4.5:
            severity_level = "black_swan"
        elif severity_score >= 3.5:
            severity_level = "critical"
        elif severity_score >= 2.5:
            severity_level = "major"
        elif severity_score >= 1.5:
            severity_level = "significant"
        else:
            severity_level = "routine"

        return {
            "type": event_type,
            "severity": severity_level,
            "description": profile.get("description", event_type),
            "trigger_article": best_article.get("title", ""),
            "trigger_source": list(sources)[0] if sources else "unknown",
            "keywords_matched": list(all_keywords),
            "sentiment_context": best_sentiment_label,
            "affected_assets": list(asset_classes) if asset_classes else list(profile.get("impact_direction", {}).keys()),
            "impact_direction": profile.get("impact_direction", {}),
            "recommended_action": profile.get("recommended_action", "monitor"),
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _build_risk_alert(events: list[dict], highest_severity: str) -> str:
        """Build a human-readable risk alert message."""
        if not events:
            return "No significant market events detected."

        # Sort events by severity (descending)
        sorted_events = sorted(
            events,
            key=lambda e: SEVERITY_RANK.get(e["severity"], 0),
            reverse=True,
        )

        alert_lines = [f"⚠️ {len(events)} market event(s) detected (highest: {highest_severity}):"]
        for event in sorted_events[:5]:  # Max 5 in alert
            severity_emoji = {
                "routine": "🟢",
                "significant": "🟡",
                "major": "🟠",
                "critical": "🔴",
                "black_swan": "⛔",
            }.get(event["severity"], "⚪")
            alert_lines.append(
                f"  {severity_emoji} [{event['severity'].upper()}] {event['description']}: "
                f"{event['trigger_article'][:80]}..."
            )

        return "\n".join(alert_lines)
