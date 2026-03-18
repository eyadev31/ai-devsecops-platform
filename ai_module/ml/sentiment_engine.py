"""
Hybrid Intelligence Portfolio System — Sentiment Engine
========================================================
Agent 5: Hybrid FinBERT + LLM sentiment analysis for financial news.

Architecture:
  1. FinBERT (primary) — Financial-domain-specific transformer
  2. LLM (secondary) — Contextual sentiment via Gemini/Groq
  3. Confidence Arbitration — When models disagree, weighted consensus

Fallback chain:
  FinBERT+LLM → FinBERT-only → LLM-only → Rule-based fallback
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  SENTIMENT LABEL MAPPING
# ═══════════════════════════════════════════════════════

SENTIMENT_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}

# Rule-based fallback keywords
BULLISH_KEYWORDS = [
    "surge", "rally", "gain", "rise", "jump", "soar", "boost",
    "record high", "all-time high", "bullish", "optimistic",
    "upgrade", "outperform", "beat expectations", "strong",
    "growth", "recovery", "stimulus", "dovish", "rate cut",
    "approval", "inflow", "adoption", "breakthrough",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "fall", "decline", "drop", "sink",
    "collapse", "selloff", "sell-off", "bearish", "pessimistic",
    "downgrade", "underperform", "miss expectations", "weak",
    "recession", "slowdown", "crisis", "default", "hawkish",
    "rate hike", "ban", "crackdown", "outflow", "warning",
    "risk", "fear", "uncertainty", "sanctions", "tariff",
]


class SentimentEngine:
    """
    Hybrid financial sentiment analysis engine.
    
    Combines FinBERT (transformer-based) with LLM (generative)
    analysis for robust, high-confidence sentiment signals.
    """

    def __init__(self):
        self._finbert_pipeline = None  # Lazy loaded
        self._finbert_available = None
        self._llm_client = None       # Lazy loaded
        self._llm_available = None
        self._analysis_stats = {
            "total_analyzed": 0,
            "finbert_used": 0,
            "llm_used": 0,
            "fallback_used": 0,
            "model_agreements": 0,
            "model_disagreements": 0,
        }

    @property
    def stats(self) -> dict:
        return self._analysis_stats.copy()

    def analyze_batch(self, articles: list[dict]) -> list[dict]:
        """
        Analyze sentiment for a batch of processed articles.
        
        Args:
            articles: List of processed article dicts (from NewsProcessor)
            
        Returns:
            Articles enriched with sentiment analysis results
        """
        self._analysis_stats = {
            "total_analyzed": 0,
            "finbert_used": 0,
            "llm_used": 0,
            "fallback_used": 0,
            "model_agreements": 0,
            "model_disagreements": 0,
        }

        results = []
        for article in articles:
            text = article.get("combined_text", article.get("title", ""))
            sentiment = self._analyze_single(text)
            enriched = article.copy()
            enriched["sentiment"] = sentiment
            results.append(enriched)
            self._analysis_stats["total_analyzed"] += 1

        logger.info(
            f"Sentiment Analysis Complete: {self._analysis_stats['total_analyzed']} articles, "
            f"FinBERT: {self._analysis_stats['finbert_used']}, "
            f"LLM: {self._analysis_stats['llm_used']}, "
            f"Fallback: {self._analysis_stats['fallback_used']}, "
            f"Agreement: {self._analysis_stats['model_agreements']}/{self._analysis_stats['total_analyzed']}"
        )

        return results

    def _analyze_single(self, text: str) -> dict:
        """
        Analyze sentiment for a single text.
        
        Strategy:
          1. Try FinBERT first (fast, financial-domain)
          2. If text is complex or FinBERT confidence is low, also use LLM
          3. Arbitrate between models for final result
        """
        if not text or len(text.strip()) < 10:
            return self._neutral_result()

        # Truncate for model input
        text_truncated = text[:512]

        # --- FinBERT Analysis ---
        finbert_result = self._analyze_finbert(text_truncated)

        # --- LLM Analysis Setup ---
        llm_result = None
        
        # Check if text is likely geopolitical (simple heuristic since we don't have topic here)
        text_lower = text_truncated.lower()
        is_geopolitical = any(kw in text_lower for kw in [
            "war", "geopolitical", "sanction", "missile", "military", "conflict", "nato", "embargo"
        ])
        
        # We need LLM if:
        # 1. FinBERT failed
        # 2. FinBERT confidence is low (< 0.6)
        # 3. FinBERT score is very extreme (> 0.8 or < -0.8) - requiring secondary validation
        # 4. Topic is geopolitical (FinBERT is bad at geopolitics)
        needs_llm = (
            finbert_result is None or
            finbert_result.get("confidence", 0) < 0.6 or
            abs(finbert_result.get("score", 0.0)) > 0.8 or
            is_geopolitical
        )
        
        if needs_llm:
            llm_result = self._analyze_llm(text_truncated)

        # --- Arbitration ---
        return self._arbitrate(finbert_result, llm_result, text_truncated)

    def _analyze_finbert(self, text: str) -> Optional[dict]:
        """Run FinBERT sentiment analysis."""
        if self._finbert_available is False:
            return None

        try:
            if self._finbert_pipeline is None:
                from transformers import pipeline as hf_pipeline
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                model_name = "ProsusAI/finbert"
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                self._finbert_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    top_k=None,  # Get all labels with scores
                )
                self._finbert_available = True
                logger.info("FinBERT model loaded successfully.")

            result = self._finbert_pipeline(text[:512])
            self._analysis_stats["finbert_used"] += 1

            if result and len(result) > 0:
                # result is a list of lists: [[{label, score}, ...]]
                scores = result[0] if isinstance(result[0], list) else result
                
                # Find best label
                best = max(scores, key=lambda x: x["score"])
                label = best["label"].lower()
                confidence = best["score"]

                # Convert to numeric score
                numeric = SENTIMENT_MAP.get(label, 0.0) * confidence

                return {
                    "label": label,
                    "score": round(numeric, 4),
                    "confidence": 0.70,  # User priority: finbert = 0.7
                    "model_source": "finbert",
                }

        except ImportError:
            logger.info("transformers not installed. FinBERT unavailable.")
            self._finbert_available = False
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            self._finbert_available = False

        return None

    def _analyze_llm(self, text: str) -> Optional[dict]:
        """Run LLM-based sentiment analysis using existing Gemini/Groq client."""
        if self._llm_available is False:
            return None

        try:
            if self._llm_client is None:
                try:
                    from llm.gemini_client import LLMFactory
                    self._llm_client = LLMFactory.create()
                    self._llm_available = True
                except Exception as e:
                    logger.debug(f"Failed to initialize LLM for sentiment engine: {e}")
                    self._llm_available = False
                    return None

            prompt = (
                "You are a senior financial analyst. Analyze the sentiment of the following "
                "financial news text. Respond with ONLY a JSON object:\n"
                '{"label": "positive|neutral|negative", "score": <float -1.0 to 1.0>, '
                '"confidence": <float 0.0 to 1.0>}\n\n'
                f"Text: {text[:400]}\n\n"
                "JSON Response:"
            )

            response = self._llm_client.generate(prompt)
            self._analysis_stats["llm_used"] += 1

            if response:
                import json
                
                # Ensure we are extracting from the content string
                content_text = response.get("content", "") if isinstance(response, dict) else str(response)
                
                # Try to extract JSON from response text
                json_match = re.search(r'\{[\s\S]*\}', content_text)
                if json_match:
                    clean_json = re.sub(r'[\x00-\x1F]+', ' ', json_match.group())
                    data = json.loads(clean_json)
                    label = data.get("label", "neutral").lower()
                    score = float(data.get("score", 0.0))
                    confidence = float(data.get("confidence", 0.5))

                    # Clamp values
                    score = max(-1.0, min(1.0, score))
                    confidence = max(0.0, min(1.0, confidence))

                    return {
                        "label": label,
                        "score": round(score, 4),
                        "confidence": 0.85,  # User priority: llm = 0.85
                        "model_source": "llm",
                    }

        except Exception as e:
            logger.debug(f"LLM sentiment analysis failed: {e}")
            self._llm_available = False

        return None

    def _arbitrate(
        self,
        finbert_result: Optional[dict],
        llm_result: Optional[dict],
        text: str,
    ) -> dict:
        """
        Arbitrate between FinBERT and LLM results.
        
        Priority: Hybrid > FinBERT-only > LLM-only > Rule-based fallback
        """
        # Case 1: Both models available — hybrid consensus
        if finbert_result and llm_result:
            finbert_score = finbert_result["score"]
            llm_score = llm_result["score"]
            finbert_conf = finbert_result["confidence"]
            llm_conf = llm_result["confidence"]

            # Check agreement (same direction)
            agree = (finbert_score >= 0 and llm_score >= 0) or (finbert_score <= 0 and llm_score <= 0)

            if agree:
                self._analysis_stats["model_agreements"] += 1
                # Weighted average (FinBERT gets 60% weight — domain-specific)
                hybrid_score = (finbert_score * 0.6) + (llm_score * 0.4)
                hybrid_conf = 0.95  # User priority: hybrid = 0.95
            else:
                self._analysis_stats["model_disagreements"] += 1
                # Disagreement: use LLM as priority tie-breaker
                hybrid_score = llm_score
                hybrid_conf = 0.85  # Fallback to LLM confidence

            label = "positive" if hybrid_score > 0.05 else ("negative" if hybrid_score < -0.05 else "neutral")

            return {
                "label": label,
                "score": round(hybrid_score, 4),
                "confidence": round(hybrid_conf, 4),
                "model_source": "hybrid",
                "finbert_score": finbert_result["score"],
                "llm_score": llm_result["score"],
                "agreement": agree,
            }

        # Case 2: FinBERT only
        if finbert_result:
            finbert_result["agreement"] = True
            finbert_result["finbert_score"] = finbert_result["score"]
            finbert_result["llm_score"] = None
            return finbert_result

        # Case 3: LLM only
        if llm_result:
            llm_result["agreement"] = True
            llm_result["finbert_score"] = None
            llm_result["llm_score"] = llm_result["score"]
            return llm_result

        # Case 4: Rule-based fallback
        self._analysis_stats["fallback_used"] += 1
        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> dict:
        """
        Fallback: Rule-based sentiment using keyword matching.
        Used when no ML models are available.
        """
        text_lower = text.lower()

        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
        total = max(bullish_count + bearish_count, 1)

        if bullish_count > bearish_count:
            score = min(bullish_count / total, 1.0)
            label = "positive"
        elif bearish_count > bullish_count:
            score = -min(bearish_count / total, 1.0)
            label = "negative"
        else:
            score = 0.0
            label = "neutral"

        return {
            "label": label,
            "score": round(score, 4),
            "confidence": 0.40,  # User priority: rule = 0.40
            "model_source": "rule_based",
            "finbert_score": None,
            "llm_score": None,
            "agreement": True,
        }

    @staticmethod
    def _neutral_result() -> dict:
        """Return a neutral/unknown sentiment result."""
        return {
            "label": "neutral",
            "score": 0.0,
            "confidence": 0.1,
            "model_source": "none",
            "finbert_score": None,
            "llm_score": None,
            "agreement": True,
        }
