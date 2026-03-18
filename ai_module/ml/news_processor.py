"""
Hybrid Intelligence Portfolio System — News Processor
=======================================================
Agent 5: Text cleaning, relevance filtering, entity extraction,
and topic classification for financial news articles.

Pipeline:
  1. Clean raw text (HTML, boilerplate, noise)
  2. Score financial relevance
  3. Extract entities (spaCy NER + custom financial matcher)
  4. Classify topics
  5. Map entities to portfolio asset classes
"""

import re
import logging
import hashlib
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  FINANCIAL ENTITY PATTERNS (Custom NER)
# ═══════════════════════════════════════════════════════

FINANCIAL_ENTITIES = {
    # Tickers & Assets
    r"\bBTC\b": ("Bitcoin", "crypto"),
    r"\bETH\b": ("Ethereum", "crypto"),
    r"\bSOL\b": ("Solana", "crypto"),
    r"\bXRP\b": ("XRP", "crypto"),
    r"\bBNB\b": ("BNB", "crypto"),
    r"\bbitcoin\b": ("Bitcoin", "crypto"),
    r"\bethereum\b": ("Ethereum", "crypto"),
    r"\bcrypto(?:currency|currencies)?\b": ("Crypto", "crypto"),
    r"\bS&P\s?500\b": ("S&P 500", "equities"),
    r"\bSPY\b": ("SPY", "equities"),
    r"\bNasdaq\b": ("Nasdaq", "equities"),
    r"\bQQQ\b": ("QQQ", "equities"),
    r"\bDow\s?Jones\b": ("Dow Jones", "equities"),
    r"\bNVIDIA\b": ("NVIDIA", "equities"),
    r"\bApple\b": ("Apple", "equities"),
    r"\bMicrosoft\b": ("Microsoft", "equities"),
    r"\bTesla\b": ("Tesla", "equities"),
    r"\bAmazon\b": ("Amazon", "equities"),
    r"\bGoogle\b": ("Google", "equities"),
    r"\bMeta\b": ("Meta", "equities"),
    r"\bgold\b": ("Gold", "commodities"),
    r"\bsilver\b": ("Silver", "commodities"),
    r"\boil\b": ("Oil", "commodities"),
    r"\bcrude\b": ("Crude Oil", "commodities"),
    r"\bcopper\b": ("Copper", "commodities"),
    r"\blithium\b": ("Lithium", "commodities"),
    r"\bTreasur(?:y|ies)\b": ("Treasury", "bonds"),
    r"\bbond(?:s)?\b": ("Bonds", "bonds"),
    r"\byield\s?curve\b": ("Yield Curve", "bonds"),
    r"\bdollar\b": ("USD", "forex"),
    r"\beuro\b": ("EUR", "forex"),
    r"\byen\b": ("JPY", "forex"),
    r"\bsterling\b": ("GBP", "forex"),

    # Institutions
    r"\bFederal\s?Reserve\b": ("Federal Reserve", "institution"),
    r"\bFed\b": ("Federal Reserve", "institution"),
    r"\bFOMC\b": ("FOMC", "institution"),
    r"\bECB\b": ("ECB", "institution"),
    r"\bBank\s?of\s?Japan\b": ("BOJ", "institution"),
    r"\bSEC\b": ("SEC", "institution"),
    r"\bCFTC\b": ("CFTC", "institution"),
    r"\bIMF\b": ("IMF", "institution"),
    r"\bWorld\s?Bank\b": ("World Bank", "institution"),
    r"\bGoldman\s?Sachs\b": ("Goldman Sachs", "institution"),
    r"\bJPMorgan\b": ("JPMorgan", "institution"),
    r"\bMorgan\s?Stanley\b": ("Morgan Stanley", "institution"),
    r"\bBlackRock\b": ("BlackRock", "institution"),
}

# ═══════════════════════════════════════════════════════
#  TOPIC CLASSIFICATION PATTERNS
# ═══════════════════════════════════════════════════════

TOPIC_PATTERNS = {
    "interest_rates": [
        r"interest\s?rate", r"rate\s?(cut|hike|decision|increase|decrease)",
        r"fed\s?funds", r"basis\s?point", r"monetary\s?policy",
    ],
    "monetary_policy": [
        r"federal\s?reserve", r"\bfed\b", r"\bfomc\b", r"central\s?bank",
        r"quantitative\s?(easing|tightening)", r"\bQE\b", r"\bQT\b",
    ],
    "inflation": [
        r"inflation", r"\bCPI\b", r"\bPCE\b", r"consumer\s?price",
        r"price\s?(increase|pressure)", r"cost\s?of\s?living",
    ],
    "regulation": [
        r"\bSEC\b", r"regulat", r"compliance", r"enforcement",
        r"crackdown", r"lawsuit", r"legal\s?action", r"ban(?:ned)?",
    ],
    "geopolitical": [
        r"geopolitical", r"war\b", r"conflict", r"sanction",
        r"tariff", r"trade\s?war", r"embargo", r"invasion",
    ],
    "recession": [
        r"recession", r"slowdown", r"contraction", r"downturn",
        r"unemployment\s?(rise|increase|claims)", r"layoff",
    ],
    "crypto": [
        r"bitcoin", r"ethereum", r"crypto", r"\bETF\b.*crypto",
        r"\bDeFi\b", r"\bNFT\b", r"blockchain", r"stablecoin",
    ],
    "earnings": [
        r"earnings", r"revenue", r"profit", r"quarterly\s?result",
        r"beat\s?expectations", r"miss(?:ed)?\s?expectations", r"\bEPS\b",
    ],
    "commodities": [
        r"gold\s?price", r"oil\s?price", r"crude\s?oil", r"commodity",
        r"supply\s?chain", r"OPEC", r"mining", r"energy\s?market",
    ],
    "equities": [
        r"stock\s?market", r"S&P\s?500", r"Nasdaq", r"equity",
        r"bull\s?market", r"bear\s?market", r"rally", r"selloff",
    ],
    "forex": [
        r"forex", r"currency", r"exchange\s?rate", r"dollar\s?(strength|weakness)",
        r"carry\s?trade", r"\bFX\b",
    ],
    "ipo_merger": [
        r"\bIPO\b", r"initial\s?public\s?offering", r"merger",
        r"acquisition", r"takeover", r"buyout",
    ],
}


class NewsProcessor:
    """
    Financial news processing engine.
    
    Cleans, filters, classifies, and extracts structured data
    from raw news articles.
    """

    def __init__(self):
        self._spacy_model = None  # Lazy loaded
        self._processing_stats = {
            "total_input": 0,
            "filtered_low_relevance": 0,
            "processed_output": 0,
        }

    @property
    def stats(self) -> dict:
        return self._processing_stats.copy()

    def process_batch(
        self,
        articles: list[dict],
        min_relevance: float = 0.15,
    ) -> list[dict]:
        """
        Process a batch of raw articles.
        
        Args:
            articles: Raw article dicts from NewsCollector
            min_relevance: Minimum relevance score (0-1) to keep
            
        Returns:
            Processed articles with entities, topics, relevance scores
        """
        self._processing_stats = {
            "total_input": len(articles),
            "filtered_low_relevance": 0,
            "processed_output": 0,
        }

        processed = []
        for article in articles:
            result = self._process_single(article)
            if result and result["relevance_score"] >= min_relevance:
                processed.append(result)
            else:
                self._processing_stats["filtered_low_relevance"] += 1

        self._processing_stats["processed_output"] = len(processed)

        # Sort by relevance
        processed.sort(key=lambda a: a["relevance_score"], reverse=True)

        logger.info(
            f"Processing Complete: {len(processed)}/{len(articles)} articles passed "
            f"({self._processing_stats['filtered_low_relevance']} filtered)"
        )

        return processed

    def _process_single(self, article: dict) -> Optional[dict]:
        """Process a single article through the full pipeline."""
        title = article.get("title", "").strip()
        summary = article.get("summary", "").strip()

        if not title:
            return None

        # Step 1: Clean text
        title = self._clean_text(title)
        summary = self._clean_text(summary)
        combined_text = f"{title}. {summary}" if summary else title

        # Step 2: Score relevance
        relevance = self._score_relevance(combined_text)

        # Step 3: Extract entities
        entities, asset_classes, organizations = self._extract_entities(combined_text)

        # Step 4: Classify topic
        topic = self._classify_topic(combined_text)

        # Step 5: Generate article ID
        article_id = hashlib.sha256(
            f"{title}_{article.get('source', '')}_{article.get('published_at', '')}".encode()
        ).hexdigest()[:16]

        return {
            "article_id": article_id,
            "title": title,
            "summary": summary,
            "combined_text": combined_text,
            "source": article.get("source", "unknown"),
            "url": article.get("url", ""),
            "published_at": article.get("published_at", datetime.now(timezone.utc).isoformat()),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "relevance_score": relevance,
            "entities": entities,
            "asset_classes": asset_classes,
            "organizations": organizations,
            "topic": topic,
        }

    # ═════════════════════════════════════════════════════
    #  TEXT CLEANING
    # ═════════════════════════════════════════════════════

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean article text: remove HTML, excess whitespace, boilerplate."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        # Remove special characters (keep financial symbols)
        text = re.sub(r"[^\w\s.,;:!?$%&'\"()-/]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove common boilerplate phrases
        boilerplate = [
            r"subscribe to our newsletter",
            r"click here to read more",
            r"advertisement",
            r"sponsored content",
            r"cookie policy",
            r"privacy policy",
            r"terms of service",
            r"all rights reserved",
            r"read the full article",
            r"\[removed\]",
            r"\[chars\]",
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text.strip()

    # ═════════════════════════════════════════════════════
    #  RELEVANCE SCORING
    # ═════════════════════════════════════════════════════

    def _score_relevance(self, text: str) -> float:
        """
        Score article financial relevance (0.0 to 1.0).
        Uses keyword density and entity presence.
        """
        from config.settings import NewsConfig

        if not text:
            return 0.0

        text_lower = text.lower()
        word_count = max(len(text_lower.split()), 1)

        # Count financial keyword matches
        keyword_hits = 0
        for keyword in NewsConfig.FINANCIAL_KEYWORDS:
            if keyword.lower() in text_lower:
                keyword_hits += 1

        # Count entity matches
        entity_hits = 0
        for pattern in FINANCIAL_ENTITIES:
            if re.search(pattern, text, re.IGNORECASE):
                entity_hits += 1

        # Keyword density score (capped at 1.0)
        keyword_density = min(keyword_hits / max(len(NewsConfig.FINANCIAL_KEYWORDS) * 0.1, 1), 1.0)

        # Entity presence score
        entity_score = min(entity_hits / 5.0, 1.0)

        # Combined relevance
        relevance = (keyword_density * 0.5) + (entity_score * 0.5)

        return round(min(relevance, 1.0), 4)

    # ═════════════════════════════════════════════════════
    #  ENTITY EXTRACTION
    # ═════════════════════════════════════════════════════

    def _extract_entities(self, text: str) -> tuple[list[str], list[str], list[str]]:
        """
        Extract financial entities from text.
        
        Returns:
            (entities, asset_classes, organizations)
        """
        entities = []
        asset_classes = set()
        organizations = []

        # Custom financial entity matching (primary — always available)
        for pattern, (entity_name, entity_type) in FINANCIAL_ENTITIES.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(entity_name)
                if entity_type == "institution":
                    organizations.append(entity_name)
                else:
                    asset_classes.add(entity_type)

        # spaCy NER (secondary — if available)
        spacy_entities = self._extract_spacy_entities(text)
        for ent_text, ent_label in spacy_entities:
            if ent_label in ("ORG", "PERSON") and ent_text not in entities:
                if ent_text not in organizations:
                    organizations.append(ent_text)
            elif ent_label in ("GPE", "LOC") and ent_text not in entities:
                entities.append(ent_text)

        # Deduplicate
        entities = list(dict.fromkeys(entities))
        organizations = list(dict.fromkeys(organizations))

        return entities, list(asset_classes), organizations

    def _extract_spacy_entities(self, text: str) -> list[tuple[str, str]]:
        """Extract entities using spaCy NER (lazy loaded)."""
        try:
            if self._spacy_model is None:
                import spacy
                try:
                    self._spacy_model = spacy.load("en_core_web_sm")
                except OSError:
                    logger.info("spaCy model 'en_core_web_sm' not found. Using regex-only NER.")
                    self._spacy_model = False
                    return []

            if self._spacy_model is False:
                return []

            doc = self._spacy_model(text[:1000])  # Limit text length for performance
            return [(ent.text, ent.label_) for ent in doc.ents]

        except ImportError:
            logger.debug("spaCy not installed. Using regex-only entity extraction.")
            return []
        except Exception as e:
            logger.debug(f"spaCy NER failed: {e}")
            return []

    # ═════════════════════════════════════════════════════
    #  TOPIC CLASSIFICATION
    # ═════════════════════════════════════════════════════

    @staticmethod
    def _classify_topic(text: str) -> str:
        """Classify article into primary financial topic."""
        if not text:
            return "general_market"

        scores = {}
        for topic, patterns in TOPIC_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches)
            if score > 0:
                scores[topic] = score

        if not scores:
            return "general_market"

        # Return highest-scoring topic
        return max(scores, key=scores.get)
