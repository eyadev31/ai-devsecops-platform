"""
Hybrid Intelligence Portfolio System — LLM Client Abstraction Layer
=====================================================================
Provides a provider-agnostic LLM interface with:
  - Gemini as default provider (free tier)
  - Abstraction layer for future migration to self-hosted or enterprise LLMs
  - Retry logic with exponential backoff
  - Token usage tracking
  - Response validation
  - Structured JSON output enforcement

To add a new provider, implement the BaseLLMClient interface.
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

from config.settings import LLMConfig, APIKeys

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
#  ABSTRACT BASE — Provider-Agnostic Interface
# ══════════════════════════════════════════════════════
class BaseLLMClient(ABC):
    """
    Abstract interface for LLM providers.
    Implement this to add a new LLM backend (OpenAI, Anthropic,
    self-hosted vLLM, Ollama, etc.).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = LLMConfig.TEMPERATURE,
        max_tokens: int = LLMConfig.MAX_OUTPUT_TOKENS,
        json_mode: bool = False,
    ) -> dict:
        """
        Generate LLM response.

        Returns:
            {
                "content": str,
                "usage": {"input_tokens": int, "output_tokens": int},
                "model": str,
                "latency_ms": float,
            }
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is configured and available."""
        pass


# ══════════════════════════════════════════════════════
#  GEMINI PROVIDER — Default Implementation
# ══════════════════════════════════════════════════════
class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client.
    Uses gemini-2.0-flash (free tier) by default.
    """

    def __init__(self, model: str = LLMConfig.MODEL):
        self._model_name = model
        self._client = None
        self._model = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    def _initialize(self):
        """Lazy initialization of Gemini client."""
        if self._client is not None:
            return

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )

        api_key = APIKeys.GEMINI_API_KEY
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY in your .env file. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=api_key)
        self._client = genai
        self._model = genai.GenerativeModel(self._model_name)
        logger.info(f"Gemini client initialized (model: {self._model_name})")

    def is_available(self) -> bool:
        """Check if Gemini is configured."""
        key_status = APIKeys.validate()
        return key_status.get("gemini", False)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = LLMConfig.TEMPERATURE,
        max_tokens: int = LLMConfig.MAX_OUTPUT_TOKENS,
        json_mode: bool = False,
    ) -> dict:
        """
        Generate response from Gemini with retry logic.

        Args:
            prompt: The user/analysis prompt
            system_prompt: System-level instruction
            temperature: Sampling temperature (low = deterministic)
            max_tokens: Maximum output tokens
            json_mode: If True, configure for JSON output

        Returns:
            {"content": str, "usage": dict, "model": str, "latency_ms": float}
        """
        self._initialize()

        # Build the full prompt with system instruction
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt

        if json_mode:
            full_prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no code blocks, no explanation outside JSON."

        # Generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": LLMConfig.TOP_P,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        # Retry with exponential backoff
        last_error = None
        for attempt in range(LLMConfig.MAX_RETRIES):
            try:
                start_time = time.time()

                response = self._model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract content
                content = ""
                if response.parts:
                    content = response.text
                elif hasattr(response, 'text'):
                    content = response.text

                # Usage tracking
                usage = {"input_tokens": 0, "output_tokens": 0}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage["input_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    usage["output_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0)

                self._total_input_tokens += usage["input_tokens"]
                self._total_output_tokens += usage["output_tokens"]
                self._call_count += 1

                logger.info(
                    f"Gemini response: {len(content)} chars, "
                    f"{usage['output_tokens']} tokens, {latency_ms:.0f}ms"
                )

                return {
                    "content": content,
                    "usage": usage,
                    "model": self._model_name,
                    "latency_ms": round(latency_ms, 1),
                }

            except Exception as e:
                last_error = e
                wait = LLMConfig.RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"Gemini attempt {attempt + 1}/{LLMConfig.MAX_RETRIES} failed: {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini failed after {LLMConfig.MAX_RETRIES} attempts: {last_error}")

    def get_usage_stats(self) -> dict:
        """Get cumulative token usage statistics."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_calls": self._call_count,
            "model": self._model_name,
        }


# ══════════════════════════════════════════════════════
#  LLM FACTORY — Provider Selection
# ══════════════════════════════════════════════════════
class LLMFactory:
    """
    Factory for creating LLM clients.
    Add new providers here as the system evolves.
    """

    _providers = {
        "gemini": GeminiClient,
    }

    @classmethod
    def create(cls, provider: Optional[str] = None) -> BaseLLMClient:
        """
        Create an LLM client for the specified provider.

        Args:
            provider: Provider name ('gemini', etc.). Default: from config.

        Returns:
            Configured LLM client instance
        """
        provider = provider or LLMConfig.PROVIDER

        # Lazy import of Groq if needed to avoid circular dependencies
        if provider == "groq" and "groq" not in cls._providers:
            try:
                from llm.groq_client import GroqClient
                cls.register_provider("groq", GroqClient)
            except ImportError:
                logger.error("GroqClient requested but not found or failed to import")

        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. Available: {available}"
            )

        client = cls._providers[provider](model=LLMConfig.MODEL)
        logger.info(f"LLM client created: {provider}")
        return client

    @classmethod
    def register_provider(cls, name: str, client_class: type) -> None:
        """
        Register a new LLM provider.
        For future extensibility: OpenAI, Anthropic, vLLM, Ollama, etc.
        """
        if not issubclass(client_class, BaseLLMClient):
            raise TypeError(f"Provider must implement BaseLLMClient interface")
        cls._providers[name] = client_class
        logger.info(f"LLM provider registered: {name}")
