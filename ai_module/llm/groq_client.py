"""
Hybrid Intelligence Portfolio System — Groq LLM Client
======================================================
High-performance LPU-powered LLM client for:
  - Llama-3.3-70B-SpecDec
  - Mixtral-8x7b-32768
  - deepseek-r1-distill-llama-70b

Provides sub-second latency and high reliability for quant reasoning.
"""

import json
import time
import logging
from typing import Optional

from config.settings import LLMConfig, APIKeys
from llm.gemini_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """
    Groq Cloud API client.
    Uses llama-3.3-70b-versatile by default for institutional reasoning.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self._model_name = model
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    def _initialize(self):
        """Lazy initialization of Groq client."""
        if self._client is not None:
            return

        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq not installed. Run: pip install groq"
            )

        api_key = APIKeys.GROQ_API_KEY
        if not api_key:
            raise ValueError(
                "Groq API key not configured. "
                "Set GROQ_API_KEY in your .env file. "
                "Get a free key at https://console.groq.com/keys"
            )

        self._client = Groq(api_key=api_key)
        logger.info(f"Groq client initialized (model: {self._model_name})")

    def is_available(self) -> bool:
        """Check if Groq is configured."""
        key_status = APIKeys.validate()
        return key_status.get("groq", False)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = LLMConfig.TEMPERATURE,
        max_tokens: int = LLMConfig.MAX_OUTPUT_TOKENS,
        json_mode: bool = False,
    ) -> dict:
        """Generate response from Groq."""
        self._initialize()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        # Groq specific JSON mode requirement: the word "JSON" must be in the system/user prompt
        if json_mode and "JSON" not in prompt and (not system_prompt or "JSON" not in system_prompt):
            messages[-1]["content"] += "\n\nReturn the output in valid JSON format."

        try:
            start_time = time.time()
            
            completion = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=LLMConfig.TOP_P,
                stream=False,
                response_format={"type": "json_object"} if json_mode else {"type": "text"},
            )

            latency_ms = (time.time() - start_time) * 1000
            content = completion.choices[0].message.content
            
            # Usage tracking
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }

            self._total_input_tokens += usage["input_tokens"]
            self._total_output_tokens += usage["output_tokens"]
            self._call_count += 1

            logger.info(
                f"Groq response: {len(content)} chars, "
                f"{usage['output_tokens']} tokens, {latency_ms:.0f}ms ({self._model_name})"
            )

            return {
                "content": content,
                "usage": usage,
                "model": self._model_name,
                "latency_ms": round(latency_ms, 1),
            }

        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise RuntimeError(f"Groq failed: {e}")

    def get_usage_stats(self) -> dict:
        """Get cumulative token usage statistics."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_calls": self._call_count,
            "model": self._model_name,
        }
