"""Ollama integration for local LLM inference."""

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

from .models import PlanningError

logger = logging.getLogger(__name__)

# Recommended models by hardware tier (from PRD)
RECOMMENDED_MODELS = {
    "low": "llama3.1:8b-instruct-q4_K_M",  # 8GB VRAM
    "mid": "llama3.1:14b-instruct-q8_0",  # 12-16GB VRAM
    "high": "llama3.1:14b-instruct-fp16",  # 24GB+ VRAM
    "cpu": "llama3.1:8b-instruct-q4_K_M",  # CPU fallback
}

# Fallback models if recommended not available
FALLBACK_MODELS = [
    "llama3.1:latest",
    "llama3:latest",
    "mistral:latest",
    "gemma2:latest",
]


@dataclass
class OllamaResponse:
    """Response from Ollama API."""

    content: str
    model: str
    total_duration_ms: int = 0
    eval_count: int = 0

    @property
    def tokens_per_second(self) -> float:
        if self.total_duration_ms > 0:
            return self.eval_count / (self.total_duration_ms / 1000)
        return 0.0


class OllamaClient:
    """
    Client for Ollama local LLM inference.

    Uses the ollama CLI for simplicity and reliability.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the Ollama client.

        Args:
            model: Model name to use (auto-selects if None)
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self._model = model
        self._available_models: Optional[list[str]] = None

    @property
    def model(self) -> str:
        """Get the current model, selecting one if not set."""
        if self._model is None:
            self._model = self._select_model()
        return self._model

    def _select_model(self) -> str:
        """Select the best available model."""
        available = self.list_models()

        if not available:
            raise PlanningError(
                "No Ollama models available. Run: ollama pull llama3.1:8b-instruct-q4_K_M"
            )

        # Check for recommended models first
        for tier_model in RECOMMENDED_MODELS.values():
            base_name = tier_model.split(":")[0]
            for avail in available:
                if avail.startswith(base_name):
                    logger.info(f"Selected model: {avail}")
                    return avail

        # Check fallbacks
        for fallback in FALLBACK_MODELS:
            base_name = fallback.split(":")[0]
            for avail in available:
                if avail.startswith(base_name):
                    logger.info(f"Selected fallback model: {avail}")
                    return avail

        # Use whatever is available
        model = available[0]
        logger.warning(f"Using available model: {model}")
        return model

    def list_models(self) -> list[str]:
        """List available Ollama models."""
        if self._available_models is not None:
            return self._available_models

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"Failed to list models: {result.stderr}")
                return []

            # Parse output (skip header line)
            models = []
            for line in result.stdout.strip().split("\n")[1:]:
                if line.strip():
                    # First column is model name
                    model_name = line.split()[0]
                    models.append(model_name)

            self._available_models = models
            return models

        except subprocess.TimeoutExpired:
            logger.error("Timeout listing models - is Ollama running?")
            return []
        except FileNotFoundError:
            logger.error("Ollama not found - is it installed?")
            return []

    def is_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> OllamaResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            OllamaResponse with generated content

        Raises:
            PlanningError: If generation fails
        """
        model = self.model

        # Build the full prompt
        full_prompt = ""
        if system:
            full_prompt = f"System: {system}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant:"

        logger.debug(f"Generating with model {model}, temp={temperature}")

        try:
            result = subprocess.run(
                [
                    "ollama",
                    "run",
                    model,
                    "--nowordwrap",
                ],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise PlanningError(f"Ollama generation failed: {result.stderr}")

            content = result.stdout.strip()

            return OllamaResponse(
                content=content,
                model=model,
            )

        except subprocess.TimeoutExpired:
            raise PlanningError("LLM generation timed out after 5 minutes")
        except FileNotFoundError:
            raise PlanningError("Ollama not found - is it installed?")

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,  # Lower temp for structured output
    ) -> dict:
        """
        Generate a JSON response from the LLM.

        Args:
            prompt: User prompt (should request JSON output)
            system: System prompt
            temperature: Sampling temperature

        Returns:
            Parsed JSON dictionary

        Raises:
            PlanningError: If generation or parsing fails
        """
        # Add JSON instruction to prompt
        json_prompt = prompt + "\n\nRespond with valid JSON only, no other text."

        response = self.generate(
            prompt=json_prompt,
            system=system,
            temperature=temperature,
        )

        # Extract JSON from response
        content = response.content

        # Try to find JSON in the response
        try:
            # First try direct parsing
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                try:
                    return json.loads(content[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try to find any JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        raise PlanningError(f"Failed to parse JSON from LLM response: {content[:200]}...")


# Global client instance
_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create the global Ollama client."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
