import os
from typing import Any, Dict, List, Optional
from vllm import AsyncLLMEngine, SamplingParams


class VLLMEngine:
    """VLLM engine for LLM interactions."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the VLLM engine."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_params = model_params or {}

        # Initialize VLLM engine
        self.engine = AsyncLLMEngine(model=model, **self.model_params)

    async def __call__(
        self, prompt: str, response_format: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate a response from the LLM."""
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Generate response
            outputs = await self.engine.generate(prompt, sampling_params)

            # Extract and return the generated text
            return outputs[0].outputs[0].text
        except Exception as e:
            raise Exception(f"Error generating response from VLLM: {str(e)}")
