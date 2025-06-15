import os
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI


class OpenAIEngine:
    """OpenAI engine for LLM interactions."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI engine."""
        self.model = model
        self.temperature = temperature
        self.model_params = model_params or {}

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = AsyncOpenAI(api_key=api_key)

    async def __call__(
        self, prompt: str, response_format: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate a response from the LLM."""
        try:
            # Create message with proper content structure
            message = {"role": "user", "content": [{"type": "text", "text": prompt}]}

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[message],
                temperature=self.temperature,
                response_format=response_format,
                **self.model_params,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response from OpenAI: {str(e)}")
