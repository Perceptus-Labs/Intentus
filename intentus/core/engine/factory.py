from typing import Any, List, Optional
from ..config import LLMConfig
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_llm_engine(config: LLMConfig) -> Any:
    """Create an LLM engine instance based on the configuration."""
    model_string = config.engine.lower()

    if model_string.startswith("together-"):
        from .together import TogetherEngine

        return TogetherEngine(config)
    elif model_string.startswith("vllm-"):
        from .vllm import VLLMEngine

        return VLLMEngine(config)
    else:
        return MockLLMEngine(config)


class MockLLMEngine:
    """Mock LLM engine for development purposes."""

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        response = await self.client.chat.completions.create(
            model=self.config.engine,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content

    async def analyze_text(self, text: str) -> str:
        """Analyze text using the LLM."""
        response = await self.client.chat.completions.create(
            model=self.config.engine,
            messages=[{"role": "user", "content": text}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content

    async def __call__(
        self, input_data: List[Any], response_format: Optional[Any] = None
    ) -> Any:
        """Call the LLM engine with input data and optional response format."""
        if not input_data:
            raise ValueError("Input data cannot be empty")

        # Convert input data to string if it's a list
        if isinstance(input_data, list):
            prompt = (
                input_data[0] if isinstance(input_data[0], str) else str(input_data[0])
            )
        else:
            prompt = str(input_data)

        response = await self.client.chat.completions.create(
            model=self.config.engine,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            response_format=response_format,
        )

        return response.choices[0].message.content
