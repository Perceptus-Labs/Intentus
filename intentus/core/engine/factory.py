from typing import Any, List, Optional, Dict, Union
from ..config import CoreConfig
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_llm_engine(config: Union[str, CoreConfig]) -> Any:
    """Create an LLM engine based on the configuration."""
    # If config is a string, create a basic CoreConfig
    if isinstance(config, str):
        config = CoreConfig(llm_engine=config)

    if config.llm_engine == "gpt-4.1-mini":
        from .openai_engine import OpenAIEngine

        return OpenAIEngine(
            model=config.llm_engine,
            temperature=config.temperature,
            model_params=config.model_params,
        )
    elif config.llm_engine == "vllm":
        from .vllm_engine import VLLMEngine

        return VLLMEngine(
            model=config.llm_engine,
            temperature=config.temperature,
            model_params=config.model_params,
        )
    else:
        raise ValueError(f"Unsupported LLM engine: {config.llm_engine}")


class MockLLMEngine:
    """Mock LLM engine for development."""

    def __init__(self, config: Union[str, CoreConfig]):
        """Initialize the mock engine."""
        if isinstance(config, str):
            config = CoreConfig(llm_engine=config)
        self.config = config
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        response = await self.client.chat.completions.create(
            model=self.config.llm_engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content

    async def analyze_text(self, text: str) -> str:
        """Analyze text using the LLM."""
        response = await self.client.chat.completions.create(
            model=self.config.engine,
            messages=[{"role": "user", "content": text}],
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

        # If response_format is a Pydantic model, use beta API
        if response_format and hasattr(response_format, "__pydantic_model__"):
            response = await self.client.beta.chat.completions.parse(
                model=self.config.engine,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                response_format=response_format,
            )
        else:
            response = await self.client.chat.completions.create(
                model=self.config.engine,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                response_format=response_format,
            )

        return response.choices[0].message.content
