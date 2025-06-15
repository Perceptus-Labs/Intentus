from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM engine."""

    engine: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_tokens: int = 4000
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerConfig:
    """Configuration for the Planner component."""

    max_steps: int = 10
    max_time: int = 300
    verification_threshold: float = 0.8
    custom_prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for the Memory component."""

    max_history: int = 100
    persist_path: Optional[Path] = None
    auto_save: bool = True
    save_interval: int = 10


@dataclass
class ExecutorConfig:
    """Configuration for the Executor component."""

    cache_dir: Path = Path("cache")
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1


@dataclass
class CoreConfig:
    """Main configuration for core components."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    verbose: bool = True
    log_level: str = "DEBUG"

    def __post_init__(self):
        # Ensure cache directory exists
        self.executor.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure memory persist directory exists if specified
        if self.memory.persist_path:
            self.memory.persist_path.mkdir(parents=True, exist_ok=True)
