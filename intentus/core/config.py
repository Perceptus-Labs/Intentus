from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CoreConfig:
    """Main configuration for the agent."""

    # LLM settings
    llm_engine: str = "gpt-4.1-mini"
    temperature: float = 0.7
    model_params: Dict[str, Any] = field(default_factory=dict)
    use_cache: bool = False  # Whether to use caching for LLM responses

    # Tool settings
    enabled_tools: List[str] = field(default_factory=lambda: ["all"])
    tools_dir: Path = Path("intentus/tools")
    config_path: Optional[Path] = None

    # Execution settings
    max_steps: int = 10
    max_time: int = 300
    cache_dir: Optional[Path] = None  # Optional cache directory
    num_threads: int = 1

    # Logging settings
    verbose: bool = True
    log_level: str = "DEBUG"

    def __post_init__(self):
        # Only create cache directory if caching is enabled
        if self.use_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure tools directory exists
        if not self.tools_dir.exists():
            raise ValueError(f"Tools directory not found: {self.tools_dir}")
