from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CoreConfig:
    """Main configuration for the agent."""

    # LLM settings
    llm_engine: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_tokens: int = 4000
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Tool settings
    enabled_tools: List[str] = field(default_factory=lambda: ["all"])
    tools_dir: Path = Path("intentus/tools")
    config_path: Optional[Path] = None

    # Execution settings
    max_steps: int = 10
    max_time: int = 300
    cache_dir: Path = Path("cache")
    num_threads: int = 1

    # Logging settings
    verbose: bool = True
    log_level: str = "DEBUG"

    def __post_init__(self):
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure tools directory exists
        if not self.tools_dir.exists():
            raise ValueError(f"Tools directory not found: {self.tools_dir}")
