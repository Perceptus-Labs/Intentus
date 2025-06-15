from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolMetadata:
    """Metadata for a tool."""

    name: str
    description: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ToolConfig:
    """Configuration for a specific tool."""

    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    custom_params: Dict[str, Any] = field(default_factory=dict)
    cache_results: bool = True
    cache_ttl: int = 3600  # 1 hour


@dataclass
class ToolboxConfig:
    """Configuration for the entire toolbox."""

    tools_dir: Path = Path("tools")
    enabled_tools: List[str] = field(default_factory=lambda: ["all"])
    default_timeout: int = 30
    default_retry_attempts: int = 3
    tool_configs: Dict[str, ToolConfig] = field(default_factory=dict)
    custom_tool_paths: List[Path] = field(default_factory=list)

    def get_tool_config(self, tool_name: str) -> ToolConfig:
        """Get configuration for a specific tool."""
        return self.tool_configs.get(tool_name, ToolConfig())

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        if "all" in self.enabled_tools:
            return True
        return tool_name in self.enabled_tools
