import os
import json
import logging
from typing import Dict, Any, List, Optional

from ..engine.factory import create_llm_engine
from ..memory import Memory

# Initialize logger
logger = logging.getLogger(__name__)


class Executor:
    """Executor class for Intentus agent."""

    def __init__(
        self,
        llm_engine: str,
        toolbox_metadata: Dict[str, Any],
        available_tools: List[str],
        verbose: bool = True,
    ):
        """Initialize the executor."""
        self.llm_engine = create_llm_engine(llm_engine)
        self.toolbox_metadata = toolbox_metadata
        self.available_tools = available_tools
        self.verbose = verbose
        self.logger = logger

    async def execute_step(
        self, context: str, subgoal: str, tool: str, memory: Memory
    ) -> Dict[str, Any]:
        """Execute a single step using the specified tool."""
        if tool not in self.available_tools:
            return {
                "success": False,
                "error": f"Tool '{tool}' is not available",
                "result": None,
            }

        try:
            # Generate command
            command = await self.generate_tool_command(
                context=context, subgoal=subgoal, tool=tool
            )

            # Execute command
            result = await self.execute_tool_command(tool, command)

            return {"success": True, "command": command, "result": result}

        except Exception as e:
            return {"success": False, "error": str(e), "result": None}

    async def generate_tool_command(self, context: str, subgoal: str, tool: str) -> str:
        """Generate a command for the specified tool."""
        self.logger.debug(f"Generating command for tool: {tool}")
        self.logger.debug(f"Context: {context}")
        self.logger.debug(f"Subgoal: {subgoal}")
        self.logger.debug(f"Tool metadata: {self.toolbox_metadata[tool]}")

        prompt = f"""
Task: Generate a command for the {tool} tool based on the given context and subgoal.

Context: {context}
Subgoal: {subgoal}
Tool: {tool}
Tool Metadata: {self.toolbox_metadata[tool]}

Instructions:
1. Analyze the context and subgoal carefully.
2. Generate a command that will help achieve the subgoal using the specified tool.
3. Ensure the command follows the tool's requirements and best practices.

Response Format:
Your response MUST be a JSON object with exactly these fields:
{{
    "analysis": "Your analysis of how to use the tool",
    "command": "The actual command to execute"
}}

Rules:
- The command MUST be valid for the specified tool.
- The command MUST be specific and actionable.
- The command MUST help achieve the subgoal.
- The command MUST be a simple keyword or search term for Wikipedia searches.
- DO NOT include any explanatory text outside the JSON object.
"""

        self.logger.debug("Calling LLM engine for command generation")
        response = await self.llm_engine(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ToolCommand",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "analysis": {"type": "string"},
                            "command": {"type": "string"},
                        },
                        "required": ["analysis", "command"],
                    },
                },
            },
        )
        self.logger.debug(f"Raw LLM response: {response}")

        if isinstance(response, dict):
            command = response.get("command", "").strip()
            self.logger.debug(f"Extracted command from dict: '{command}'")
            if not command:
                raise ValueError("Empty command generated")
            return command
        else:
            # Parse the response to extract the command
            self.logger.debug("Response is not dict, parsing as string")
            try:
                data = json.loads(str(response))
                command = data.get("command", "").strip()
                self.logger.debug(f"Extracted command from JSON string: '{command}'")
                if not command:
                    raise ValueError("Empty command generated")
                return command
            except json.JSONDecodeError:
                self.logger.error("Failed to parse response as JSON")
                raise ValueError("Invalid response format")

    async def execute_tool_command(self, tool: str, command: str) -> Any:
        """Execute a command using the specified tool."""
        # Import the tool module
        try:
            # Convert tool name to directory name (e.g., Google_Search_Tool -> google_search)
            tool_dir = tool.lower().replace("_tool", "")
            module = __import__(f"intentus.tools.{tool_dir}.tool", fromlist=["Tool"])
            tool_class = getattr(module, tool)
            tool_instance = tool_class()

            # Execute the command
            result = await tool_instance.execute(command)
            return result

        except Exception as e:
            raise Exception(f"Error executing tool command: {str(e)}")
