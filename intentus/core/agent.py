from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
import json
import os
import logging
from pathlib import Path
import asyncio

from .config import CoreConfig
from ..tools.config import ToolboxConfig
from .planner import Planner
from .memory import Memory
from .executor import Executor
from .initializer import Initializer
from .engine.factory import create_llm_engine

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("intentus.log")
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
console_format = logging.Formatter("%(message)s")
file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


@dataclass
class AgentConfig:
    """Configuration for the Intentus agent."""

    core: CoreConfig = field(default_factory=CoreConfig)
    toolbox: ToolboxConfig = field(default_factory=ToolboxConfig)

    def __post_init__(self):
        """Post-initialization setup."""
        # Set up logging level based on config
        logging.basicConfig(
            level=logging.DEBUG if self.core.verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


class IntentusAgent:
    """Main agent class for orchestrating reasoning and execution."""

    def __init__(self, config: AgentConfig):
        logger.info("Initializing IntentusAgent...")
        self.config = config

        # Initialize components
        self._initialize_components()

        # Set up cache directory
        logger.debug(f"Setting up cache directory: {config.core.executor.cache_dir}")
        self.executor.set_query_cache_dir(str(config.core.executor.cache_dir))
        os.makedirs(config.core.executor.cache_dir, exist_ok=True)
        logger.info("IntentusAgent initialized successfully")

    def _initialize_components(self):
        """Initialize core components."""
        # Initialize LLM engine
        self.llm_engine = create_llm_engine(self.config.core.llm)

        # Initialize tools
        self.initializer = Initializer(
            enabled_tools=self.config.toolbox.enabled_tools,
            llm_engine=self.config.core.llm.engine,
            verbose=self.config.core.verbose,
        )

        # Initialize planner
        self.planner = Planner(
            llm_engine=self.llm_engine, config=self.config.core.planner
        )

        # Initialize memory
        self.memory = Memory()

        # Initialize executor
        self.executor = Executor(
            llm_engine_name=self.config.core.llm.engine,
            root_cache_dir=str(self.config.core.executor.cache_dir),
            max_time=self.config.core.executor.timeout,
            verbose=self.config.core.verbose,
        )

    async def run(
        self,
        task: str,
        context: Dict[str, Any] = None,
        image: str = None,
    ) -> Dict[str, Any]:
        """
        Run the agent on a task.

        Args:
            task: The task to execute
            context: Additional context for the task
            image: Optional image path for visual tasks

        Returns:
            Dict containing the execution results
        """
        start_time = time.time()
        logger.info(f"Starting task execution: {task}")
        logger.info(f"🔍 Task: {task}")
        logger.info(f"📝 Context: {json.dumps(context, indent=2)}")

        try:
            # Analyze task
            logger.debug("Analyzing task...")
            task_analysis = await self.planner.analyze_query(task, image or "")

            # Validate tools
            logger.debug("Validating tools...")
            if not self.initializer.available_tools:
                raise ValueError("No tools available for execution")

            # Generate command
            logger.debug("Generating command...")
            command = await self.executor.generate_tool_command(
                question=task,
                image=image or "",
                context=json.dumps(context) if context else "",
                sub_goal=task_analysis,
                tool_name=self.initializer.available_tools[0],
                tool_metadata=self.initializer.toolbox_metadata,
            )

            # Execute command
            logger.debug("Executing command...")
            execution_result = await self.executor.execute_tool_command(
                self.initializer.available_tools[0], command
            )

            # Generate final output
            logger.debug("Generating final output...")
            final_output = await self.planner.generate_final_output(
                task, image or "", self.memory
            )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Prepare result
            result = {
                "task": task,
                "analysis": task_analysis,
                "step_count": 1,  # For now, we're only doing one step
                "execution_time": execution_time,
                "final_output": final_output,
            }

            return result

        except Exception as e:
            logger.error(f"Error executing task: {str(e)}", exc_info=True)
            raise
