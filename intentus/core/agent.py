from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import json
import os
import logging
from pathlib import Path

from .planner import Planner
from .memory import Memory
from .executor import Executor
from .initializer import Initializer

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
    """Configuration for the IntentusAgent."""

    llm_engine: str = "gpt-4.1-mini"
    enabled_tools: List[str] = None
    max_steps: int = 10
    max_time: int = 300
    max_tokens: int = 4000
    cache_dir: str = "cache"
    verbose: bool = True
    log_level: str = "DEBUG"

    def __post_init__(self):
        if self.enabled_tools is None:
            self.enabled_tools = ["all"]
        # Set log level based on config
        logger.setLevel(getattr(logging, self.log_level.upper()))


class IntentusAgent:
    """Main agent class for orchestrating reasoning and execution."""

    def __init__(self, config: AgentConfig):
        logger.info("Initializing IntentusAgent...")
        self.config = config

        # Initialize components
        logger.debug("Initializing components...")
        self.initializer = Initializer(
            llm_engine=config.llm_engine, enabled_tools=config.enabled_tools
        )
        logger.debug(f"Available tools: {self.initializer.get_available_tools()}")

        self.planner = Planner(
            llm_engine=config.llm_engine,
            available_tools=self.initializer.get_available_tools(),
            toolbox_metadata=self.initializer.get_toolbox_metadata(),
        )

        self.memory = Memory()

        self.executor = Executor(
            llm_engine=config.llm_engine,
            toolbox_metadata=self.initializer.get_toolbox_metadata(),
        )

        # Set up cache directory
        logger.debug(f"Setting up cache directory: {config.cache_dir}")
        self.executor.set_query_cache_dir(config.cache_dir)
        os.makedirs(config.cache_dir, exist_ok=True)
        logger.info("IntentusAgent initialized successfully")

    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the agent on a given task.

        Args:
            task: The task to execute
            context: Optional context information (e.g., image paths, additional data)

        Returns:
            Dict containing execution results and metadata
        """
        logger.info(f"Starting task execution: {task}")
        start_time = time.time()
        result = {"task": task, "context": context, "start_time": start_time}

        if self.config.verbose:
            logger.info(f"🔍 Task: {task}")
            if context:
                logger.info(f"📝 Context: {json.dumps(context, indent=2)}")

        # [1] Analyze task
        logger.debug("Analyzing task...")
        task_analysis = await self.planner.analyze_query(task, context)
        result["analysis"] = task_analysis

        if self.config.verbose:
            logger.info(f"🔍 Task Analysis:\n{json.dumps(task_analysis, indent=2)}")

        # [2] Generate and execute plan
        step_count = 0
        while (
            step_count < self.config.max_steps
            and (time.time() - start_time) < self.config.max_time
        ):
            step_count += 1
            step_start = time.time()
            logger.info(f"Starting step {step_count}")

            # Generate next step
            logger.debug("Generating next step...")
            next_step = await self.planner.generate_next_step(
                task,
                context,
                task_analysis,
                self.memory,
                step_count,
                self.config.max_steps,
            )

            # Extract components
            context, sub_goal, tool_name = (
                self.planner.extract_context_subgoal_and_tool(next_step)
            )

            if self.config.verbose:
                logger.info(f"🎯 Step {step_count}:")
                logger.info(f"Context: {context}")
                logger.info(f"Sub-goal: {sub_goal}")
                logger.info(f"Tool: {tool_name}")

            # Validate tool
            if tool_name not in self.planner.available_tools:
                logger.warning(f"⚠️ Tool '{tool_name}' not available")
                continue

            # Generate and execute command
            logger.debug(f"Generating command for tool: {tool_name}")
            tool_command = await self.executor.generate_tool_command(
                task,
                context,
                context,
                sub_goal,
                tool_name,
                self.planner.toolbox_metadata[tool_name],
            )

            analysis, explanation, command = (
                self.executor.extract_explanation_and_command(tool_command)
            )

            if self.config.verbose:
                logger.info(f"📝 Command Generation:")
                logger.info(f"Analysis: {analysis}")
                logger.info(f"Explanation: {explanation}")
                logger.info(f"Command: {command}")

            # Execute command
            logger.debug(f"Executing command for tool: {tool_name}")
            execution_result = await self.executor.execute_tool_command(
                tool_name, command
            )

            if self.config.verbose:
                logger.info(f"🛠️ Execution Result:")
                logger.info(json.dumps(execution_result, indent=2))

            # Update memory
            logger.debug("Updating memory...")
            self.memory.add_action(
                step_count, tool_name, sub_goal, command, execution_result
            )

            # Verify if task is complete
            logger.debug("Verifying context...")
            verification = await self.planner.verificate_context(
                task, context, task_analysis, self.memory
            )

            context_verification, conclusion = self.planner.extract_conclusion(
                verification
            )

            if self.config.verbose:
                logger.info(f"🤖 Verification:")
                logger.info(f"Analysis: {context_verification}")
                logger.info(f"Conclusion: {conclusion}")

            if conclusion == "STOP":
                logger.info("Task verification complete - stopping execution")
                break

        # Generate final output
        logger.debug("Generating final output...")
        final_output = await self.planner.generate_final_output(
            task, context, self.memory
        )
        result.update(
            {
                "final_output": final_output,
                "memory": self.memory.get_actions(),
                "step_count": step_count,
                "execution_time": round(time.time() - start_time, 2),
            }
        )

        if self.config.verbose:
            logger.info(f"✅ Task Complete!")
            logger.info(f"Time: {result['execution_time']}s")
            logger.info(f"Steps: {step_count}")
            logger.info(f"\nFinal Output:\n{final_output}")

        logger.info("Task execution completed successfully")
        return result
