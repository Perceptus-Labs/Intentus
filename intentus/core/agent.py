from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import json
from pathlib import Path
import os

from .planner import Planner
from .memory import Memory
from .executor import Executor
from .initializer import Initializer


@dataclass
class AgentConfig:
    """Configuration for the IntentusAgent."""

    llm_engine: str = "gpt-41-mini"
    enabled_tools: List[str] = None
    max_steps: int = 10
    max_time: int = 300
    max_tokens: int = 4000
    cache_dir: str = "cache"
    verbose: bool = True

    def __post_init__(self):
        if self.enabled_tools is None:
            self.enabled_tools = ["all"]


class IntentusAgent:
    """Main agent class for orchestrating reasoning and execution."""

    def __init__(self, config: AgentConfig):
        self.config = config

        # Initialize components
        self.initializer = Initializer(
            llm_engine=config.llm_engine, enabled_tools=config.enabled_tools
        )

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
        self.executor.set_query_cache_dir(config.cache_dir)
        os.makedirs(config.cache_dir, exist_ok=True)

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
        start_time = time.time()
        result = {"task": task, "context": context, "start_time": start_time}

        if self.config.verbose:
            print(f"\n==> 🔍 Task: {task}")
            if context:
                print(f"==> 📝 Context: {json.dumps(context, indent=2)}")

        # [1] Analyze task
        task_analysis = await self.planner.analyze_query(task, context)
        result["analysis"] = task_analysis

        if self.config.verbose:
            print(f"\n==> 🔍 Task Analysis:\n{json.dumps(task_analysis, indent=2)}")

        # [2] Generate and execute plan
        step_count = 0
        while (
            step_count < self.config.max_steps
            and (time.time() - start_time) < self.config.max_time
        ):
            step_count += 1
            step_start = time.time()

            # Generate next step
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
                print(f"\n==> 🎯 Step {step_count}:")
                print(f"Context: {context}")
                print(f"Sub-goal: {sub_goal}")
                print(f"Tool: {tool_name}")

            # Validate tool
            if tool_name not in self.planner.available_tools:
                if self.config.verbose:
                    print(f"==> ⚠️ Tool '{tool_name}' not available")
                continue

            # Generate and execute command
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
                print(f"\n==> 📝 Command Generation:")
                print(f"Analysis: {analysis}")
                print(f"Explanation: {explanation}")
                print(f"Command: {command}")

            # Execute command
            execution_result = await self.executor.execute_tool_command(
                tool_name, command
            )

            if self.config.verbose:
                print(f"\n==> 🛠️ Execution Result:")
                print(json.dumps(execution_result, indent=2))

            # Update memory
            self.memory.add_action(
                step_count, tool_name, sub_goal, command, execution_result
            )

            # Verify if task is complete
            verification = await self.planner.verificate_context(
                task, context, task_analysis, self.memory
            )

            context_verification, conclusion = self.planner.extract_conclusion(
                verification
            )

            if self.config.verbose:
                print(f"\n==> 🤖 Verification:")
                print(f"Analysis: {context_verification}")
                print(f"Conclusion: {conclusion}")

            if conclusion == "STOP":
                break

        # Generate final output
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
            print(f"\n==> ✅ Task Complete!")
            print(f"Time: {result['execution_time']}s")
            print(f"Steps: {step_count}")
            print(f"\nFinal Output:\n{final_output}")

        return result
