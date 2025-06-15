import argparse
import time
import json
import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .config import CoreConfig
from .engine.factory import create_llm_engine
from .initializer import Initializer
from .planner import Planner
from .memory import Memory
from .executor import Executor
from .formatters import QueryAnalysis, NextStep, MemoryVerification, ToolCommand


@dataclass
class AgentConfig:
    """Configuration for the Intentus agent."""

    llm_engine: str = "gpt-4.1-mini"
    enabled_tools: List[str] = None
    verbose: bool = True
    config_path: str = None
    max_steps: int = 5
    temperature: float = 0.7
    max_tokens: int = 4000


class IntentusAgent:
    """Main agent class for Intentus."""

    def __init__(self, config: AgentConfig):
        """Initialize the agent."""
        self.config = config
        self.llm_engine = create_llm_engine(config.llm_engine)

        # Initialize components
        self.initializer = Initializer(
            enabled_tools=config.enabled_tools,
            llm_engine=config.llm_engine,
            verbose=config.verbose,
            config_path=config.config_path,
        )

        # Get toolbox metadata and available tools
        self.toolbox_metadata = self.initializer.toolbox_metadata
        self.available_tools = self.initializer.available_tools

        # Initialize planner with new interface
        self.planner = Planner(
            llm_engine_name=config.llm_engine,
            toolbox_metadata=self.toolbox_metadata,
            available_tools=self.available_tools,
            verbose=config.verbose,
        )

        self.memory = Memory()
        self.executor = Executor(
            llm_engine=config.llm_engine,
            toolbox_metadata=self.toolbox_metadata,
            available_tools=self.available_tools,
            verbose=config.verbose,
        )

    async def run(self, question: str, image: str = None) -> Dict[str, Any]:
        """Run the agent on a task."""
        start_time = time.time()

        # Step 1: Analyze the query
        query_analysis = await self.planner.analyze_query(question, image)

        # Step 2: Generate base response
        base_response = await self.planner.generate_base_response(question, image)

        # Step 3: Main execution loop
        step_count = 0
        while step_count < self.config.max_steps:
            # Generate next step
            next_step = await self.planner.generate_next_step(
                question=question,
                image=image,
                query_analysis=query_analysis,
                memory=self.memory,
                step_count=step_count,
                max_step_count=self.config.max_steps,
            )

            # Extract context, subgoal, and tool
            context, subgoal, tool = self.planner.extract_context_subgoal_and_tool(
                next_step
            )

            # Execute the step
            result = await self.executor.execute_step(
                context=context, subgoal=subgoal, tool=tool, memory=self.memory
            )

            # Add to memory
            self.memory.add_action(
                step_count=step_count,
                tool_name=tool,
                sub_goal=subgoal,
                command=context,
                result=result,
            )

            # Verify if we should stop
            verification = await self.planner.verificate_context(
                question=question,
                image=image,
                query_analysis=query_analysis,
                memory=self.memory,
            )

            analysis, conclusion = self.planner.extract_conclusion(verification)
            if conclusion == "STOP":
                break

            step_count += 1

        # Step 4: Generate final output
        final_output = await self.planner.generate_final_output(
            question=question, image=image, memory=self.memory
        )

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "query_analysis": query_analysis,
            "base_response": base_response,
            "final_output": final_output,
            "execution_time": execution_time,
            "steps_taken": step_count + 1,
            "memory": self.memory.get_actions(),
        }


def create_agent(config: CoreConfig, verbose: bool = True) -> IntentusAgent:
    """
    Create an IntentusAgent instance with the given configuration.

    Args:
        config (CoreConfig): Configuration for the agent
        verbose (bool): Whether to print detailed logs

    Returns:
        IntentusAgent: Configured agent instance
    """
    return IntentusAgent(config, verbose=verbose)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the octotools demo with specified parameters."
    )
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)",
    )
    parser.add_argument(
        "--enabled_tools",
        default="Generalist_Solution_Generator_Tool",
        help="List of enabled tools.",
    )
    parser.add_argument(
        "--root_cache_dir",
        default="solver_cache",
        help="Path to solver cache directory.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="Maximum tokens for LLM generation.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=10, help="Maximum number of steps to execute."
    )
    parser.add_argument(
        "--max_time", type=int, default=300, help="Maximum time allowed in seconds."
    )
    parser.add_argument(
        "--verbose", type=bool, default=True, help="Enable verbose output."
    )
    return parser.parse_args()


def main(args):
    config = CoreConfig(
        llm_engine=args.llm_engine_name,
        enabled_tools=args.enabled_tools,
        cache_dir=args.root_cache_dir,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_output_length=args.max_tokens,
    )
    agent = create_agent(config, verbose=args.verbose)

    # Solve the task or problem
    result = agent.run("What is the capital of France?")
    print(f"\nTask Result:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
