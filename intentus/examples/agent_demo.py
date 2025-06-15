import asyncio
import logging
from pathlib import Path
from intentus.core.agent import AgentConfig, IntentusAgent
from intentus.core.config import (
    CoreConfig,
    LLMConfig,
    PlannerConfig,
    MemoryConfig,
    ExecutorConfig,
)
from intentus.tools.config import ToolboxConfig, ToolConfig
import json

# Set up logging for the example
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intentus/example.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def run_agent_demo():
    """Run a demonstration of the IntentusAgent with various tasks."""

    # Create core configuration
    core_config = CoreConfig(
        llm=LLMConfig(
            engine="gpt-4.1-mini",
            temperature=0.7,
            max_tokens=4000,
        ),
        planner=PlannerConfig(
            max_steps=5,
            max_time=60,
            verification_threshold=0.8,
        ),
        memory=MemoryConfig(
            max_history=50,
            persist_path=Path("intentus/example_memory"),
            auto_save=True,
        ),
        executor=ExecutorConfig(
            cache_dir=Path("intentus/example_cache"),
            timeout=30,
            retry_attempts=3,
        ),
        verbose=True,
        log_level="DEBUG",
    )

    # Create toolbox configuration
    toolbox_config = ToolboxConfig(
        tools_dir=Path("tools"),  # This is relative to the project root
        enabled_tools=["Google_Search_Tool"],
        tool_configs={
            "Google_Search_Tool": ToolConfig(
                timeout=60,
                retry_attempts=2,
                cache_results=True,
                cache_ttl=3600,
            ),
        },
    )

    # Create agent configuration
    config = AgentConfig(
        core=core_config,
        toolbox=toolbox_config,
    )

    # Initialize agent
    logger.info("Initializing IntentusAgent...")
    agent = IntentusAgent(config)

    # Example tasks
    tasks = [
        {
            "task": "What is the current weather in Tokyo?",
            "context": {
                "require_accuracy": True,
                "preferred_units": "metric",
            },
        },
        {
            "task": "Calculate the population density of New York City",
            "context": {
                "require_accuracy": True,
                "preferred_units": "km²",
            },
        },
    ]

    # Create output directory
    output_dir = Path("intentus/example_outputs")
    output_dir.mkdir(exist_ok=True)

    # Run each task
    for i, task_info in enumerate(tasks, 1):
        logger.info(f"\n=== Running Task {i} ===")
        logger.info(f"Task: {task_info['task']}")
        logger.info(f"Context: {task_info['context']}")

        try:
            result = await agent.run(
                task=task_info["task"],
                context=task_info["context"],
            )

            # Print detailed results
            logger.info("\n=== Task Results ===")
            logger.info(f"Task: {result['task']}")
            logger.info(f"Analysis: {result['analysis']}")
            logger.info(f"Steps taken: {result['step_count']}")
            logger.info(f"Execution time: {result['execution_time']}s")
            logger.info(f"\nFinal output: {result['final_output']}")

            # Save results to file
            output_file = output_dir / f"task_{i}_result.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Error executing task {i}: {str(e)}", exc_info=True)

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(run_agent_demo())
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
