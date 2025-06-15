import asyncio
import logging
from intentus.core.agent import AgentConfig, IntentusAgent


async def main():
    # Create agent configuration
    config = AgentConfig(
        llm_engine="gpt-4.1-mini",
        enabled_tools=["Google_Search_Tool"],
        verbose=True,
        max_steps=5,
        temperature=0.7,
        max_tokens=4000,
    )

    # Create agent
    agent = IntentusAgent(config)

    # Run agent
    result = await agent.run(
        question="What is the current weather in New York?", image=None
    )

    # Print results
    print("\nQuery Analysis:")
    print(result["query_analysis"])
    print("\nBase Response:")
    print(result["base_response"])
    print("\nFinal Output:")
    print(result["final_output"])
    print("\nExecution Time:", result["execution_time"])
    print("Steps Taken:", result["steps_taken"])
    print("\nMemory:")
    for action in result["memory"]:
        print(f"- {action}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nDemo interrupted by user")
