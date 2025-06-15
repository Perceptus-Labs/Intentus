# Intentus

An SDK for robotics interaction with audio and video processing capabilities.

## Installation

```bash
# Clone the repository
git clone https://github.com/Perceptus-Labs/Intentus.git
cd intentus

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

## Usage

Here's a basic example of how to use Intentus from the example script in the `intentus/examples` directory:

```python
import asyncio
from intentus.core.utils import setup_logging
from intentus.core.agent import AgentConfig, IntentusAgent


async def main():
    # Set up logging
    setup_logging()

    # Create agent configuration
    config = AgentConfig(
        llm_engine="gpt-4.1-mini",
        enabled_tools=["Wikipedia_Knowledge_Searcher_Tool"],
        verbose=True,
        max_steps=5,
        temperature=0.7,
    )

    # Create agent
    agent = IntentusAgent(config)

    # Run agent
    result = await agent.run(question="What is the capital of France?", image=None)

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
```

## Features

- Asynchronous execution
- Comprehensive logging
- Tool-based architecture
- Context-aware processing
- Configurable components

## Development

To run the example:

```bash
python intentus/examples/agent_demo.py
```

This will:
1. Create a new agent with the specified configuration
2. Run multiple example tasks
3. Save detailed logs to `example.log`
4. Save task results to JSON files in `example_outputs/`

## License

MIT License
