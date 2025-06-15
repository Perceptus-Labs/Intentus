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

Here's a basic example of how to use Intentus:

```python
import asyncio
from intentus.core.agent import CoreConfig, IntentusAgent

async def main():
    # Create agent configuration
    config = CoreConfig(
        llm_engine="gpt-4.1-mini",
        enabled_tools=["web_search", "calculator"],
        max_steps=5,
        max_time=60,
        verbose=True
    )
    
    # Initialize agent
    agent = IntentusAgent(config)
    
    # Run a task
    result = await agent.run(
        task="What is the population of Tokyo?",
        context={
            "require_accuracy": True,
            "preferred_currency": "USD"
        }
    )
    
    print(f"Final output: {result['final_output']}")

if __name__ == "__main__":
    asyncio.run(main())
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
python examples/agent_demo.py
```

This will:
1. Create a new agent with the specified configuration
2. Run multiple example tasks
3. Save detailed logs to `example.log`
4. Save task results to JSON files in `example_outputs/`

## License

MIT License
