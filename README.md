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

# Intentus Orchestrator API

This orchestrator API receives intention results from your Go application and processes them using the Intentus agent.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export ORCHESTRATOR_API_KEY="your-secret-api-key"
   export ORCHESTRATOR_HOST="0.0.0.0"  # Optional, defaults to 0.0.0.0
   export ORCHESTRATOR_PORT="8000"     # Optional, defaults to 8000
   ```

3. **Run the orchestrator:**
   ```bash
   python main.py
   ```

The server will start on `http://localhost:8000` (or your configured host/port).

## API Endpoints

### POST /orchestrate
Main endpoint that receives intention results and processes them with the Intentus agent.

**Request Body:**
```json
{
  "session_id": "session-123",
  "intention_type": "user_query",
  "description": "User asked about the weather",
  "confidence": 0.95,
  "transcript": "What's the weather like today?",
  "environment_context": "User is in San Francisco, CA",
  "timestamp": 1703123456
}
```

**Headers:**
```
Content-Type: application/json
Authorization: Bearer your-secret-api-key
```

**Response:**
```json
{
  "success": true,
  "query_analysis": "Analysis of the query...",
  "base_response": "Base response from agent...",
  "final_output": "Final processed output...",
  "execution_time": 2.5,
  "steps_taken": 3,
  "memory": ["Step 1: ...", "Step 2: ..."]
}
```

### GET /health
Health check endpoint.

### GET /
Root endpoint with API information.

## Testing

Run the test script to verify everything works:

```bash
python test_orchestrator.py
```

Make sure to update the `API_KEY` in `test_orchestrator.py` to match your `ORCHESTRATOR_API_KEY`.

## Integration with Go Application

Your Go application should send requests to the `/orchestrate` endpoint with the same payload structure as shown in the request body example above.

The orchestrator will:
1. Receive the intention result
2. Format it into a context string for the agent
3. Run the Intentus agent with the formatted context
4. Return the agent's response

## Configuration

The agent configuration is set in the `get_agent()` function in `main.py`. You can modify:
- `llm_engine`: The LLM engine to use
- `enabled_tools`: List of tools to enable
- `max_steps`: Maximum steps for the agent
- `temperature`: Temperature for LLM responses

## Logging

The orchestrator uses the same logging setup as the Intentus agent. Logs will show:
- Incoming requests
- Agent execution details
- Errors and exceptions

## Security

- API key authentication is required for the `/orchestrate` endpoint
- Set `ORCHESTRATOR_API_KEY` environment variable
- If not set, authentication is skipped (not recommended for production) 

## Local Agent Demo

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

