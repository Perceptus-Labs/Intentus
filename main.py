import asyncio
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn

from intentus.core.utils import setup_logging
from intentus.core.agent import AgentConfig, IntentusAgent

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Intentus Orchestrator", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Security
security = HTTPBearer()


# Pydantic models for request/response
class IntentionResult(BaseModel):
    session_id: str
    intention_type: str
    description: str
    confidence: float
    transcript: str
    environment_context: Optional[str] = None
    timestamp: int


class AgentResponse(BaseModel):
    success: bool
    query_analysis: Optional[str] = None
    base_response: Optional[str] = None
    final_output: Optional[str] = None
    execution_time: Optional[float] = None
    steps_taken: Optional[int] = None
    memory: Optional[list] = None
    error: Optional[str] = None


# Global agent instance
agent: Optional[IntentusAgent] = None


def get_agent() -> IntentusAgent:
    """Get or create the global agent instance"""
    global agent
    if agent is None:
        config = AgentConfig(
            llm_engine="gpt-4.1-mini",
            enabled_tools=["Wikipedia_Knowledge_Searcher_Tool"],
            verbose=True,
            max_steps=5,
            temperature=0.7,
        )
        agent = IntentusAgent(config)
    return agent


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """Verify the API key from the Authorization header"""
    expected_api_key = os.getenv("ORCHESTRATOR_API_KEY")
    if not expected_api_key:
        logger.warning("ORCHESTRATOR_API_KEY not set, skipping authentication")
        return True

    if credentials.credentials != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def format_context_for_agent(intention_result: IntentionResult) -> str:
    """Format the intention result into a text context for the agent"""
    context_parts = []

    # Add session info
    context_parts.append(f"Session ID: {intention_result.session_id}")

    # Add intention details
    context_parts.append(f"Intention Type: {intention_result.intention_type}")
    context_parts.append(f"Description: {intention_result.description}")
    context_parts.append(f"Confidence: {intention_result.confidence:.2f}")

    # Add transcript
    if intention_result.transcript:
        context_parts.append(f"Transcript: {intention_result.transcript}")

    # Add environment context if available
    if intention_result.environment_context:
        context_parts.append(
            f"Environment Context: {intention_result.environment_context}"
        )

    # Add timestamp
    timestamp_str = datetime.fromtimestamp(intention_result.timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    context_parts.append(f"Timestamp: {timestamp_str}")

    return "\n".join(context_parts)


@app.post("/orchestrate", response_model=AgentResponse)
async def orchestrate_intention(
    intention_result: IntentionResult, _: bool = Depends(verify_api_key)
):
    """
    Receive intention result from the other repo and process it with the Intentus agent
    """
    try:
        logger.info(
            f"Received intention request for session {intention_result.session_id}"
        )

        # Get the agent instance
        agent = get_agent()

        # Format the context for the agent
        context = format_context_for_agent(intention_result)

        # Create a question based on the intention
        question = f"Based on the following context, what should be done?\n\n{context}"

        logger.info(f"Running agent with question: {question[:100]}...")

        # Run the agent
        result = await agent.run(question=question, image="")

        logger.info(f"Agent completed successfully in {result['execution_time']:.2f}s")

        return AgentResponse(
            success=True,
            query_analysis=result["query_analysis"],
            base_response=result["base_response"],
            final_output=result["final_output"],
            execution_time=result["execution_time"],
            steps_taken=result["steps_taken"],
            memory=result["memory"],
        )

    except Exception as e:
        logger.error(f"Error processing intention: {str(e)}", exc_info=True)
        return AgentResponse(success=False, error=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "Intentus Orchestrator",
        "version": "1.0.0",
        "endpoints": {
            "POST /orchestrate": "Process intention results with Intentus agent",
            "GET /health": "Health check",
            "GET /": "This info",
        },
    }


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    port = int(os.getenv("ORCHESTRATOR_PORT", "8000"))

    logger.info(f"Starting Intentus Orchestrator on {host}:{port}")

    # Run the server
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")
