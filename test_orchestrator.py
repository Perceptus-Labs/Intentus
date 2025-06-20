#!/usr/bin/env python3
"""
Test script for the Intentus Orchestrator API
"""
import requests
import json
import time
from datetime import datetime

# Configuration
ORCHESTRATOR_URL = "http://localhost:8000/orchestrate"
API_KEY = "test-api-key"  # Set this to match your ORCHESTRATOR_API_KEY


def test_orchestrator():
    """Test the orchestrator endpoint with a sample intention result"""

    # Sample intention result matching the Go struct
    sample_intention = {
        "session_id": "test-session-123",
        "intention_type": "user_query",
        "description": "User asked about the weather",
        "confidence": 0.95,
        "transcript": "What's the weather like today?",
        "environment_context": "User is in San Francisco, CA",
        "timestamp": int(datetime.now().timestamp()),
    }

    print("Testing Intentus Orchestrator API")
    print("=" * 50)
    print(f"URL: {ORCHESTRATOR_URL}")
    print(f"Payload: {json.dumps(sample_intention, indent=2)}")
    print()

    try:
        # Make the request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        response = requests.post(
            ORCHESTRATOR_URL, json=sample_intention, headers=headers, timeout=30
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print()

        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Success: {result.get('success')}")
            print(f"Execution Time: {result.get('execution_time', 'N/A')}s")
            print(f"Steps Taken: {result.get('steps_taken', 'N/A')}")
            print()
            print("Final Output:")
            print(result.get("final_output", "No output"))
            print()
            print("Memory:")
            memory = result.get("memory", [])
            for i, action in enumerate(memory, 1):
                print(f"  {i}. {action}")
        else:
            print("❌ Error!")
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the orchestrator server is running")
        print("Run: python main.py")
    except requests.exceptions.Timeout:
        print("❌ Timeout: Request took too long")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(response.json())
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Health check error: {e}")


if __name__ == "__main__":
    print("Intentus Orchestrator Test")
    print("=" * 30)
    print()

    # Test health first
    test_health()
    print()

    # Test main endpoint
    test_orchestrator()
