#!/bin/bash

# Intentus Orchestrator Docker Deployment Script

set -e

echo "🐳 Building Intentus Orchestrator Docker image..."

# Build the Docker image
docker build -t intentus-orchestrator .

echo "✅ Docker image built successfully!"

# Check if ORCHESTRATOR_API_KEY is set
if [ -z "$ORCHESTRATOR_API_KEY" ]; then
    echo "⚠️  Warning: ORCHESTRATOR_API_KEY not set. Using default key."
    echo "   Set it with: export ORCHESTRATOR_API_KEY='your-secret-key'"
fi

echo "🚀 Starting Intentus Orchestrator container..."

# Run the container
docker run -d \
    --name intentus-orchestrator \
    -p 8000:8000 \
    -e ORCHESTRATOR_API_KEY=${ORCHESTRATOR_API_KEY:-your-secret-api-key} \
    -e ORCHESTRATOR_HOST=0.0.0.0 \
    -e ORCHESTRATOR_PORT=8000 \
    --restart unless-stopped \
    intentus-orchestrator

echo "✅ Intentus Orchestrator is running!"
echo "📍 API available at: http://localhost:8000"
echo "🔑 API Key: ${ORCHESTRATOR_API_KEY:-your-secret-api-key}"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker logs intentus-orchestrator"
echo "   Stop: docker stop intentus-orchestrator"
echo "   Remove: docker rm intentus-orchestrator"
echo "   Health check: curl http://localhost:8000/health" 