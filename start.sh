#!/usr/bin/env bash

# Exit if any command fails
set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "=========================================="
echo "🚀 Starting KG-RAG Chat System..."
echo "=========================================="

echo "Starting FastAPI Backend..."
cd "$BACKEND_DIR"
# Activate the virtual environment
source venv/bin/activate
# Install requirements just in case
pip install -r requirements.txt > /dev/null 2>&1

# Start the uvicorn server in the background (using standard asyncio loop to avoid nest-asyncio uvloop errors)
uvicorn api:app --host 0.0.0.0 --port 8000 --loop asyncio &
BACKEND_PID=$!
echo "✓ Backend running on http://localhost:8000 (PID: $BACKEND_PID)"

echo "Starting Angular Frontend..."
cd "$FRONTEND_DIR"
# Start the angular preview server
npx ng serve &
FRONTEND_PID=$!
echo "✓ Frontend starting on http://localhost:4200 (PID: $FRONTEND_PID)"

echo "=========================================="
echo "Press Ctrl+C to stop all services"
echo "=========================================="

# Wait for process to exit
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
