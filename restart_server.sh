#!/bin/bash

# Script to restart FastAPI server

echo "🔄 Restarting FastAPI server..."

# 1. Stop existing uvicorn processes
echo "⏹️  Stopping existing server..."
pkill -f "uvicorn app.main:app" || echo "   (No running server found)"

# Wait for processes to fully stop
sleep 2

# 2. Check if port is released
if lsof -i :8000 > /dev/null 2>&1; then
    echo "⚠️  Port 8000 is still in use, attempting to force release..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# 3. Check Docker services
echo "🐳 Checking Docker services..."
cd "$(dirname "$0")"
if ! docker-compose ps db redis 2>/dev/null | grep -q "Up"; then
    echo "   ⚠️  Docker services not running, starting..."
    docker-compose up -d db redis
    echo "   ⏳ Waiting for services to be ready..."
    sleep 5
fi

# 4. Start server
echo "🚀 Starting FastAPI server..."
cd "$(dirname "$0")"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 &

# Wait for server to start
sleep 3

# 5. Check server status
if curl -s http://127.0.0.1:8000/ > /dev/null 2>&1; then
    echo "✅ Server started successfully!"
    echo "   🌐 Access: http://127.0.0.1:8000/"
else
    echo "❌ Server failed to start, please check logs"
    echo "   💡 Tip: Check terminal output or run 'tail -f logs/*.log'"
fi
