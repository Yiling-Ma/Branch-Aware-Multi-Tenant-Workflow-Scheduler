#!/bin/bash

# Script to start FastAPI server

echo "Starting FastAPI server..."

# 1. Clean up old processes
echo "Cleaning up old processes..."
pkill -f "uvicorn.*app.main" 2>/dev/null || true
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
sleep 2

# 2. Check port
if lsof -i :8000 > /dev/null 2>&1; then
    echo "Port 8000 is still in use, please check manually"
    exit 1
fi

# 3. Check Docker services
echo "Checking Docker services..."
cd "$(dirname "$0")"
if ! docker-compose ps db redis 2>/dev/null | grep -q "Up"; then
    echo "   Docker services not running, starting..."
    docker-compose up -d db redis
    echo "   Waiting for services to be ready..."
    sleep 5
fi

# Wait for database to be fully ready
echo "   Waiting for database to be ready..."
for i in {1..30}; do
    if docker-compose exec -T db pg_isready -U user -d scheduler_db > /dev/null 2>&1; then
        echo "   Database is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   Database startup timeout"
        exit 1
    fi
    sleep 1
done

# 4. Activate conda environment and start server
echo "Starting server..."
cd "$(dirname "$0")"

# Check if conda is available
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate penn
fi

# 5. Start scheduler (optional)
echo ""
echo "Starting scheduler..."
if pgrep -f "app.services.scheduler" > /dev/null; then
    echo "   Scheduler is already running"
else
    nohup python -m app.services.scheduler > scheduler.log 2>&1 &
    SCHEDULER_PID=$!
    echo "   Scheduler started (PID: $SCHEDULER_PID, log: scheduler.log)"
fi

# 6. Start Workers (default 4)
echo ""
WORKER_COUNT=${1:-4}
echo "Starting $WORKER_COUNT Workers..."
for i in $(seq 1 $WORKER_COUNT); do
    if pgrep -f "app.services.worker.*--worker-id $i" > /dev/null; then
        echo "   Worker-$i is already running"
    else
        nohup python -m app.services.worker --worker-id $i > worker_$i.log 2>&1 &
        WORKER_PID=$!
        echo "   Worker-$i started (PID: $WORKER_PID, log: worker_$i.log)"
        sleep 1
    fi
done

# 7. Start server (run in foreground for easy log viewing)
echo ""
echo "Starting FastAPI server..."
echo "   📝 Server logs will be displayed below..."
echo "   Access: http://127.0.0.1:8000/"
echo "   Press Ctrl+C to stop server"
echo ""
echo "Tip: Scheduler and Workers run in background"
echo "   View scheduler logs: tail -f scheduler.log"
echo "   View Worker logs: tail -f worker_1.log"
echo ""
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
