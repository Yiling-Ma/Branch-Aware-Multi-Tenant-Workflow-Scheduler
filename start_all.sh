#!/bin/bash

# One-click startup script for all services: Scheduler + Workers

echo "🚀 Starting all services..."
echo ""

# Ensure conda environment is activated
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate penn
fi

# Switch to project directory
cd "$(dirname "$0")"

# 1. Check Docker services
echo "🐳 Checking Docker services..."
if ! docker-compose ps db redis 2>/dev/null | grep -q "Up"; then
    echo "   ⚠️  Docker services not running, starting..."
    docker-compose up -d db redis
    echo "   ⏳ Waiting for services to be ready..."
    sleep 5
fi

# Wait for database to be ready
for i in {1..30}; do
    if docker-compose exec -T db pg_isready -U user -d scheduler_db > /dev/null 2>&1; then
        echo "   ✅ Database is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ Database startup timeout"
        exit 1
    fi
    sleep 1
done

# 2. Start scheduler
echo ""
echo "📅 Starting scheduler..."
if pgrep -f "app.scheduler" > /dev/null; then
    echo "   ⚠️  Scheduler is already running"
else
    nohup python -m app.scheduler > scheduler.log 2>&1 &
    SCHEDULER_PID=$!
    echo "   ✅ Scheduler started (PID: $SCHEDULER_PID, log: scheduler.log)"
fi

# 3. Start Workers
echo ""
WORKER_COUNT=${1:-4}
echo "👷 Starting $WORKER_COUNT Workers..."
for i in $(seq 1 $WORKER_COUNT); do
    if pgrep -f "app.worker.*--worker-id $i" > /dev/null; then
        echo "   ⚠️  Worker-$i is already running"
    else
        nohup python -m app.worker --worker-id $i > worker_$i.log 2>&1 &
        WORKER_PID=$!
        echo "   ✅ Worker-$i started (PID: $WORKER_PID, log: worker_$i.log)"
        sleep 1
    fi
done

echo ""
echo "✅ All services started!"
echo ""
echo "📊 Check running status:"
echo "   Scheduler: ps aux | grep app.scheduler"
echo "   Workers: ps aux | grep app.worker"
echo ""
echo "📝 View logs:"
echo "   Scheduler: tail -f scheduler.log"
echo "   Workers: tail -f worker_1.log"
echo ""
echo "🛑 Stop all services:"
echo "   ./stop_all.sh"
