#!/bin/bash

# Stop all services: Scheduler + Workers

echo "🛑 Stopping all services..."

# Stop scheduler
if pgrep -f "app.scheduler" > /dev/null; then
    pkill -f "app.scheduler"
    echo "✅ Scheduler stopped"
else
    echo "ℹ️  Scheduler is not running"
fi

# Stop all Workers
if pgrep -f "app.worker" > /dev/null; then
    pkill -f "app.worker"
    echo "✅ All Workers stopped"
else
    echo "ℹ️  Workers are not running"
fi

echo ""
echo "✅ All services stopped"
