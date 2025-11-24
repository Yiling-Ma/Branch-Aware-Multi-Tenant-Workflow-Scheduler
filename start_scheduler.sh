#!/bin/bash

# Start scheduler process

echo "🚀 Starting scheduler..."

# Ensure conda environment is activated
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate penn
fi

# Switch to project directory
cd "$(dirname "$0")"

# Check if scheduler is already running
if pgrep -f "app.scheduler" > /dev/null; then
    echo "⚠️  Scheduler is already running"
    exit 1
fi

# Start scheduler (run in background, output to log file)
nohup python -m app.scheduler > scheduler.log 2>&1 &
PID=$!

echo "✅ Scheduler started (PID: $PID)"
echo "📝 Log file: scheduler.log"
echo "   View logs: tail -f scheduler.log"
echo "   Stop scheduler: kill $PID"
