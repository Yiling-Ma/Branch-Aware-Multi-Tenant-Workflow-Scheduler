#!/bin/bash

# Start multiple Worker processes for parallel execution
# Usage: ./start_workers.sh [worker_count]
# Default: start 4 workers (corresponds to MAX_WORKERS=4)

WORKER_COUNT=${1:-4}

echo "🚀 Starting $WORKER_COUNT Worker processes..."
echo "   Note: Each Worker runs independently and can process jobs from different branches in parallel"
echo ""

# Ensure conda environment is activated
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate penn
fi

# Switch to project directory
cd "$(dirname "$0")"

# Start multiple worker processes
for i in $(seq 1 $WORKER_COUNT); do
    echo "👷 Starting Worker-$i..."
    nohup python -m app.worker --worker-id $i > worker_$i.log 2>&1 &
    PID=$!
    echo "   Worker-$i started (PID: $PID, log: worker_$i.log)"
    sleep 1  # Avoid resource competition from simultaneous startup
done

echo ""
echo "✅ Started $WORKER_COUNT Worker processes"
echo "   Check running status: 'ps aux | grep app.worker'"
echo "   Stop all Workers: 'pkill -f app.worker'"
