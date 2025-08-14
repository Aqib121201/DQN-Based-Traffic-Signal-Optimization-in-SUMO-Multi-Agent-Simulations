#!/bin/bash

# Entrypoint script for DQN Traffic Signal Control Docker container

set -e

echo "🚦 Starting DQN Traffic Signal Control Application"

# Check if SUMO is properly installed
if ! command -v sumo &> /dev/null; then
    echo "❌ SUMO is not installed or not in PATH"
    exit 1
fi

echo "✅ SUMO found: $(sumo --version | head -n 1)"

# Check if Python dependencies are installed
if ! python -c "import tensorflow, traci, streamlit" &> /dev/null; then
    echo "❌ Required Python packages are not installed"
    exit 1
fi

echo "✅ Python dependencies verified"

# Create necessary directories if they don't exist
mkdir -p /app/data/raw /app/data/processed /app/data/external \
         /app/models /app/visualizations /app/logs

# Set proper permissions
chmod -R 755 /app

# Function to handle graceful shutdown
cleanup() {
    echo "🛑 Shutting down gracefully..."
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Check if we should run the training pipeline
if [ "$RUN_TRAINING" = "true" ]; then
    echo "🏋️ Starting training pipeline..."
    cd /app
    python run_pipeline.py --mode train --episodes ${TRAINING_EPISODES:-1000} --max_steps ${MAX_STEPS:-1000}
fi

# Check if we should run evaluation
if [ "$RUN_EVALUATION" = "true" ]; then
    echo "📊 Starting evaluation..."
    cd /app
    python run_pipeline.py --mode evaluate --model_path ${MODEL_PATH:-models/dqn_traffic_model.pkl} --episodes ${EVAL_EPISODES:-10}
fi

# Default: run Streamlit app
if [ "$RUN_STREAMLIT" != "false" ]; then
    echo "🌐 Starting Streamlit web application..."
    cd /app
    
    # Set Streamlit configuration
    export STREAMLIT_SERVER_PORT=${STREAMLIT_PORT:-8501}
    export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_ADDRESS:-0.0.0.0}
    export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_HEADLESS:-true}
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    
    echo "📍 Streamlit will be available at: http://localhost:${STREAMLIT_SERVER_PORT}"
    
    # Start Streamlit
    exec streamlit run app/app.py \
        --server.port=${STREAMLIT_SERVER_PORT} \
        --server.address=${STREAMLIT_SERVER_ADDRESS} \
        --server.headless=${STREAMLIT_HEADLESS} \
        --browser.gatherUsageStats=false
fi

# If no specific mode is set, just keep the container running
echo "⏳ Container is running. Use docker exec to run commands."
while true; do
    sleep 3600
done
