#!/bin/bash

# Benchmark automation script for LMCache testing
# This script runs benchmarks across multiple QPS values in tmux

set -e

# Configuration
TMUX_SESSION="kobe"
BASE_DIR="/home/ubuntu/kobe/LMCache/output-llama70b"
# QPS_VALUES=(1.4 1.7 2.0 2.3 2.6 2.9)
QPS_VALUES=(2.0 3.2 3.5 3.8)
DATASET_PATH="/home/ubuntu/kobe/LMCache/conversation_qps_6.jsonl"

# Tmux window indices (you have 5 separate windows numbered 1-5)
PANE_CONTROL=1  # Terminal 1 - control/metrics (window 1)
PANE_LMCACHE=2  # Terminal 2 - lmcache server (window 2)
PANE_BENCH1=3   # Terminal 3 - benchmark 1 (window 3)
PANE_VLLM=4     # Terminal 4 - vllm server (window 4)
PANE_BENCH2=5   # Terminal 5 - benchmark 2 (window 5)

# Function to send command to tmux window
send_to_pane() {
    local window=$1
    local command=$2
    tmux send-keys -t "${TMUX_SESSION}:${window}" "$command" C-m
}

# Function to convert QPS to folder name
qps_to_folder() {
    local qps=$1
    echo "qps_$(echo $qps | tr '.' '_')"
}

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local max_wait=${2:-1800}  # Default 30 minutes
    local elapsed=0
    
    echo "Waiting for server on port $port to be ready..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s http://0.0.0.0:$port/health > /dev/null 2>&1; then
            echo "Server on port $port is ready!"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  Waiting... ${elapsed}s elapsed"
    done
    
    echo "WARNING: Server on port $port did not become ready within ${max_wait}s"
    return 1
}

# Function to check if process is running on port
is_port_busy() {
    local port=$1
    netstat -tuln 2>/dev/null | grep -q ":${port} " || \
    ss -tuln 2>/dev/null | grep -q ":${port} "
}

# Function to kill servers
kill_servers() {
    echo "Killing servers on ports 8020 and 8030..."
    
    # Kill processes on port 8020
    if is_port_busy 8020; then
        lsof -ti:8020 | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Kill processes on port 8030
    if is_port_busy 8030; then
        lsof -ti:8030 | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Also send Ctrl+C to the panes to clean up
    send_to_pane $PANE_LMCACHE "C-c"
    send_to_pane $PANE_VLLM "C-c"
    sleep 2
}

# Main loop
for qps in "${QPS_VALUES[@]}"; do
    echo "========================================="
    echo "Starting benchmark for QPS: $qps"
    echo "========================================="
    
    folder_name=$(qps_to_folder $qps)
    folder_path="${BASE_DIR}/${folder_name}"
    
    # Check if folder exists
    if [ ! -d "$folder_path" ]; then
        echo "ERROR: Folder $folder_path does not exist!"
        echo "Creating folder..."
        mkdir -p "$folder_path"
    fi
    
    # Change to the directory in terminal 1
    send_to_pane $PANE_CONTROL "cd $folder_path"
    sleep 1
    
    # Start LMCache server in terminal 2
    # COMMENTED OUT - GPUs 0-3 are occupied
    # echo "Starting LMCache server on port 8020..."
    # send_to_pane $PANE_LMCACHE "cd /home/ubuntu/kobe/LMCache"
    # sleep 1
    # send_to_pane $PANE_LMCACHE "export PYTHONHASHSEED=0"
    # sleep 0.5
    # send_to_pane $PANE_LMCACHE "CUDA_VISIBLE_DEVICES=0,1,2,3 LMCACHE_CONFIG_FILE=\"example.yaml\" LMCACHE_USE_EXPERIMENTAL=True vllm serve meta-llama/Llama-3.1-70B --port 8020 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}' -tp 4 2>&1 | tee ${folder_path}/lmcache_server.log"
    
    # Start VLLM server in terminal 4
    echo "Starting VLLM server on port 8030..."
    send_to_pane $PANE_VLLM "cd /home/ubuntu/kobe/LMCache"
    sleep 1
    send_to_pane $PANE_VLLM "export PYTHONHASHSEED=0"
    sleep 0.5
    send_to_pane $PANE_VLLM "CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve meta-llama/Llama-3.1-70B --port 8030 -tp 4 2>&1 | tee ${folder_path}/vllm_server.log"
    
    # Wait for server to be ready
    echo "Waiting for VLLM server to start (this may take ~5 minutes)..."
    sleep 30  # Give server time to start initialization
    
    # wait_for_server 8020 1800 &
    # PID_WAIT_8020=$!
    wait_for_server 8030 1800 &
    PID_WAIT_8030=$!
    
    # wait $PID_WAIT_8020
    wait $PID_WAIT_8030
    
    echo "VLLM server is ready. Starting benchmark..."
    sleep 5
    
    # Run benchmark on LMCache server (terminal 3)
    # COMMENTED OUT - GPUs 0-3 are occupied
    # echo "Starting benchmark on LMCache server (port 8020)..."
    # send_to_pane $PANE_BENCH1 "cd $folder_path"
    # sleep 1
    # send_to_pane $PANE_BENCH1 "vllm bench serve --base-url http://0.0.0.0:8020 --backend openai --model meta-llama/Llama-3.1-70B --endpoint /v1/completions --dataset-name custom --dataset-path ${DATASET_PATH} --skip-chat-template --num-prompts 10000 --request-rate ${qps} --temperature=0 --disable-shuffle --custom-output-len 200"
    
    # Run benchmark on VLLM server (terminal 5)
    echo "Starting benchmark on VLLM server (port 8030)..."
    send_to_pane $PANE_BENCH2 "cd $folder_path"
    sleep 1
    send_to_pane $PANE_BENCH2 "vllm bench serve --base-url http://0.0.0.0:8030 --backend openai --model meta-llama/Llama-3.1-70B --endpoint /v1/completions --dataset-name custom --dataset-path ${DATASET_PATH} --skip-chat-template --num-prompts 10000 --request-rate ${qps} --temperature=0 --disable-shuffle --custom-output-len 200"
    
    # Wait for benchmark to complete
    # With 10000 prompts at QPS rates, this will take time
    # Estimate: 10000 / qps seconds + processing overhead
    estimated_time=$(echo "scale=0; 10000 / $qps + 300" | bc)
    echo "Benchmark started. Estimated completion time: ~${estimated_time} seconds"
    echo "Waiting for benchmark to complete..."
    
    # Poll to check if benchmarks are done
    # We'll wait for a reasonable amount of time based on QPS
    sleep $estimated_time
    # sleep 60
    
    # Give some extra time for final processing
    echo "Waiting additional time for final processing..."
    sleep 60
    
    # Collect metrics
    echo "Collecting metrics..."
    # send_to_pane $PANE_CONTROL "curl http://0.0.0.0:8020/metrics > lmcache_metrics_output.log"
    # sleep 2
    send_to_pane $PANE_CONTROL "curl http://0.0.0.0:8030/metrics > vllm_metrics_output.log"
    sleep 2
    
    # Kill servers
    kill_servers
    
    # Wait a bit before next iteration
    echo "Waiting before next QPS iteration..."
    sleep 10
    
    echo "Completed benchmark for QPS: $qps"
    echo ""
done

echo "========================================="
echo "All benchmarks completed!"
echo "========================================="


