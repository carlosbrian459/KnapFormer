#!/bin/bash
# Experiment template script with ARG_ prefixed environment variables
# This template is instantiated by the Python experiment launcher


# Experiment parameters (to be replaced by Python, with defaults)
LOG_DIR="${ARG_LOG_DIR:-logs/default_experiment}"
DATA_CODES="${ARG_DATA_CODES:-g8b32i256f1s0}"
BALANCER_CONFIG="${ARG_BALANCER_CONFIG:-g1n8}"
D_MODEL="${ARG_D_MODEL:-3072}"
D_HEAD="${ARG_D_HEAD:-128}"
N_LAYERS="${ARG_N_LAYERS:-57}"
SHARD_SIZE="${ARG_SHARD_SIZE:-8}"
CAUSAL="${ARG_CAUSAL:-0}"
USE_FLUX="${ARG_USE_FLUX:-0}"
N_DS_LAYERS="${ARG_N_DS_LAYERS:-19}"
N_SS_LAYERS="${ARG_N_SS_LAYERS:-38}"
GAMMA_FILE="$LOG_DIR/workload_estimator_gamma.txt"

# Create log directory
mkdir -p "$LOG_DIR"


# Set distributed training variables
export NUM_NODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export NCCL_DEBUG="WARN"

# Run workload estimator (rank 0 only)
if [ $NODE_RANK -eq 0 ]; then
    python knapformer/workload_estimator.py "$GAMMA_FILE" \
        --d_model $D_MODEL --d_head $D_HEAD --causal $CAUSAL \
        2>&1 | tee "$LOG_DIR/workload_estimator.log"
fi

# Synchronize all nodes using DDP barrier
torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$((MASTER_PORT + 1)) \
    knapformer/utils/ddp_barrier.py "Waiting for workload estimator to complete"

GAMMA=$(cat "$GAMMA_FILE")

# Run simulator with balancer
torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_OF_GPUS --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    knapformer/simulator/simulate.py --data_codes "$DATA_CODES" --balancer_config "$BALANCER_CONFIG" \
    --gamma $GAMMA --d_model $D_MODEL --d_head $D_HEAD --n_layers $N_LAYERS \
    --shard_size $SHARD_SIZE --causal $CAUSAL --use_flux $USE_FLUX \
    --n_ds_layers $N_DS_LAYERS --n_ss_layers $N_SS_LAYERS \
    2>&1 | tee "$LOG_DIR/simulator_with_balancer_$NODE_RANK.log"

# Synchronize all nodes before running without balancer
torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$((MASTER_PORT + 1)) \
    knapformer/utils/ddp_barrier.py "Waiting for simulator with balancer to complete"

# Run simulator without balancer
torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_OF_GPUS --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    knapformer/simulator/simulate.py --data_codes "$DATA_CODES" --balancer_config "" \
    --gamma $GAMMA --d_model $D_MODEL --d_head $D_HEAD --n_layers $N_LAYERS \
    --shard_size $SHARD_SIZE --causal $CAUSAL --use_flux $USE_FLUX \
    --n_ds_layers $N_DS_LAYERS --n_ss_layers $N_SS_LAYERS \
    2>&1 | tee "$LOG_DIR/simulator_without_balancer_$NODE_RANK.log"

# Synchronize all nodes after running without balancer
torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$((MASTER_PORT + 1)) \
    knapformer/utils/ddp_barrier.py "Waiting for simulator without balancer to complete"
