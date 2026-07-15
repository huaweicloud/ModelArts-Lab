#!/bin/bash

# 启动图像生成服务
# 用法:
#   bash start_server.sh [single|multi] [host] [port] [warmup] [额外参数...]


MODE=${1:-"single"}
HOST=${2:-"0.0.0.0"}
PORT=${3:-5000}
WARMUP=${4:-"false"}

shift 4 2>/dev/null || shift $# 2>/dev/null
EXTRA_ARGS="$@"

WARMUP_ARGS=""
if [ "$WARMUP" == "true" ] || [ "$WARMUP" == "yes" ] || [ "$WARMUP" == "1" ]; then
    WARMUP_ARGS="--warmup --warmup-width 512 --warmup-height 512 --warmup-steps 10"
    echo "Warmup enabled: 512x512, 10 steps"
fi

if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args: $EXTRA_ARGS"
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export INF_NAN_MODE_FORCE_DISABLE=1


if [ "$MODE" == "multi" ]; then
    echo "启动多卡模式 (卡0、卡1，双卡并行)"
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    N_NPUS=2

    torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT image_server.py \
        --host $HOST \
        --port $PORT \
        --config config.json \
        --init \
        --cfg-parallel-size $N_NPUS \
        $WARMUP_ARGS \
        $EXTRA_ARGS
else
    echo "启动单卡模式 (卡0)"
    export ASCEND_RT_VISIBLE_DEVICES=0

    python image_server.py \
        --host $HOST \
        --port $PORT \
        --config config.json \
        --init \
        --cfg-parallel-size 1 \
        $WARMUP_ARGS \
        $EXTRA_ARGS
fi
