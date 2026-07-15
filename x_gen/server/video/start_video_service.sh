#!/bin/bash

set -e

usage() {
    echo "Usage: $0 <model> <num_gpus> <port> [options]"
    echo ""
    echo "Arguments:"
    echo "  model      Model name: wan2.2-t2v-14b or wan2.2-i2v-14b"
    echo "  num_gpus   Number of GPUs (1, 2, 4, or 8)"
    echo "  port       Service port (e.g., 5001)"
    echo ""
    echo "Options:"
    echo "  --devices <ids>           Device IDs (default: auto-select, e.g., '0,1' for 2 GPUs)"
    echo "  --steps <n>               Default inference steps (default: 40 for single GPU, 20 for multi-GPU)"
    echo "  --ten-second              Enable 10-second video generation mode (requires 8 GPUs)"
    echo "  --no-init                 Skip pre-initialization"
    echo "  --warmup                  Run warmup inference on startup"
    echo "  --warmup-steps <n>        Warmup inference steps (default: 1)"
    echo ""
    echo "Model Paths:"
    echo "  --model-path <path>       Custom model path"
    echo "  --x-model-path <path>     Distilled model weights path 1"
    echo "  --x-model-path-2 <path>   Distilled model weights path 2"
    echo "  --frame-model-path <path> Frame interpolation model path"
    echo "  --seedvr2-model-dir <dir> SeedVR2 super-resolution model directory"
    echo "  --lora-path-list <paths>  LoRA weight paths (space-separated, wrap in quotes)"
    echo ""
    echo "Network:"
    echo "  --external-base-url <url> External base URL for download links"
    echo ""
    echo "Examples:"
    echo "  $0 wan2.2-t2v-14b 1 5001                    # Single GPU T2V on port 5001"
    echo "  $0 wan2.2-i2v-14b 1 5002 --devices 6       # Single GPU I2V on NPU 6"
    echo "  $0 wan2.2-t2v-14b 2 5003 --devices 0,1     # 2-GPU T2V on NPU 0,1"
    echo "  $0 wan2.2-t2v-14b 4 5004 --warmup          # 4-GPU T2V with warmup"
    echo "  $0 wan2.2-t2v-14b 8 5005 --ten-second      # 8-GPU T2V with 10s mode"
    echo "  $0 wan2.2-t2v-14b 4 5006 --model-path /path/to/model  # Custom model path"
    exit 1
}

if [ $# -lt 3 ]; then
    usage
fi

MODEL=$1
NUM_GPUS=$2
PORT=$3
shift 3

DEVICES=""
STEPS=""
DO_INIT="--init"
DO_WARMUP=""
WARMUP_STEPS=""
TEN_SECOND=""
MODEL_PATH=""
LORA_PATH_LIST=""
X_MODEL_PATH=""
X_MODEL_PATH_2=""
FRAME_MODEL_PATH=""
SEEDVR2_MODEL_DIR=""
EXTERNAL_BASE_URL_OPT=""
TURBO_MODE=""

while [ $# -gt 0 ]; do
    case $1 in
        --devices)
            DEVICES=$2
            shift 2
            ;;
        --steps)
            STEPS=$2
            shift 2
            ;;
        --ten-second)
            TEN_SECOND="--ten-second"
            shift
            ;;
        --x)
            USE_X="--x"
            shift
            ;;
        --no-init)
            DO_INIT=""
            shift
            ;;
        --warmup)
            DO_WARMUP="--warmup"
            shift
            ;;
        --warmup-steps)
            WARMUP_STEPS="--warmup-steps $2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="--model-path $2"
            shift 2
            ;;
        --lora-path-list)
            LORA_PATH_LIST="--lora-path-list $2"
            shift 2
            ;;
        --turbo_mode)
            TURBO_MODE="--turbo_mode $2"
            shift 2
            ;;
        --x-model-path)
            X_MODEL_PATH="--x-model-path $2"
            shift 2
            ;;
        --x-model-path-2)
            X_MODEL_PATH_2="--x-model-path-2 $2"
            shift 2
            ;;
        --frame-model-path)
            FRAME_MODEL_PATH="--frame-model-path $2"
            shift 2
            ;;
        --seedvr2-model-dir)
            SEEDVR2_MODEL_DIR="--seedvr2-model-dir $2"
            shift 2
            ;;
        --external-base-url)
            EXTERNAL_BASE_URL_OPT=$2
            export EXTERNAL_BASE_URL=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ "$MODEL" != "wan2.2-t2v-14b" && "$MODEL" != "wan2.2-i2v-14b" ]]; then
    echo "Error: Invalid model '$MODEL'. Must be 'wan2.2-t2v-14b' or 'wan2.2-i2v-14b'"
    exit 1
fi

if [[ ! "$NUM_GPUS" =~ ^(1|2|4|8)$ ]]; then
    echo "Error: NUM_GPUS must be 1, 2, 4, or 8"
    exit 1
fi

if [ -n "$TEN_SECOND" ] && [ "$NUM_GPUS" -lt 8 ]; then
    echo "Error: 10-second mode requires at least 8 GPUs"
    exit 1
fi

if [ -z "$DEVICES" ]; then
    case $NUM_GPUS in
        1) DEVICES="0" ;;
        2) DEVICES="0,1" ;;
        4) DEVICES="0,1,2,3" ;;
        8) DEVICES="0,1,2,3,4,5,6,7" ;;
    esac
fi

if [ -z "$STEPS" ]; then
    if [ "$NUM_GPUS" -eq 1 ]; then
        STEPS=40
    else
        STEPS=20
    fi
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + PORT - 5000))
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=1200
export ENCRYPTION_WEIGHT=false
export ASCEND_RT_VISIBLE_DEVICES=$DEVICES

echo "=========================================="
echo "Starting Video Generation Service"
echo "=========================================="
echo "  Model:      $MODEL"
echo "  GPUs:       $NUM_GPUS (devices: $DEVICES)"
echo "  Port:       $PORT"
echo "  Steps:      $STEPS"
echo "  Use X:      $([ -n '$USE_X' ] && echo 'Yes' || echo 'No')"
echo "  Init:       $([ -n '$DO_INIT' ] && echo 'Yes' || echo 'No')"
echo "  Warmup:     $([ -n '$DO_WARMUP' ] && echo 'Yes' || echo 'No')"
if [ -n "$TURBO_MODE" ]; then
    echo "  Turbo_mode: $(echo $TURBO_MODE | cut -d' ' -f2)"
fi
if [ -n "$MODEL_PATH" ]; then
    echo "  Model Path: $(echo $MODEL_PATH | cut -d' ' -f2)"
fi
if [ -n "$X_MODEL_PATH" ]; then
    echo "  X Model Path: $(echo $X_MODEL_PATH | cut -d' ' -f2)"
fi
if [ -n "$X_MODEL_PATH_2" ]; then
    echo "  X Model Path 2: $(echo $X_MODEL_PATH_2 | cut -d' ' -f2)"
fi
if [ -n "$FRAME_MODEL_PATH" ]; then
    echo "  Frame Model Path: $(echo $FRAME_MODEL_PATH | cut -d' ' -f2)"
fi
if [ -n "$SEEDVR2_MODEL_DIR" ]; then
    echo "  SeedVR2 Model Dir: $(echo $SEEDVR2_MODEL_DIR | cut -d' ' -f2)"
fi
if [ -n "$EXTERNAL_BASE_URL_OPT" ]; then
    echo "  External URL: $EXTERNAL_BASE_URL_OPT"
fi
echo "=========================================="
echo ""

if [ -n "$EXTERNAL_BASE_URL" ]; then
    export EXTERNAL_BASE_URL
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    python video_server.py \
        --model $MODEL \
        --sp 1 \
        --no-fsdp \
        $DO_INIT \
        $DO_WARMUP \
        $WARMUP_STEPS \
        $TEN_SECOND \
        $USE_X \
        $MODEL_PATH \
        $TURBO_MODE \
        $LORA_PATH_LIST \
        $X_MODEL_PATH \
        $X_MODEL_PATH_2 \
        $FRAME_MODEL_PATH \
        $SEEDVR2_MODEL_DIR \
        --port $PORT
else
    torchrun --nproc_per_node=$NUM_GPUS \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        video_server.py \
        --model $MODEL \
        --sp $NUM_GPUS \
        $DO_INIT \
        $DO_WARMUP \
        $WARMUP_STEPS \
        $TEN_SECOND \
        $USE_X \
        $MODEL_PATH \
        $TURBO_MODE \
        $LORA_PATH_LIST \
        $X_MODEL_PATH \
        $X_MODEL_PATH_2 \
        $FRAME_MODEL_PATH \
        $SEEDVR2_MODEL_DIR \
        --port $PORT
fi
