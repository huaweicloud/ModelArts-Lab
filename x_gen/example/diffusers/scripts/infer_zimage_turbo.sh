export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_RT_VISIBLE_DEVICES=0

N_NPUS=1
HEIGHT=1024
WIDTH=1024

# for Z-Image-Turbo
MODEL_NAME=Z-Image-Turbo
MODEL_PATH=/models/Z-Image-Turbo
SAVE_PATH=./zimage-turbo-output.png
NUM_INFERENCE_STEPS=8

python ../infer_image_gen.py \
  --model $MODEL_NAME \
  --pretrained_model_name_or_path $MODEL_PATH \
  --prompt "a cup of coffee on the table" \
  --negative_prompt "bad quality." \
  --seed 42 \
  --num_inference_steps $NUM_INFERENCE_STEPS \
  --guidance_scale 5.0 \
  --height $HEIGHT \
  --width $WIDTH \
  --save_path $SAVE_PATH \
  --atten_laser \
  --matmul_a8w8
