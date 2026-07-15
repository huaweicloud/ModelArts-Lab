export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_RT_VISIBLE_DEVICES=0,1

N_NPUS=2
CFG_SIZE=$N_NPUS
HEIGHT=1024
WIDTH=1024
MODEL_PATH=/models/LongCat-Image
SAVE_PATH=./longcat-image-output.png


# LongCat-Image
torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer_image_gen.py \
  --model LongCat-Image \
  --pretrained_model_name_or_path $MODEL_PATH \
  --prompt "a cup of tea on the table" \
  --negative_prompt "bad quality." \
  --seed 42 \
  --num_inference_steps 30 \
  --guidance_scale 4.5 \
  --height $HEIGHT \
  --width $WIDTH \
  --save_path $SAVE_PATH \
  --atten_laser \
  --matmul_a8w8 \
  --cfg_parallel_size $CFG_SIZE \
  --cache_dit
