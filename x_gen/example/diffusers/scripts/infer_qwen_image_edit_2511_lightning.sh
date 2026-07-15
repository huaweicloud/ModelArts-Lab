export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
export INF_NAN_MODE_FORCE_DISABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0

# lora系列只支持N_NPUS=1
N_NPUS=1
HEIGHT=1024
WIDTH=1024
INF_VRAM_BLOCKS_NUM=0

# for Qwen-Image-Edit-2511-Lightning
torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer_image_gen.py \
    --model Qwen-Image-Edit-2511 \
    --pretrained_model_name_or_path ./models/Qwen-Image-Edit-2511 \
    --lora_path_list ./Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    --fuse_lora \
    --prompt "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square." \
    --negative_prompt " " \
    --image_path_list ./input1.png ./input2.png \
    --height $HEIGHT \
    --width $WIDTH \
    --seed 42 \
    --num_inference_steps 4 \
    --true_cfg_scale 1.0 \
    --guidance_scale 1.0 \
    --save_path ./output.png \
    --inf_vram_blocks_num $INF_VRAM_BLOCKS_NUM \
    --cfg_parallel_size $N_NPUS \
    --matmul_a8w8 \
    --atten_laser
