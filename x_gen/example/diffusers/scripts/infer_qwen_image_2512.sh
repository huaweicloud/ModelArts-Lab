export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
export INF_NAN_MODE_FORCE_DISABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1

# for Qwen-Image-2512
N_NPUS=2
HEIGHT=1024
WIDTH=1024
INF_VRAM_BLOCKS_NUM=0

torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer_image_gen.py \
    --model Qwen-Image-2512 \
    --pretrained_model_name_or_path ./models/Qwen-Image-2512 \
    --prompt '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --negative_prompt " " \
    --height $HEIGHT \
    --width $WIDTH \
    --seed 42 \
    --num_inference_steps 20 \
    --true_cfg_scale 4.0 \
    --save_path ./output.png \
    --inf_vram_blocks_num $INF_VRAM_BLOCKS_NUM \
    --cfg_parallel_size $N_NPUS \
    --matmul_a8w8 \
    --atten_laser \
    --cache_dit
