export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ENCRYPTION_WEIGHT=false
N_NPUS=4
torchrun --nproc_per_node=$N_NPUS --master_port $MASTER_PORT ../infer.py \
         --model Wan2.2-I2V-A14B \
         --pretrained_model_name_or_path /home/lcq/Wan2.2-I2V-A14B-Diffusers \
         --x \
         --x_model_path ../weights/6steps/transformer_i2v_14B_1/model.safetensors \
         --x_model_path_2 ../weights/6steps/transformer_i2v_14B_2/model.safetensors \
         --joint \
         --joint_model_path ../weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
         --task_type i2v \
         --i2v_image_path ./astronaut.jpg \
         --save_path ./output.mp4 \
         --num_inference_steps 6 \
         --width 1280 \
         --height 720 \
         --flow_shift 5.0 \
         --frames 81 \
         --sp $N_NPUS \
         --fsdp text_encoder \
         --atten_a8w8 \
         --matmul_a8w8 \
         --inf_vram_blocks_num 1 \
         --rope_fused \
         --guidance_scale 3.5 \
         --guidance_scale_2 3.5 \
         --seed 42 \
         --ada_brighten \
         --vae_lightning "decoder" \
         --prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot." \
         --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


