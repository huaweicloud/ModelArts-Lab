export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29509

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
export ENCRYPTION_WEIGHT=false
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_NPUS=8

torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer.py \
         --model Wan2.2-I2V-A14B \
         --pretrained_model_name_or_path ../weights/Wan-AI/Wan2.2-I2V-A14B-Diffusers \
         --task_type i2v \
         --i2v_image_path ./astronaut.jpg \
         --save_path ./output.mp4 \
         --num_inference_steps 6 \
         --width 832 \
         --height 480 \
         --frames 81 \
         --sp $N_NPUS \
         --fsdp text_encoder \
         --inf_vram_blocks_num 1 \
         --rope_fused \
         --vae_lightning "decoder" \
         --atten_a8w8 \
         --matmul_a8w8 \
         --x \
         --joint \
         --joint_model_path ../weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
         --ten_second \
         --ten_second_model_id_t2v ../weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
         --ten_second_model_path ../weights/ttv_14B_4steps/transformer_14B/model.safetensors \
         --ten_second_model_path_2 ../weights/ttv_14B_4steps/transformer_14B_2/model.safetensors \
         --pusa_lora ../weights/pusa_lora_converted/pytorch_lora_weights_high.safetensors \
         --pusa_lora2 ../weights/pusa_lora_converted/pytorch_lora_weights_low.safetensors \
         --x_model_path ../weights/6steps/transformer_i2v_14B_1/model.safetensors \
         --x_model_path_2 ../weights/6steps/transformer_i2v_14B_2/model.safetensors \
         --seedvr2_model_dir ../weights/SeedVR2/ \
         --seedvr2_model_name seedvr2_ema_7b_fp16.safetensors \
         --adopt_sr \
         --resolution 1080 \
         --guidance_scale 1.0 \
         --guidance_scale_2 1.0 \
         --seed 42 \
         --prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot." \
         --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

