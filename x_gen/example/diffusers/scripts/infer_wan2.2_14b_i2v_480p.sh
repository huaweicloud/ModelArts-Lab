export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export MEMORY_FRAGMENTATION=1
export ENCRYPTION_WEIGHT=false
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/turing_ascend_cloud/op_api/lib/:${LD_LIBRARY_PATH}

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_NPUS=8

export TASK_TYPE="i2v"
export TASK_NAME="astronaut"
export VAE_LIGHTING="encoder_and_decoder"
export TURBO_MODE="next_faiz"
export FSDP_MODE="all"

torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer.py \
         --model Wan2.2-I2V-A14B \
         --pretrained_model_name_or_path ../weights/Wan-AI/Wan2.2-I2V-A14B-Diffusers \
         --task_type $TASK_TYPE \
         --i2v_image_path ./${TASK_NAME}.jpg \
         --save_path ./${TASK_NAME}_480P_${TASK_TYPE}.mp4 \
         --num_inference_steps 40 \
         --width 832 \
         --height 480 \
         --frames 81 \
         --guidance_scale 3.5 \
         --guidance_scale_2 3.5 \
         --seed 42 \
         --prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot." \
         --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
         --sp $N_NPUS \
         --atten_laser \
         --matmul_a8w8 \
         --rope_fused \
         --turbo_mode $TURBO_MODE \
         --vae_lightning $VAE_LIGHTING \
         --fsdp $FSDP_MODE
