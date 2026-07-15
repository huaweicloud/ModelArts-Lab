export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_NPUS=8
torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer.py \
         --model Wan2.1-T2V-14B \
         --pretrained_model_name_or_path "../weights/Wan-AI/Wan2.1-T2V-14B-Diffusers" \
         --save_path ./output.mp4 \
         --num_inference_steps 50 \
         --width 832 \
         --height 480 \
         --frames 81 \
         --sp $N_NPUS \
         --fsdp \
         --vae_lightning "decoder" \
         --turbo_mode faiz \
         --atten_a8w8 \
         --matmul_a8w8 \
         --rope_fused \
         --flow_shift 7.0 \
         --seed 42 \
         --prompt "A young boy with short brown hair, dressed in a dark blue t-shirt and red pants, is seen playing a KAWAI upright piano with skill and concentration. The piano's glossy black surface reflects the room's lighting, and its white and black keys are arranged in a standard layout, indicating a scene of musical practice or learning. The boy's hands move over the keys, suggesting he is engaged in playing or practicing a piece." \
         --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
