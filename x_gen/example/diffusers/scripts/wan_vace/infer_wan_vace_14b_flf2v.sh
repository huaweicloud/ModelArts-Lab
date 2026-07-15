# Wan2.1 VACE first last frame to video task
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_NPUS=8
torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../../infer.py \
         --model Wan2.1-VACE-14B \
         --pretrained_model_name_or_path ../weights/Wan2.1-VACE-14B-diffusers \
         --vace_task flf2v \
         --first_frame_path ./flf2v_input_first_frame.png \
         --last_frame_path ./flf2v_input_last_frame.png \
         --save_path ./output.mp4 \
         --num_inference_steps 50 \
         --width 832 \
         --height 480 \
         --frames 81 \
         --sp $N_NPUS \
         --phaa \
         --fsdp \
         --vae_lightning "decoder" \
         --turbo_mode next_faiz \
         --matmul_a8w8 \
         --atten_a8w8 \
         --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective." \
         --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
