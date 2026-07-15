# Wan2.1 VACE openpose task
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
         --vace_task openpose \
         --video_path ./openpose-man-contemporary-dance.mp4 \
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
         --prompt "An alien-like creature with a resemblance of leaves, branches and twigs is dancing gracefully in a post-apocalyptic world. The creature has a humanoid shape, with long, flowing limbs that resemble branches. Its skin is textured like bark, and its eyes glow softly. The background is a desolate landscape with remnants of a once-thriving city, now overgrown with vegetation. The lighting is soft and ethereal, casting a magical glow on the scene. The camera captures the creature from a low angle, emphasizing its height and gracefulness as it moves fluidly through the air." \
         --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
