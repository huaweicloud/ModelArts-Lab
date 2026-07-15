# Wan2.1 VACE random to video task
# You should use similar looking images for consistent video generation.
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
         --vace_task random2v \
         --image_path_list "./1.png" "./2.png" "./3.png" \
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
         --prompt "Various different characters appear and disappear in a fast transition video showcasting their unique features and personalities. The video is about showcasing different dance styles, with each character performing a distinct dance move. The background is a vibrant, colorful stage with dynamic lighting that changes with each dance style. The camera captures close-ups of the dancers' expressions and movements. Highly dynamic, fast-paced music video, with quick cuts and transitions between characters, cinematic, vibrant colors." \
         --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
