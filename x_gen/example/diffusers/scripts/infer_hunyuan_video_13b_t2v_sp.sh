export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29504

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_NPUS=8
torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer.py \
         --model HunyuanVideo-T2V-13B \
         --pretrained_model_name_or_path "../weights/HunyuanVideo" \
         --task_type t2v \
         --save_path ./output.mp4 \
         --width 544 \
         --height 960 \
         --frames 129 \
         --sp 8 \
         --guidance_scale 6.0 \
         --num_inference_steps 50 \
         --seed 42 \
         --fsdp \
         --turbo_mode next_faiz \
         --prompt "An astronaut floating in space, but instead of stars, there are massive, glowing jellyfish drifting through the void. Their tentacles ripple with shifting, colorful lights, creating a mesmerizing display in the darkness of space. The astronaut moves between them, touching their delicate, translucent bodies, which react to touch with bursts of light. The scene feels dreamlike, as if the astronaut has entered a cosmic undersea world." \
         --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
