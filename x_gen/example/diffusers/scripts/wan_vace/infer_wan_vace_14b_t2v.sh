# Wan2.1 VACE text to video task
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
         --vace_task t2v \
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
         --prompt "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect." \
         --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
