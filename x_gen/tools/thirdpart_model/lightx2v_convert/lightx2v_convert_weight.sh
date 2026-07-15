TARGET_DIR="LightX2V"

if [ -d "$TARGET_DIR" ]; then
    echo "目录 '$TARGET_DIR' 已存在，跳过创建"
    cd ./LightX2V/tools/convert
else
    git config --global http.sslVerify false
    git clone https://github.com/ModelTC/LightX2V.git
    cd ./LightX2V
    git checkout ca39ec400116cf57a4ee47e3f8d7f20bcd28f77e

    pip install loguru qtorch ninja triton gguf imageio_ffmpeg==0.6.0 pydantic diffusers ftfy
    pip install peft==0.17.0

    git apply ../LightX2V_covert_weight.patch

    cd ./tools/convert
fi

if [ -d "$CONVERT_WEIGHT_PATH" ]; then
    echo "目录 '$CONVERT_WEIGHT_PATH' 已存在，跳过创建"
else
    mkdir $CONVERT_WEIGHT_PATH
fi

python converter.py \
     --source "$SOURCE_WEIGHT_HIGH" \
     --output "$CONVERT_WEIGHT_PATH" \
     --output_ext .safetensors \
     --output_name "${CONVERT_WEIGHT_NAME_HIGH}_tmp" \
     --model_type wan_dit \
     --lora_path "$LORA_WEIGHT_HIGH" \
     --lora_strength 1.0 \
     --single_file

python converter.py \
     --source "$SOURCE_WEIGHT_LOW" \
     --output "$CONVERT_WEIGHT_PATH" \
     --output_ext .safetensors \
     --output_name "${CONVERT_WEIGHT_NAME_LOW}_tmp" \
     --model_type wan_dit \
     --lora_path "$LORA_WEIGHT_LOW" \
     --lora_strength 1.0 \
     --single_file

cd ${INFER_WORKSPACE}/AscendCloud/aigc_inference/torch_npu/x_gen/tools/thirdpart_model/lightx2v_convert
python wan2.2_convert_safetensors.py \
    --input "${CONVERT_WEIGHT_PATH}/${CONVERT_WEIGHT_NAME_HIGH}_tmp.safetensors" \
    --output "${CONVERT_WEIGHT_PATH}/${CONVERT_WEIGHT_NAME_HIGH}.safetensors" \
    --type sft
rm -rf ${CONVERT_WEIGHT_PATH}/${CONVERT_WEIGHT_NAME_HIGH}_tmp.safetensors

python wan2.2_convert_safetensors.py \
    --input "${CONVERT_WEIGHT_PATH}/${CONVERT_WEIGHT_NAME_LOW}_tmp.safetensors" \
    --output "${CONVERT_WEIGHT_PATH}/${CONVERT_WEIGHT_NAME_LOW}.safetensors" \
    --type sft
rm -rf ${CONVERT_WEIGHT_PATH}/${CONVERT_WEIGHT_NAME_LOW}_tmp.safetensors
