# 该文件把wan2.2 dit权重 从非diffusers转为diffusers版本

import argparse
from pathlib import Path

import safetensors
from safetensors.torch import save_file
from tqdm import tqdm


def rename_safetensors_weights(input_path, output_path, train_type):
    """
    重命名safetensors文件中的张量名称
    规则：1. 移除名称中的'default.'  2. 开头添加'diffusion_model.'
    参数：
        input_path: 输入原safetensors文件路径（如xxx.safetensors）
        output_path: 输出重命名后的safetensors文件路径
    """
    input_file = Path(input_path)
    if not input_file.exists() or not input_file.suffix == ".safetensors":
        raise FileNotFoundError(f"输入文件不存在或非safetensors格式：{input_path}")

    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        print(f"警告：输出文件已存在，将覆盖 → {output_path}")

    # 3. 读取原 Safetensors 文件
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        original_tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
        original_metadata = f.metadata() or {}
    tensor_total = len(original_tensors)
    # 打印合并任务信息
    print(f"本次处理的目标权重为: {output_path}")
    print(f"原文件读取完成，共检测到 {tensor_total} 个张量")

    # 4. 批量重命名张量名称
    renamed_tensors = {}
    if train_type == "lora":
        for old_name, tensor_data in tqdm(original_tensors.items(), desc="批量重命名张量名称", unit="tensor"):
            temp_name = old_name.replace("default.", "")
            new_name = f"diffusion_model.{temp_name}"
            renamed_tensors[new_name] = tensor_data

    elif train_type == "sft":
        for old_name, tensor_data in tqdm(original_tensors.items(), desc="批量重命名张量名称", unit="tensor"):
            new_name = old_name
            new_name = new_name.replace("head.modulation", "scale_shift_table")
            new_name = new_name.replace("head.head", "proj_out")
            new_name = new_name.replace("text_embedding.0", "condition_embedder.text_embedder.linear_1")
            new_name = new_name.replace("text_embedding.2", "condition_embedder.text_embedder.linear_2")
            new_name = new_name.replace("time_embedding.0", "condition_embedder.time_embedder.linear_1")
            new_name = new_name.replace("time_embedding.2", "condition_embedder.time_embedder.linear_2")
            new_name = new_name.replace("time_projection.1", "condition_embedder.time_proj")

            new_name = new_name.replace("self_attn", "attn1")
            new_name = new_name.replace("cross_attn", "attn2")
            new_name = new_name.replace(".ffn.0", ".ffn.net.0.proj")
            new_name = new_name.replace(".ffn.2", ".ffn.net.2")
            new_name = new_name.replace(".modulation", ".scale_shift_table")
            new_name = new_name.replace(".norm3", ".norm2")
            new_name = new_name.replace(".q", ".to_q")
            new_name = new_name.replace(".k", ".to_k")
            new_name = new_name.replace(".v", ".to_v")
            new_name = new_name.replace(".o", ".to_out.0")
            # 保存重命名后的张量
            renamed_tensors[new_name] = tensor_data

    # 5. 保存重命名后的 Safetensors 文件
    print(f"目标权重保存开始: {output_path}")
    save_file(tensors=renamed_tensors, filename=str(output_file), metadata=original_metadata)
    print(f"目标权重保存结束: {output_path}")

    # 6. 严格验证新文件有效性（校验可读取性+张量数量一致性，避免文件损坏）
    verify_ok = False
    new_tensor_count = 0
    try:
        print(f"验证目标权重有效性开始: {output_path}")
        with safetensors.safe_open(output_path, framework="pt", device="cpu") as f:
            new_tensor_count = len(f.keys())
            if new_tensor_count != tensor_total:
                raise ValueError(f"张量数量不匹配：原{tensor_total}个 → 新{new_tensor_count}个")
        print(f"目标权重有效性验证通过: {output_path}")
        verify_ok = True
    except Exception as e:
        verify_error = str(e)
    else:
        verify_error = "无"

    # 7. 打印最终转换结果（清晰直观，便于核对）
    print("""==================== 目标权重转换完成 ====================""")
    if not verify_ok:
        raise RuntimeError(f"文件验证失败，生成的文件可能损坏：{verify_error}")


def main():
    parser = argparse.ArgumentParser(description="Safetensors张量名称重命名工具")
    parser.add_argument("--input", required=True, help="输入原safetensors文件路径")
    parser.add_argument("--output", required=True, help="输出重命名后的safetensors文件路径")
    parser.add_argument("--type", required=True, choices=["lora", "sft"], help="选择训练类型，仅支持 lora / sft 二选一")
    args = parser.parse_args()

    rename_safetensors_weights(args.input, args.output, args.type)


if __name__ == "__main__":
    main()
