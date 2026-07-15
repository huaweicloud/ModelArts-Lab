#!/bin/bash

##############################################################################
# Compatibility Note:
# This version is compatible with ulysses_sp
##############################################################################


get_repo_info() {
    local repo_name=$1
    local config_file="repo_config.txt"

    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        echo "错误：配置文件 $config_file 不存在！"
        exit 1
    fi

    # 从配置文件中匹配仓库名，提取地址和commit（跳过注释行）
    repo_info=$(grep -v "^#" "$config_file" | grep "^$repo_name " | awk '{print $2, $3}')
    if [ -z "$repo_info" ]; then
        echo "错误：配置文件中未找到 $repo_name 的信息！"
        exit 1
    fi
    echo "$repo_info"
}


# prepare MindSpeed
repo_info=$(get_repo_info "MindSpeed")
clone_url=$(echo "$repo_info" | awk '{print $1}')
target_commit=$(echo "$repo_info" | awk '{print $2}')

for i in {1..10}
do
    git clone $clone_url && break
    echo "尝试第 $i 次失败，正在重试..."
done

if [ -d "MindSpeed" ]; then
    cd MindSpeed && git checkout $target_commit
    pip install -e .
    cd ..
else
    echo "Git MindSpeed 克隆失败，退出脚本！"
    exit 1
fi


# prepare long-context-attention
repo_info=$(get_repo_info "long-context-attention")
clone_url=$(echo "$repo_info" | awk '{print $1}')
target_commit=$(echo "$repo_info" | awk '{print $2}')


for i in {1..10}
do
    git clone $clone_url && break
    echo "尝试第 $i 次失败，正在重试..."
done

if [ -d "long-context-attention" ]; then
    cd long-context-attention && git checkout $target_commit
    cp -f ../../../../../../../aigc_train/torch_npu/DiffSynth-Studio/wan2.2-longframe/yunchang.patch ./
    git apply yunchang.patch || { echo '补丁应用失败'; exit 1; }
    pip install -e .
    cd ..
else
    echo "Git long-context-attention 克隆失败，退出脚本！"
    exit 1
fi
