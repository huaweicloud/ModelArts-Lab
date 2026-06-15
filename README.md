<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Ascend vLLM Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>关于昇腾</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>官方文档</b></a> | <a href="https://slack.vllm.ai"><b>#sig-ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>用户论坛</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>社区例会</b></a> |
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

## 总览

Ascend-vLLM是华为云针对NPU优化的推理框架，继承了vLLM的优点，并通过特定优化实现了更高的性能和易用性。它使得在NPU卡上运行大模型变得更加高效和便捷，为用户带来了极大的便利和性能提升。Ascend-vLLM可广泛应用于各种大模型推理任务，特别是在需要高性能和高效率的场景中，如自然语言处理、多模态理解等。

Ascend-vLLM的主要特点:
- 易用性：Ascend-vLLM简化了在大模型上的部署和推理过程，使开发者可以更轻松地使用它。
- 易开发性：提供了友好的开发和调试环境，便于模型的调整和优化。
- 高性能：通过自研特性和针对NPU的优化，如前后处理、sample等，实现了高效的推理性能。

## 准备

- 硬件：Atlas 800I A2 Inference系列、Atlas 800I A3 Inference系列
- 操作系统：Linux
- 软件：
    - Python >= 3.10, < 3.12
    - CANN == 9.0.0 (Ascend HDK 版本详见 [版本说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/releasenote_0000.html))
    - PyTorch == 2.10.0, torch-npu == 2.10.0
    - vLLM (与vllm-ascend版本一致)

## 开始使用

### 安装
#### 源码安装
```
git clone -b vllm/vllm-cloud-main https://github.com/huaweicloud/ModelArts-Lab.git
cd ModelArts-Lab
bash install_all.sh ${vllm_version} ${vllm_ascend_version}
```
_参数说明：_
- ${vllm_version} ：vLLM版本，例如：v0.20.2
- ${vllm_ascend_version}: vLLM-Ascend版本，例如：v0.20.2rc1

#### 基于源码构建镜像
```
git clone -b vllm/vllm-cloud-main https://github.com/huaweicloud/ModelArts-Lab.git
cd ModelArts-Lab
export BASE_IMAGE=${base_image}
export VLLM_VERSION=${vllm_version}
export VLLM_ASCEND_VERSION=${vllm_ascend_version}
export BUILD_IMAGE_NAME=${build_image_name}
bash build_image.sh
```
_参数说明：_
- ${base_image}: 基础镜像TAG
- ${vllm_version} ：vLLM版本，例如：v0.20.2
- ${vllm_ascend_version}: vLLM-Ascend版本，例如：v0.20.2rc1
- ${build_image_name}: 构建镜像TAG，例图：ascend-vllm:pytorch_2.10.0-cann_9.0.0-vllm_v0.20.2-vllm_ascend_v0.20.2rc1

#### 基于版本包构建镜像
```
export BASE_IMAGE=${base_image}
export BUILD_IMAGE_NAME=${build_image_name}
unzip AscendCloud-*.zip -d AscendCloud
unzip AscendCloud/AscendCloud-LLM-*.zip -d AscendCloud-LLM
cp AscendCloud/AscendCloud-LLM/llm_infer/build_image.sh ./
bash build_image.sh
```
_参数说明：_
- ${base_image}: 基础镜像TAG
- ${build_image_name}: 构建镜像TAG，例图：ascend-vllm:pytorch_2.10.0-cann_9.0.0-vllm_v0.20.2-vllm_ascend_v0.20.2rc1

## 分支策略

vllm-ascend有主干分支和开发分支。

- **vllm/vllm-cloud-main**: 主干分支，通过昇腾CI持续进行质量看护。
- **releases/vX.Y.Z**: 开发分支，随vLLM部分新版本发布而创建，比如`releases/v0.20.2`是vllm-ascend针对vLLM `v0.20.2` 版本的开发分支。

## 贡献

请参考[CONTRIBUTING](https://github.com/huaweicloud/ModelArts-Lab/tree/vllm/vllm-cloud-main)文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作：
请通过[Issue](https://github.com/huaweicloud/ModelArts-Lab/issues)来告知我们您遇到的任何Bug。


## 许可证

Apache 许可证 2.0，如 [LICENSE](./LICENSE) 文件中所示。