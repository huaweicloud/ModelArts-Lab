#!/bin/bash
set -e

script_dir=$(cd $(dirname $0);pwd)

base_image=${BASE_IMAGE:-""}
if [[ "${base_image}" == "" ]]; then
    echo "Base image not configured"
    exit 1
fi
build_image_tag="$(echo ${base_image} | awk -F '/' '{print $NF}')-$(date +%Y%m%d%H%M%S)"
if [[ -v BUILD_IMAGE_NAME && -n "${BUILD_IMAGE_NAME}" ]]; then
    build_image_tag=${BUILD_IMAGE_NAME}
fi
echo "Build image tag: ${build_image_tag}"

ascend_cloud_work_dir=${ASCEND_CLOUD_WORK_DIR:-"/opt/AscendCloud"}
pip_mirror_index_url=${PIP_MIRROR_INDEX_URL:-"https://mirrors.huaweicloud.com/repository/pypi/simple"}
pip_mirror_extra_index_url="https://triton-ascend.osinfra.cn/pypi/simple"
pip_mirror_trusted_host="${PIP_MIRROR_TRUSTED_HOST:-'mirrors.huaweicloud.com'} triton-ascend.osinfra.cn"
yum_mirror_endpoint=${YUM_MIRROR_ENDPOINT:-"http://mirrors.huaweicloud.com"}
yum_mirror="[openEuler-everything]
name=openEuler-everything
baseurl=${yum_mirror_endpoint}/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/
enabled=1
gpgcheck=0
gpgkey=${yum_mirror_endpoint}/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/RPM-GPG-KEY-openEuler

[openEuler-EPOL]
name=openEuler-epol
baseurl=${yum_mirror_endpoint}/openeuler/openEuler-22.03-LTS-SP4/EPOL/aarch64/
enabled=1
gpgcheck=0

[openEuler-update]
name=openEuler-update
baseurl=${yum_mirror_endpoint}/openeuler/openEuler-22.03-LTS-SP4/update/aarch64/
enabled=1
gpgcheck=0"
http_proxy=${http_proxy}
https_proxy=${https_proxy}
no_proxy=${no_proxy:-"127.0.0.1,localhost,*.huawei.com"}
vllm_version=${VLLM_VERSION:-"$(cat ${script_dir}/AscendCloud/AscendCloud-LLM/llm_infer/version.info | grep -m1 'vLLM Version:' | awk -F ': ' '{print $2}')"}
vllm_ascend_version=${VLLM_ASCEND_VERSION:-"$(cat ${script_dir}/AscendCloud/AscendCloud-LLM/llm_infer/version.info | grep -m1 'vLLM Ascend Version:' | awk -F ': ' '{print $2}')"}
if [[ -v SOC_VERSION && -n "${SOC_VERSION}" ]]; then
    soc_version=${SOC_VERSION}
else
    if [[ "$(echo ${base_image} | awk -F ':' '{print $NF}')" == *"snt9b23" ]]; then
        soc_version="ascend910_9391"
    else
        soc_version="ascend910b1"
    fi
fi

echo "Build Ascend-vLLM image..."
cd ${script_dir}/AscendCloud/
docker build --network host --no-cache \
--build-arg base_image=${base_image} \
--build-arg ascend_cloud_work_dir=${ascend_cloud_work_dir} \
--build-arg pip_mirror_index_url=${pip_mirror_index_url} \
--build-arg pip_mirror_extra_index_url=${pip_mirror_extra_index_url} \
--build-arg pip_mirror_trusted_host=${pip_mirror_trusted_host} \
--build-arg yum_mirror=${yum_mirror} \
--build-arg http_proxy=${http_proxy} \
--build-arg https_proxy=${https_proxy} \
--build-arg no_proxy=${no_proxy} \
--build-arg vllm_version=${vllm_version} \
--build-arg vllm_ascend_version=${vllm_ascend_version} \
--build-arg soc_version=${soc_version} \
-f ${script_dir}/AscendCloud/AscendCloud-LLM/llm_infer/Docckerfile \
-t ${build_image_tag} .
echo "Build Ascend-vLLM image success. image tag: ${build_image_tag}"
