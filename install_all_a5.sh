#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#


set -x -eo pipefail

current_path=$(cd $(dirname $0);pwd)
BUILD_ROOT=$current_path
mkdir -p "$BUILD_ROOT"/build/

vllm_version="$1"
vllm_ascend_version="$2"

# 解析可选参数
compile_mooncake="FALSE"
for arg in "$@"; do
  case "${arg,,}" in
    compile_mooncake=true) compile_mooncake="TRUE" ;;
    compile_mooncake=false) compile_mooncake="FALSE" ;;
  esac
done

# 安装 jemalloc
wget --secure-protocol=TLSv1_2 --no-check-certificate https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2
tar -xvf jemalloc-5.3.0.tar.bz2
cd jemalloc-5.3.0
./configure --prefix=/usr/local
make -j && make install
cd ..
rm -rf jemalloc-5.3.0.tar.bz2

# prepare to install by setuptools.
pip install setuptools==77.0.3 setuptools_scm  build numpy==1.26.4 msgpack==1.1.2 concurrent-log-handler==0.9.28

# 下载 vllm
VLLM_DIR="vllm-gpu-${vllm_version}"
# 清空 vllm-gpu-${vllm_version} 文件夹
if [ -d "./${VLLM_DIR}" ]; then
  echo "The ${VLLM_DIR} directory already exists. Please remove the directory or rename it."
  pip uninstall -y vllm
  rm -rf "$VLLM_DIR"
fi
echo "vllm_version: ${vllm_version}"
git clone -b v"${vllm_version}" https://github.com/vllm-project/vllm.git --depth 1 "${VLLM_DIR}"

# 下载 vllm-ascend
VLLM_ASCEND_DIR="vllm-ascend-${vllm_ascend_version}"
# 清空 vllm-ascend-${vllm_ascend_version} 文件夹
if [ -d "./${VLLM_ASCEND_DIR}" ]; then
  echo "The ${VLLM_ASCEND_DIR} directory already exists. Please remove the directory or rename it."
  pip uninstall -y vllm-ascend
  rm -rf "$VLLM_ASCEND_DIR"
fi
echo "vllm_ascend_version: ${vllm_ascend_version}"
VLLM_ASCEND_REPO="${VLLM_ASCEND_REPO:-https://github.com/vllm-project/vllm-ascend.git}"
echo "vllm_ascend_repo: ${VLLM_ASCEND_REPO}"
git clone --depth 1 "${VLLM_ASCEND_REPO}" "${VLLM_ASCEND_DIR}"
git -C "${VLLM_ASCEND_DIR}" fetch origin "${vllm_ascend_version}"
git -C "${VLLM_ASCEND_DIR}" checkout FETCH_HEAD

# 安装 vllm patch
VLLM_PATH=${BUILD_ROOT}/${VLLM_DIR}
cd "${VLLM_PATH}"
# 修改 vllm 的 torch 版本和 vllm_ascend 的 torch 版本保持一致
torch_version=$(grep -o -P "torch\s*==\s*\K[0-9.]+" ${BUILD_ROOT}/${VLLM_ASCEND_DIR}/pyproject.toml)
echo "vllm-ascend torch version: ${torch_version}"
sed -i -E "s/(torch)([[:space:]]*==[[:space:]]*)[0-9.]+/\1\2$torch_version/g" pyproject.toml

if [ -d "${BUILD_ROOT}/ascend_vllm/third_patch/vllm_patch" ]; then
  sed -i 's/\r//g' ${BUILD_ROOT}/ascend_vllm/third_patch/vllm_patch/*.patch
fi

# 安装 vllm
pip install setuptools_rust
export SETUPTOOLS_SCM_PRETEND_VERSION=${vllm_version}
VLLM_TARGET_DEVICE=empty python setup.py bdist_wheel
mv dist/vllm* "${BUILD_ROOT}"/build/
pip install "${BUILD_ROOT}"/build/vllm*whl
pip cache purge

# 安装 vllm_ascend
VLLM_ASCEND_PATH=${BUILD_ROOT}/${VLLM_ASCEND_DIR}
cd "${VLLM_ASCEND_PATH}"
# 解决A3出包A5不可用的问题，6月份950DT的芯片型号是ascend950dt_9582，后续如果有新型号可做成可配置
export SOC_VERSION=ascend950dt_9582
pip install -v -e .
pip cache purge

# 安装 ascend_vllm
cd "$current_path"
pip install -v -e .

# 安装 benchmark 工具
install_tool_whl(){
    PKG_PATH=$1
    PKG_NAME=$2
    if ls "${PKG_PATH}" &> /dev/null; then
        pip uninstall "${PKG_NAME}" -y
        pip install "${PKG_PATH}"
        echo "Install ${PKG_NAME} successfully"
    else
        echo "Skip install ${PKG_NAME}"
    fi
}

install_tool_whl ../../llm_tools/acs_bench-*-py3-none-any.whl acs_bench
install_tool_whl ../../llm_tools/acs_service_profiler-*-py3-none-any.whl acs-service-profiler
install_tool_whl ../../llm_tools/acs_advisor-*-py3-none-any.whl acs-advisor

PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}' || python --version 2>/dev/null | awk '{print $2}')
if [[ "$PYTHON_VERSION" =~ ^3\.11\. ]]; then
  echo "Python version matching required by the acs-quant tool";
  install_tool_whl ../../llm_tools/quantization/rot_quant/acs_quant-*.whl acs-quant
else
  echo "The Python version required by the acs-quant tool does not match. The installation is skipped.";
fi

# 前面安装过程urllib3会被升级，修复urllib3版本
pip install "ray>=2.47.1,<=2.48.0" "protobuf>3.20.0" "urllib3==1.26.11"

cd "$current_path"/ascend_vllm/scripts/
bash patch_third_pkg.sh

# install mooncake
if [ "$compile_mooncake" = "TRUE" ]; then
  echo "[install_all] Installing mooncake with source code (compile_mooncake=${compile_mooncake})"
  cd "$current_path"
  openeuler_yum_mirror="[openEuler-everything]
name=openEuler-everything
baseurl=http://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/
enabled=1
gpgcheck=0
gpgkey=http://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/RPM-GPG-KEY-openEuler

[openEuler-EPOL]
name=openEuler-epol
baseurl=http://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/EPOL/main/aarch64/
enabled=1
gpgcheck=0

[openEuler-update]
name=openEuler-update
baseurl=http://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/update/aarch64/
enabled=1
gpgcheck=0"
  hce_yum_mirror="[base]
name=HCE 3.0 base
baseurl=https://repo.huaweicloud.com/hce/3.0/os/aarch64/
enabled=1
gpgcheck=1
gpgkey=https://repo.huaweicloud.com/hce/3.0/os/RPM-GPG-KEY-HCE-3

[updates]
name=HCE 3.0 updates
baseurl=https://repo.huaweicloud.com/hce/3.0/updates/aarch64/
enabled=1
gpgcheck=1
gpgkey=https://repo.huaweicloud.com/hce/3.0/updates/RPM-GPG-KEY-HCE-3"
  rm -rf /etc/yum.repos.d/*.repo
  echo -e "${openeuler_yum_mirror}" | tee /etc/yum.repos.d/EulerOS.repo > /dev/null
  echo -e "${hce_yum_mirror}" | tee /etc/yum.repos.d/HCE.repo > /dev/null
  cat /etc/yum.repos.d/EulerOS.repo
  cat /etc/yum.repos.d/HCE.repo
  yum install -y rdma-core-devel \
                 gflags-devel \
                 yaml-cpp-devel \
                 gtest-devel \
                 jsoncpp-devel \
                 libunwind-devel \
                 numactl-devel \
                 boost-devel \
                 boost-system \
                 boost-thread \
                 openssl-devel \
                 grpc-devel \
                 protobuf-devel \
                 protobuf-compiler \
                 libcurl-devel \
                 hiredis-devel \
                 patchelf \
                 libzstd-devel \
                 xxhash-devel \
                 libcurl-devel \
                 glog-devel

  git clone -b v0.3.11.post1 https://github.com/kvcache-ai/Mooncake.git
  cd "$current_path"/Mooncake/

  # Check if .gitmodules exists
  if [ -f ".gitmodules" ]; then
    echo "Initializing git submodules..."
    git submodule sync --recursive
    git submodule update --init --recursive
    echo "Git submodules initialized and updated successfully"
  else
    echo -e "No .gitmodules file found. Skipping..."
    exit 1
  fi

  cd ./extern/yalantinglibs
  mkdir build
  cd build
  cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
  cmake --build . -j$(nproc)
  cmake --install .

  cd "$current_path"/Mooncake/extern/
  go_version=1.25.9
  if command -v go &> /dev/null && [ "$(go version | awk '{print $3}')" == "${go_version}" ]; then
    echo "Go ${go_version} installed. Skipping..."
  else
    arch=$(uname -m)
    if [ "${arch}" == "aarch64" ] || [ "${arch}" == "x86_64" ]; then
      arch="arm64"
    else
      echo "Unsupported architecture: ${arch}"
      exit 1
    fi
    wget -q --show-progress http://mirrors.aliyun.com/golang/go${go_version}.linux-${arch}.tar.gz
    tar -zxf go${go_version}.linux-${arch}.tar.gz -C /usr/local/
    rm -rf go${go_version}.linux-${arch}.tar.gz
  fi
  if ! grep -q "export PATH=\$PATH:/usr/local/go/bin" ~/.bashrc; then
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
  fi
  export PATH=$PATH:/usr/local/go/bin
  go env -w GO111MODULE=on
  go env -w GOPROXY=http://mirrors.huaweicloud.com/repository/goproxy/
  go env -w GONOSUMDB=*

  cd "$current_path"/Mooncake/extern/
  git clone -b cpp-7.0.0 https://github.com/msgpack/msgpack-c.git
  cp -r msgpack-c/include/* /usr/local/include/
  export CPLUS_INCLUDE_PATH=/usr/local/include:${CPLUS_INCLUDE_PATH}

  cd "$current_path"/Mooncake/
  rm -rf build
  mkdir build
  cd build
  cmake -DUSE_ASCEND_DIRECT=ON -DUSE_CUDA=OFF -DCMAKE_POLICY_VERSION_MINIMUM=4.0 -DUSE_ETCD=ON -DSTORE_USE_ETCD=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_STORE_RUST=OFF ..
  make -j
  make install
  ldconfig
else
  echo "[install_all] Installing mooncake with wheel package"
  pip install mooncake_transfer_engine_npu==0.3.11.post1
fi

# rebuild openssl
wget --no-check-certificate https://github.com/openssl/openssl/releases/download/openssl-3.4.5/openssl-3.4.5.tar.gz
tar -zxf openssl-3.4.5.tar.gz
cd openssl-3.4.5
mkdir -p /home/ma-user/AscendCloud/openssl
./config enable-md2 --prefix=/home/ma-user/AscendCloud/openssl --openssldir=/home/ma-user/AscendCloud/openssl
make -sj && make install -sj
echo 'export LD_PRELOAD=/home/ma-user/AscendCloud/openssl/lib/libcrypto.so.3:/home/ma-user/AscendCloud/openssl/lib/libssl.so.3' >> ~/.bashrc
