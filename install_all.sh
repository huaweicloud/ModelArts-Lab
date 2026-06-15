#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#


set -x -eo pipefail

export COMPILE_CUSTOM_KERNELS=1
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/c++/12:/usr/include/c++/12/`uname -i`-openEuler-linux

current_path=$(cd $(dirname $0);pwd)
BUILD_ROOT=$current_path
mkdir -p "$BUILD_ROOT"/build/

vllm_version="$1"
vllm_ascend_version="$2"
ascend_cloud_work_dir="$3"
soc_version="$4"

# install jemalloc
wget --secure-protocol=TLSv1_2 --no-check-certificate https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2
tar -xvf jemalloc-5.3.0.tar.bz2
cd jemalloc-5.3.0
sudo ./configure --prefix=/usr/local
sudo make -j && sudo make install
cd ..
rm -rf jemalloc-5.3.0.tar.bz2

# download vllm
echo "vllm_version: ${vllm_version}"
VLLM_DIR="vllm-gpu-${vllm_version}"
rm -rf "$VLLM_DIR"
git clone -b "${vllm_version}" https://github.com/vllm-project/vllm.git --depth 1 "${VLLM_DIR}"

# download vllm-ascend
echo "vllm_ascend_version: ${vllm_ascend_version}"
VLLM_ASCEND_DIR="vllm-ascend-${vllm_ascend_version}"
rm -rf "$VLLM_ASCEND_DIR"
git clone -b "${vllm_ascend_version}" https://github.com/vllm-project/vllm-ascend.git --depth 1 "${VLLM_ASCEND_DIR}"

# install vllm patch
VLLM_PATH=${BUILD_ROOT}/${VLLM_DIR}
cd "${VLLM_PATH}"
torch_version=$(grep -o -P "torch\s*==\s*\K[0-9.]+" ${BUILD_ROOT}/${VLLM_ASCEND_DIR}/pyproject.toml)
echo "vllm-ascend torch version: ${torch_version}"
sed -i -E "s/(torch)([[:space:]]*==[[:space:]]*)[0-9.]+/\1\2$torch_version/g" pyproject.toml

if [ -d "${BUILD_ROOT}/ascend_vllm/third_patch/vllm_patch" ]; then
  sed -i 's/\r//g' ${BUILD_ROOT}/ascend_vllm/third_patch/vllm_patch/*.patch
fi

# prepare to install by setuptools.
pip install setuptools==77.0.3 setuptools_scm build numpy==1.26.4 msgpack==1.1.2 concurrent-log-handler==0.9.28

# install vllm
pip uninstall -y vllm
export SETUPTOOLS_SCM_PRETEND_VERSION=${vllm_version}
VLLM_TARGET_DEVICE=empty python setup.py bdist_wheel
mv dist/vllm* "${BUILD_ROOT}"/build/
pip install "${BUILD_ROOT}"/build/vllm*whl
pip uninstall -y triton
pip cache purge

# install vllm_ascend
pip uninstall -y vllm-ascend
VLLM_ASCEND_PATH=${BUILD_ROOT}/${VLLM_ASCEND_DIR}
cd "${VLLM_ASCEND_PATH}"
pip install -v -e .
pip cache purge

# install ascend_vllm
cd "$current_path"
pip install -v -e .

# fix urllib3 version
pip install "ray>=2.47.1,<=2.48.0" "protobuf>3.20.0" "urllib3==1.26.11"

cd "$current_path"/ascend_vllm/scripts/
bash patch_third_pkg.sh

# install mooncake
cd "$current_path"
yum install -y rdma-core-devel gflags-devel yaml-cpp-devel gtest-devel jsoncpp-devel libunwind-devel numactl-devel boost-devel boost-system boost-thread openssl-devel grpc-devel protobuf-devel protobuf-compiler libcurl-devel hiredis-devel patchelf

git clone -b v0.3.8.post1 https://github.com/kvcache-ai/Mooncake.git
cd /Mooncake/

mkdir thirdparties/
cd thirdparties
rm -rf yalantinglibs
git clone -b 0.5.5 https://github.com/alibaba/yalantinglibs.git
cd yalantinglibs
mkdir build
cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build . -j$(nproc)
cmake --install .

cd "$current_path"/Mooncake/thirdparties/
rm -rf glog
git clone -b v0.7.1 https://github.com/google/glog.git
cd glog
cmake -DWITH_GTEST=OFF -S . -B build -G "Unix Makefiles"
cmake --build build --target install -j$(nproc)

cd "$current_path"/Mooncake/thirdparties/
go_version=1.23.8
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
go env -w GO111MODULE=on
go env -w GOPROXY=http://mirrors.huaweicloud.com/repository/goproxy/
go env -w GONOSUMDB=*

cd "$current_path"/Mooncake/
if [ -f ".gitmodules" ]; then
  FIRST_SUBMODULE=$(grep "path" .gitmodules | head -1 | awk '{print $3}')
  echo "Enter respository root: ${current_path}/Mooncake/"
  if [ -d "${current_path}/Mooncake/${FIRST_SUBMODULE}/.git" ] || [ -f "${current_path}/Mooncake/${FIRST_SUBMODULE}/.git" ]; then
    echo "Git submodules already initialized. Skipping..."
  else
    echo "initializing git submodules..."
    git submodule update --init
    echo "Git submodules initialized and updated successfully"
  fi
else
  echo "No .gitmodules file."
  exit 1
fi
source ~/.bashrc
rm -rf build
mkdir build
cd build
cmake -DUSE_ASCEND_DIRECT=ON -DUSE_CUDA=OFF -DCMAKE_POLICY_VERSION_MINIMUM=4.0 -DUSE_ETCD=ON -DSTORE_USE_ETCD=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF ..
make -j
make install
ldconfig
