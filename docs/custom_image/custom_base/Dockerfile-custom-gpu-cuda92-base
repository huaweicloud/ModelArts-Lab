FROM dls.io/dls/custom-cpu-base:1.0-latest
 
ARG apt_source
ARG download
 
#LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
 
RUN mv /etc/apt/sources.list /etc/apt/sources.list~ 2>/dev/null && \
    wget ${download}/dls-release/ubuntu-16.04/ci-config/sources.list -P /etc/apt/ && \
    mkdir -p /etc/apt/sources.list.d && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys $apt_source/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    wget ${download}/dls-release/ubuntu-16.04/ci-config/sources.list.d/cuda.list -P /etc/apt/sources.list.d/ && \
    wget ${download}/dls-release/ubuntu-16.04/ci-config/sources.list.d/nvidia-ml.list -P /etc/apt/sources.list.d/
ENV CUDA_VERSION 9.2.148
 
ENV CUDA_PKG_VERSION 9-2=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*
# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
 
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.2"
 
ENV NCCL_VERSION 2.3.7
 
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
		cuda-nvtx-$CUDA_PKG_VERSION \
		libnccl2=$NCCL_VERSION-1+cuda9.2 \
		cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda9.2 && \
	apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
 
 
# cuda dnn
ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.2 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*
	
# clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    mv /etc/apt/sources.list~ /etc/apt/sources.list
