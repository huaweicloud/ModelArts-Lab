ARG base_image
FROM ${base_image}

USER root

# A3: ascend910_9391, A2: ascend910b1
ARG SOC_VERSION

ARG ascend_cloud_work_dir
RUN mkdir -p ${ascend_cloud_work_dir}

ENV ascend_cloud_work_dir=${ascend_cloud_work_dir}

WORKDIR ${ascend_cloud_work_dir}/

COPY ./ ./

ARG pip_mirror_index_url
ARG pip_mirror_extra_index_url
ARG pip_mirror_trusted_host
RUN pip config set global.index-url ${pip_mirror_index_url} && \
    pip config set global.extra-index-url ${pip_mirror_extra_index_url} && \
    pip config set global.trusted-host ${pip_mirror_trusted_host} && \
    git config --global http.sslVerify false && \
    git config --global http.sslCipherSuite "DEFAULT:@SECLEVEL=1"

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG vllm_version
ARG vllm_ascend_version
RUN if [ -n "${http_proxy}" ] && [ -n "${https_proxy}" ]; then && \
        export http_proxy=${http_proxy}; \
        export https_proxy=${https_proxy}; \
        export no_proxy=${no_proxy}; \
    fi && \
    cd ${ascend_cloud_work_dir}/AscendCloud-LLM/llm_infer/ && \
    bash install_all.sh ${vllm_version} ${vllm_ascend_version} && \
    unset http_proxy https_proxy no_proxy

RUN source ~/.bashrc && \
    chmod 755 ${ascend_cloud_work_dir} && \
    rm -rf ${ascend_cloud_work_dir}/*.zip

ENV VLLM_PLUGINS=ascend_vllm;kv_connectors
ENV VLLM_USE_V1=1