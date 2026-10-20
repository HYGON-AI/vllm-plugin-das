# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# 直接基于预装 DTK 的基础镜像构建 (DTK 已内置于 BASE_IMAGE, CI 内不再安装 DTK):
#   基础镜像来自镜像总表 ${HARBOR_SITE}/hcu/base/dtk:<DTK版本>-<OS>-py<版本>,
#   由 CI 通过 build-arg 按固定格式拼接传入; 换 DTK 包 = 换基础镜像 tag
# pip nightly 归档路径不再由参数拼接: 基础镜像自带 ENV DTK_VERSION
# (token 格式 dtk2604/dtk2610), FROM 之后直接读它拼出 nightly/dtk2604/+simple/
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV ROCM_PATH=/opt/dtk

RUN echo "source /opt/dtk/env.sh" >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH' >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

# 内网 pip 源 / 资源服务器地址由 CI 通过 build-arg 注入 (仓库变量)
ARG PYPI_URL
ARG RESOURCE_SERVER_URL

# pip 配置 (nightly 归档按基础镜像的 DTK_VERSION token 拼接) + das-install
RUN TRUSTED_HOST="${PYPI_URL#*://}" && TRUSTED_HOST="${TRUSTED_HOST%%/*}" && \
    mkdir -p ~/.pip && \
    printf '%s\n' \
    '[global]' \
    "index-url = ${PYPI_URL}/nightly/${DTK_VERSION}/+simple/" \
    "trusted-host = ${TRUSTED_HOST}" \
    > ~/.pip/pip.conf && \
    pip install --ignore-installed --no-cache-dir blinker && \
    printf '%s\n' \
    '#!/bin/bash' \
    '# usage: das-install <pkg>[==<ver>] <torch_tag>' \
    'pkg=$1 tag=$2 prefix=' \
    'if [[ "$pkg" == *==* ]]; then prefix="${pkg#*==}"; pkg="${pkg%%==*}"; fi' \
    'ver=$(pip index versions "$pkg" 2>/dev/null | grep -oP "${prefix}[^ ()]*${tag}[^ ()]*" | head -1)' \
    '[ -z "$ver" ] && echo "Error: No version for $pkg matching $tag" && exit 1' \
    'echo "Installing $pkg==$ver"' \
    'pip install --no-cache-dir --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com "$pkg==$ver"' \
    > /usr/local/bin/das-install && chmod +x /usr/local/bin/das-install

# AICC 编译器安装 (安装脚本安装成功后会在 $DTK_DIR/hcu_llvm.info 写入本次完整版本/commit 等记录);
# AICC 大版本由 CI 按矩阵的 aicc_major 字段 --build-arg 传入, 脚本 --major 装该版本中日期最新的包
# (大版本与 dtk 段对应: dtk2604/dtk26042 -> 1.0.0/2.0.0 按归档实际为准, dtk2610 -> 2.0.0)
ARG AICC_MAJOR
RUN wget -q "${RESOURCE_SERVER_URL}/ai_cc/Nightly/hcu_llvm_installer.sh" -O /tmp/hcu_llvm_installer.sh && \
    chmod +x /tmp/hcu_llvm_installer.sh && \
    bash /tmp/hcu_llvm_installer.sh --major "${AICC_MAJOR}" && \
    rm -f /tmp/hcu_llvm_installer.sh

# 从 nightly 归档拉取blas cut 最新包, 解压后覆盖到 DTK 安装目录(默认 /opt/dtk);
# blas cut 版本由 CI 按矩阵的 blas_cut_version 字段传入, 与基础镜像 DTK 大版本保持一致
ARG BLAS_CUT_VERSION
RUN wget -q "${RESOURCE_SERVER_URL}/Jenkins/CI_scripts/dtk_blas_cut_installer.sh" -O /tmp/dtk_blas_cut_installer.sh && \
    chmod +x /tmp/dtk_blas_cut_installer.sh && \
    bash /tmp/dtk_blas_cut_installer.sh --version "${BLAS_CUT_VERSION}" --os ubuntu20.04 && \
    rm -f /tmp/dtk_blas_cut_installer.sh

# ---layer--- vLLM 框架依赖
ARG VLLM_VERSION
RUN pip install --no-cache-dir ninja wheel setuptools \
    && pip install --no-cache-dir ray[data,train,tune,serve] -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com \
    && pip install --no-cache-dir amdsmi==1.0.0+630c16a6.dirty \
    && pip install --no-cache-dir cupy \
    && pip install --no-cache-dir vllm==0.28.1 \
    && pip install --no-cache-dir --pre "vllm-hcu>0.28.0,<=0.28.1" \
    && pip cache purge

# ---layer--- torch + das 加速库
ARG TORCH_VERSION
# flash_attn 包名按 DTK 版本编译, 归档里的版本串是 dtk<DTK_VERSION>.torch<TORCH_TAG>.<时间戳>.<commit>;
# 用 das-install 按 torch_tag 前缀匹配, 避免硬编码完整包名 (换 dtk/torch 版本不用改这里)
RUN TORCH_TAG="torch${TORCH_VERSION//./}" \
    && pip install --no-cache-dir torch==${TORCH_VERSION} torchvision \
    && das-install triton==3.6.0 ${TORCH_TAG} \
    && das-install flash_attn ${TORCH_TAG} \
    && das-install lightop ${TORCH_TAG} \
    && das-install lmslim ${TORCH_TAG} \
    && das-install deepgemm ${TORCH_TAG} \
    && das-install aiter ${TORCH_TAG} \
    && pip install --no-cache-dir mooncake_transfer_engine \
    && das-install deep_ep ${TORCH_TAG} \
    && das-install boltops ${TORCH_TAG} \
    && das-install causal_conv1d==1.5.4 ${TORCH_TAG} \
    && das-install flash_mla ${TORCH_TAG} \
    && das-install fastsafetensors ${TORCH_TAG} \
    && das-install flash_kda ${TORCH_TAG} \
    && das-install vllm-router ${TORCH_TAG} \
    && das-install nixl ${TORCH_TAG} \
    # tilelang 放最后: 安装真实包会替换上面 vLLM 层写的 0.1.10 stub, 之后不应再有触发 vllm 依赖解析的安装
    && das-install tilelang ${TORCH_TAG} \
    && pip install --no-cache-dir apache-tvm-ffi==0.1.11 \
    && pip install --no-cache-dir numpy==2.0.2 \
    && pip cache purge
    

# UCX 已在基础镜像内 (/opt/ucx, 路径已在其 ENV 中), 不再重复编译

# 构建完成后移除内网 pip 源, 避免运行时意外拉取内网依赖
RUN rm -rf ~/.pip/pip.conf

# /etc/image.info (口径同基础镜像 dtk 段: ASCII 大字 + 名称/构建时间, 容器启动时由
# ENTRYPOINT /usr/local/bin/hcu_image_info 打印); 大字换成 vllm, 覆盖基础镜像的 DTK 记录;
# figlet 字体与基础镜像同源 (资源服务器 pkg/slant.flf); BUILD_TIME 由 CI 传入, 与产物 tag 的时间段一致
ARG BUILD_TIME
RUN cd /tmp && \
    wget -q "${RESOURCE_SERVER_URL}/pkg/slant.flf" -O /tmp/slant.flf && \
    { figlet -f slant -d /tmp -w 80 vllm; \
      echo "dtk    : ${DTK_VERSION}"; \
      echo "built  : ${BUILD_TIME:-$(date +%Y%m%d-%H%M)}"; } > /etc/image.info && \
    rm -f /tmp/slant.flf
