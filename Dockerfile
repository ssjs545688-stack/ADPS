FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

SHELL ["/bin/bash", "-c"]

# 빌드 및 ADAS 필수 개발 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    pkg-config \
    libopencv-dev \
    libcjson-dev \
    v4l-utils \
    can-utils \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime C++ SDK 설치 (/usr/local 에 압축해제 및 헤더 경로 정리)
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz -o onnxruntime.tgz \
    && tar -xvf onnxruntime.tgz -C /usr/local --strip-components=1 \
    && mkdir -p /usr/local/include/onnxruntime \
    && cp /usr/local/include/onnxruntime*.h /usr/local/include/onnxruntime/ \
    && rm onnxruntime.tgz \
    && ldconfig

WORKDIR /workspace/dms_project

# 소스 코드 복사 및 빌드 (기존 빌드 아티팩트 정리 후 수행)
COPY . .
RUN rm -rf build && mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# =========================================================
# Runtime Stage (온디바이스 최적화 경량 실행 환경)
# =========================================================
FROM ubuntu:22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

SHELL ["/bin/bash", "-c"]

# 런타임 공유 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-videoio4.5d \
    libopencv-highgui4.5d \
    libopencv-dnn4.5d \
    libcjson1 \
    v4l-utils \
    can-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime 런타임 공유 라이브러리 복사 및 ldconfig
COPY --from=builder /usr/local/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app

COPY --from=builder /workspace/dms_project/build/dms_node /app/dms_node
COPY --from=builder /workspace/dms_project/models /app/models
COPY --from=builder /workspace/dms_project/configs /app/configs

CMD ["/bin/bash"]