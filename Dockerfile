# Use the NVIDIA CUDA 12.8 runtime image as the base image
ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.8.0-runtime-ubuntu20.04 as base

# Define the base image for amd64 architecture
FROM base as base-amd64

ENV NV_CUDNN_VERSION=9.7.0.66-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12=${NV_CUDNN_VERSION}

# Define the base image for arm64 architecture
FROM base as base-arm64

ENV NV_CUDNN_VERSION=9.7.0.66-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12=${NV_CUDNN_VERSION}

# Use the appropriate base image based on the target architecture
ARG TARGETARCH
FROM base-${TARGETARCH}

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

# Install cuDNN package
RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to avoid interactive prompts
ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Install pip
RUN python3 -m ensurepip --upgrade

# Install PyTorch with CUDA 12.1 support
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy and install Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
