# Dockerfile - Optimized for RTX 4060 + CUDA Toolkit Parity
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04   
# Prevent interactive prompts blocking the build process
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Explicitly prepare build environments for wheels
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# --- CRITICAL CUDA COMPILATION CORRECTION ---
# Replaces obsolete LLAMA_CUBLAS flags with modern unified CUDA backends
ENV GGML_CUDA=on
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python --force-reinstall --no-cache-dir

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]