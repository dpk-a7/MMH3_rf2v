# Stage 1: Build Frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Rust Backend
FROM rust:1.99-slim-bookworm AS rust-builder
WORKDIR /backend
COPY backend-rust/Cargo.toml backend-rust/Cargo.lock ./
COPY backend-rust/src ./src
RUN cargo build --release --locked

# Stage 3: CUDA 13.0 Runtime + PyTorch cu130 + Rust Binary
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/comfyui:/app/inference-worker \
    COMFYUI_ROOT=/opt/comfyui \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    TORCH_HOME=/tmp/torch \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl \
    BIND_ADDR=0.0.0.0:8443 \
    MODEL_ROOT=/app/model \
    STATIC_DIR=/app/static \
    EPHEMERAL_DIR=/app/ephemeral \
    IPC_SOCK=/dev/shm/minimax_h3.sock \
    RUST_LOG=off \
    RUST_BACKTRACE=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libgl1 libglib2.0-0 ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /dev/null /var/log/dpkg.log \
    && ln -sf /dev/null /var/log/alternatives.log

RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 \
    && pip3 install --no-cache-dir --break-system-packages \
    safetensors einops transformers accelerate msgpack pyyaml scipy pillow numpy

# Headless ComfyUI core pinned at v0.30.2, which contains MiniMaxH3ReferenceToVideo (PR #15224).
# The HTTP server is never started; worker_daemon imports the node classes in-process.
RUN git clone --depth 1 --branch v0.30.2 --quiet https://github.com/Comfy-Org/ComfyUI.git /opt/comfyui \
    && rm -rf /opt/comfyui/.git \
    && grep -viE '^(torch|torchvision|torchaudio|comfyui-frontend-package|comfyui-workflow-templates|comfyui-embedded-docs)([=<>]|$)' \
        /opt/comfyui/requirements.txt > /tmp/comfy-requirements.txt \
    && pip3 install --no-cache-dir --break-system-packages -r /tmp/comfy-requirements.txt \
    && rm -f /tmp/comfy-requirements.txt

WORKDIR /app
COPY --from=rust-builder /backend/target/release/minimax-h3-backend /app/minimax-h3-backend
COPY --from=frontend-builder /ui/dist /app/static
COPY inference-worker/ /app/inference-worker/

RUN mkdir -p /app/model/diffussion /app/model/lora /app/model/controlnet /app/model/clip /app/model/vae \
    && chmod 0755 /app/minimax-h3-backend

ENTRYPOINT ["/bin/sh", "-c", "mkdir -p /app/ephemeral/uploads /app/ephemeral/outputs /tmp/cache /tmp/torch && chmod 0700 /app/ephemeral /app/ephemeral/uploads /app/ephemeral/outputs && GPU_CFG=$(python3 -c 'import sys; sys.path.insert(0, \"/app/inference-worker\"); import worker_daemon; devs, n = worker_daemon.select_optimal_gpus(); print(f\"{devs} {n}\")' 2>/dev/null || echo '0 1') && SEL_DEVS=$(echo \"$GPU_CFG\" | awk '{print $1}') && NGPU=$(echo \"$GPU_CFG\" | awk '{print $2}') && CUDA_VISIBLE_DEVICES=$SEL_DEVS torchrun --standalone --nproc_per_node=$NGPU /app/inference-worker/worker_daemon.py >/dev/null 2>&1 & exec /app/minimax-h3-backend >/dev/null 2>&1"]
