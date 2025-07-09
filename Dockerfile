# Stage 1: Build dependencies
FROM python:3.9-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime image
FROM python:3.9-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache /wheels/*

# Install gRPC tools
RUN pip install grpcio-tools

# Create directories if they don't exist
RUN mkdir -p inference_service streaming_simulator

# Copy project files (use wildcard to prevent errors if directories are empty)
COPY inference_service/ ./inference_service/ || true
COPY streaming_simulator/ ./streaming_simulator/ || true

# Generate gRPC code
RUN python -m grpc_tools.protoc -I./inference_service --python_out=./inference_service --grpc_python_out=./inference_service ./inference_service/image_classification.proto

# Expose gRPC and Prometheus ports
EXPOSE 50051 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD python -c "import grpc; import sys; sys.path.append('./inference_service'); import image_classification_pb2_grpc; channel = grpc.insecure_channel('localhost:50051'); stub = image_classification_pb2_grpc.ImageClassificationServiceStub(channel)" || exit 1

# Default command
CMD ["python", "-m", "inference_service.inference_server"] 