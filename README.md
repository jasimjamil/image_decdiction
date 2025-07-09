# Kookree ML Systems Engineer Technical Assignment

## Project Overview
This project implements a real-time image classification pipeline using:
- PyTorch for model inference
- OpenCV for image preprocessing
- gRPC for service communication
- Kafka for streaming simulation
- Docker for containerization

## Prerequisites
- Docker
- Python 3.8+
- Kafka/Redpanda

## Setup Instructions
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate gRPC code:
```bash
python -m grpc_tools.protoc -I./inference_service --python_out=. --grpc_python_out=. ./inference_service/image_classification.proto
```

3. Build Docker image:
```bash
docker build -t kookree-ml-service .
```

4. Run the service:
```bash
docker-compose up
```

## Performance Metrics
- Latency tracking per frame
- Throughput (frames per second)
- Prometheus metrics endpoint

## Testing
```bash
python test_grpc_endpoint.py
```

## Optional Optimizations
- TorchScript model conversion
- ONNX runtime support
- GPU/CPU fallback
