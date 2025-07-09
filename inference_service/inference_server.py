import time
import grpc
import torch
import torchvision
import numpy as np
import cv2
from concurrent import futures
import prometheus_client

import image_classification_pb2
import image_classification_pb2_grpc

# Prometheus metrics
INFERENCE_TIME = prometheus_client.Gauge(
    'image_classification_inference_time_seconds', 
    'Time taken for image classification'
)
TOTAL_REQUESTS = prometheus_client.Counter(
    'image_classification_total_requests', 
    'Total number of classification requests'
)

class ImageClassificationServicer(image_classification_pb2_grpc.ImageClassificationServiceServicer):
    def __init__(self, model_path=None):
        # Load pretrained ResNet18
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()
        
        # Load ImageNet class labels
        self.labels = self._load_imagenet_labels()
        
        # Preprocessing transforms
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_imagenet_labels(self):
        # Load ImageNet class labels
        url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
        import urllib.request
        import json
        with urllib.request.urlopen(url) as f:
            return json.loads(f.read().decode())

    def ClassifyImage(self, request, context):
        TOTAL_REQUESTS.inc()
        start_time = time.time()

        # Convert bytes to numpy array
        nparr = np.frombuffer(request.image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Get top prediction
        _, predicted = torch.max(outputs, 1)
        label_idx = predicted.item()
        label = self.labels[label_idx]
        confidence = torch.softmax(outputs, dim=1)[0][label_idx].item()
        
        # Calculate inference time
        inference_time = time.time() - start_time
        INFERENCE_TIME.set(inference_time)
        
        return image_classification_pb2.ClassificationResponse(
            label=label,
            confidence=confidence,
            inference_time=inference_time
        )

    def HealthCheck(self, request, context):
        return image_classification_pb2.HealthCheckResponse(
            is_healthy=True,
            version='1.0.0'
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_classification_pb2_grpc.add_ImageClassificationServiceServicer_to_server(
        ImageClassificationServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    
    # Start Prometheus metrics server
    prometheus_client.start_http_server(8000)
    
    print("Server starting on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 