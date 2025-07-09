import grpc
import time
import concurrent.futures
import cv2
import numpy as np
import sys
sys.path.append('./inference_service')
import image_classification_pb2
import image_classification_pb2_grpc

def generate_test_image():
    # Generate a random test image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_single_inference(stub):
    start_time = time.time()
    request = image_classification_pb2.ImageRequest(
        image_data=generate_test_image(),
        image_format='jpg'
    )
    response = stub.ClassifyImage(request)
    latency = time.time() - start_time
    return response, latency

def load_test(num_requests=100, concurrency=10):
    channel = grpc.insecure_channel('localhost:50051')
    stub = image_classification_pb2_grpc.ImageClassificationServiceStub(channel)

    latencies = []
    
    def worker():
        response, latency = test_single_inference(stub)
        latencies.append(latency)
        return response

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(num_requests)]
        concurrent.futures.wait(futures)

    # Performance analysis
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print("\nLoad Test Results:")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Average Latency: {avg_latency:.4f} seconds")
    print(f"95th Percentile Latency: {p95_latency:.4f} seconds")

def main():
    load_test()

if __name__ == '__main__':
    main() 