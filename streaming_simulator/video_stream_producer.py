import cv2
import time
import base64
import json
from kafka import KafkaProducer
from kafka import KafkaConsumer
import grpc
import sys
sys.path.append('../inference_service')
import image_classification_pb2
import image_classification_pb2_grpc

class VideoStreamProducer:
    def __init__(self, kafka_broker='localhost:9092', topic='image_classification'):
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic
        
        # gRPC client setup
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = image_classification_pb2_grpc.ImageClassificationServiceStub(channel)

    def stream_video(self, video_path=0, duration=30):
        cap = cv2.VideoCapture(video_path)
        start_time = time.time()
        frame_count = 0
        total_latency = 0

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Send to Kafka
            self.producer.send(self.topic, {
                'frame': jpg_as_text,
                'timestamp': time.time()
            })

            # Perform inference via gRPC
            try:
                inference_start = time.time()
                request = image_classification_pb2.ImageRequest(
                    image_data=buffer.tobytes(),
                    image_format='jpg'
                )
                response = self.stub.ClassifyImage(request)
                
                inference_latency = time.time() - inference_start
                total_latency += inference_latency
                frame_count += 1

                print(f"Frame {frame_count}: {response.label} (Confidence: {response.confidence:.2f})")
            except Exception as e:
                print(f"Inference error: {e}")

        cap.release()
        self.producer.flush()

        # Performance summary
        avg_latency = total_latency / frame_count if frame_count > 0 else 0
        fps = frame_count / duration
        print(f"\nPerformance Summary:")
        print(f"Total Frames: {frame_count}")
        print(f"Average Latency: {avg_latency:.4f} seconds")
        print(f"Throughput: {fps:.2f} FPS")

def main():
    producer = VideoStreamProducer()
    producer.stream_video()

if __name__ == '__main__':
    main() 