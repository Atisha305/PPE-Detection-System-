import torch
import cv2
import os
from datetime import datetime

# Load YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Modify labels to simulate PPE (Helmet = 'hardhat', Vest = 'vest') if using a custom model
# For now, we'll simulate detection using 'person' class as a placeholder
TARGET_CLASSES = ['person']  # Replace with 'helmet', 'vest' if using a custom trained model

def detect_ppe(video_path, save_path="runs/detections/"):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(save_path, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(save_path, 'output.avi'), fourcc, 20.0, (640, 480))

    frame_id = 0
    violation_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        results = model(frame)
        df = results.pandas().xyxy[0]  # pandas DataFrame

        detected_classes = df['name'].tolist()

        # Simulate violation: if person detected but not wearing PPE (in real case, check for helmet/vest)
        if 'person' in detected_classes:
            cv2.putText(frame, "PPE Compliance: CHECK MANUALLY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            log = f"Frame {frame_id}: Manual check needed at {datetime.now()}"
            violation_log.append(log)

        # Draw boxes
        results.render()
        out.write(results.ims[0])

        cv2.imshow('PPE Detection', results.ims[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save logs
    with open(os.path.join(save_path, 'violations.log'), 'w') as f:
        for line in violation_log:
            f.write(line + '\n')

if __name__ == "__main__":
    detect_ppe("test_videos/sample.mp4")
