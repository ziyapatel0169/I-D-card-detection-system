from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')  # You can also use yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data='data/data.yaml',
    epochs=30,  # Adjust based on your needs
    imgsz=640,
    batch=16,    # Adjust based on your GPU memory
    project='runs/train',
    name='exp'
)