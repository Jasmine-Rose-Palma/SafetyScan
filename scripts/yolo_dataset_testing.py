from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train for 1 epoch (quick test only)
model.train(
    data="data/dataset/data.yaml",
    epochs=1,
    imgsz=640
)