from ultralytics import YOLO
import torch
import os

def main():
    # 1. CONFIG
    DATA_PATH = "data/dataset/data.yaml"
    MODEL_NAME = "yolov8n.pt"  # smallest → best for CPU
    EPOCHS = 50               # good balance for your setup
    IMG_SIZE = 640
    BATCH_SIZE = 8            # safe for 16GB RAM
    PROJECT_NAME = "runs/train"
    EXP_NAME = "ppe_yolo_v1"

    # 2. DEVICE SETUP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. LOAD MODEL
    model = YOLO(MODEL_NAME)

    # 4. TRAIN
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,

        # optimization
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,

        # augmentation (important!)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        # regularization
        patience=10,  # early stopping
        cos_lr=True,

        # logging
        project=PROJECT_NAME,
        name=EXP_NAME,
        exist_ok=True,

        # reproducibility
        seed=42
    )

    print("Training completed.")

if __name__ == "__main__":
    main()