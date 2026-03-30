from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/dataset/data.yaml",

        # FAST SETTINGS (IMPORTANT)
        epochs=15,
        imgsz=512,
        batch=16,   # increase since smaller imgs
        workers=4,

        # speed optimizations
        device="cpu",
        cache=True,
        amp=False,

        # minimal augmentation (still valid)
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,

        project="runs/train",
        name="ppe_yolo_fast",
        exist_ok=True,
        seed=42
    )

if __name__ == "__main__":
    main()