import os
import cv2

INPUT_DIR = "data/dataset/valid"
OUTPUT_DIR = "data/cnn_dataset/valid"

CLASSES = ["head", "helmet"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

images_path = os.path.join(INPUT_DIR, "images")
labels_path = os.path.join(INPUT_DIR, "labels")

for img_name in os.listdir(images_path):
    img_path = os.path.join(images_path, img_name)
    label_path = os.path.join(labels_path, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except:
        print(f"Skipping corrupted label: {label_path}")
        continue

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except:
        print(f"Skipping corrupted label: {label_path}")
        continue

    for i, line in enumerate(lines):
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            cls_id, x, y, bw, bh = map(float, parts)
        except:
            continue

        cls_name = CLASSES[int(cls_id)]

        # Convert YOLO → pixel coords
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        save_path = os.path.join(
            OUTPUT_DIR, cls_name, f"{img_name}_{i}.jpg"
        )

        cv2.imwrite(save_path, crop)

print("CNN dataset prepared!")