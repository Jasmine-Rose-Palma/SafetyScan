import os

# Path to your dataset labels
label_dirs = [
    "data/dataset/train/labels",
    "data/dataset/valid/labels",
    "data/dataset/test/labels"
]

REMOVE_CLASS = 2  # person

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file)

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            class_id = int(line.split()[0])

            if class_id != REMOVE_CLASS:
                new_lines.append(line)

        with open(file_path, "w") as f:
            f.writelines(new_lines)

print("✅ Removed 'person' class from all labels.")