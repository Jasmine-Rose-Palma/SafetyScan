
# 📦 Extract Dataset from ZIP files

This folder contains the datasets required for training and running the PPE Helmet Detection System.

## Contents 

```
data/
├── dataset.zip         # Original YOLO dataset (images + labels + dataset license)
└── cnn_dataset.zip     # Preprocessed dataset for CNN training
```

## Setup Instructions

**1. Extract Datasets**

After cloning the repository, unzip both files:
```
unzip dataset.zip -d data/
unzip cnn_dataset.zip -d data/
```
Note: Extract files on ``` /data ``` upon download on local device. Make sure that your final project-root folder structure follows the exact structure provided on the SafetyScan README.md general guide.

**2. Expected Structure After Extraction**

YOLO Dataset
```
data/dataset/
    train/
        images/
        labels/
    valid/
        images/
        labels/
    test/
        images/
        labels/
    data.yaml
```

CNN Dataset
```
data/cnn_dataset/
    train/
        head/
        helmet/
    valid/
        head/
        helmet/
```

## Notes

- The YOLO dataset is used for object detection training
- The CNN dataset is derived from YOLO bounding boxes and used for classification

If cnn_dataset.zip is not provided, generate it using:
```
python cnn/prepare_cnn_dataset.py
```

## Important

- Do not rename folders after extraction
- Ensure data.yaml paths are correct before training
- Large dataset files are zipped to keep the repository lightweight

