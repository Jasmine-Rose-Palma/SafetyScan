
# 📦 Extract Dataset from ZIP files

This folder contains the datasets required for training and running the PPE Helmet Detection System.

## Dataset Download

Due to GitHub file size limitations, datasets are hosted externally.

Download here:
- YOLO Dataset: [Download dataset.zip](https://drive.google.com/file/d/1fQVDbDDnimwIh6XeLpiShOvm_DvKzOds/view?usp=sharing)
- CNN Dataset: [Download cnn_dataset.zip](https://drive.google.com/file/d/1KTkeKuwaqZEPOGEEtZ4neUhZ5ivFp6_s/view?usp=sharing)

**1. After Download**

Place the files inside the ` data/ ` folder, then extract:
```
unzip dataset.zip -d data/
unzip cnn_dataset.zip -d data/
```

Note: Make sure that your final project-root folder structure follows the exact structure provided on the SafetyScan README.md general guide.

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

