
# 🦺 PPE Helmet Detection System

An intelligent real-time PPE (Personal Protective Equipment) monitoring system that detects helmet compliance using computer vision, reinforcement learning, and natural language processing.

## 🚀 Features

**YOLOv8 Object Detection** – detects heads/person regions
**Convolutional Neural Network (CNN)** – classifies helmet vs no-helmet
**Reinforcement Learning (Q-learning)** – dynamically optimizes detection threshold
**Natural Language Processing (NLP) Module** – generates human-readable safety reports
**GUI Interface (Tkinter + OpenCV)** – supports:
- Webcam monitoring
- Image upload testing
**Learning Curves** – RL performance visualization

## 🧠 System Pipeline

```
Camera/Image → YOLO → CNN → RL → NLP → GUI Display
```

## 📁 Project Structure

```
project-root/
│
├── main_system.py                    # Main GUI system
│
├── models/                           # Saved trained models
│   ├── yolo_ppe_final.pt             # Final YOLO model (trained via Colab)
│   └── cnn_model.pth                 # CNN model (trained locally)
│
├── cnn/                              
│   ├── model.py                      # CNN architecture
│   ├── train.py                      # CNN training script
│   ├── inference.py                  # CNN inference for classification
│   └── prepare_cnn_dataset.py        # Converts YOLO dataset to CNN dataset
│
├── src/                              # Core system modules
│   ├── rl_agent.py                   # RL agent (Q-learning)
│   ├── rl_environment.py             # Reward function
│   ├── rl_inference.py               # RL + YOLO integration logic
│   └── nlp_module.py                 # NLP safety report generator
│
├── data/                             # Datasets
│   ├── dataset/                      # Original YOLO dataset
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── test/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── data.yaml
│   │   └── dataset.zip               # Raw dataset archive (to be downloaded for Colab model training)
│   │
│   └── cnn_dataset/                  # Processed dataset for CNN
│       ├── train/
│       │   ├── head/
│       │   └── helmet/
│       └── valid/
│           ├── head/
│           └── helmet/
│
├── experiments/
│   └── results/
│       ├── yolo/                     # YOLO evaluation outputs (from Colab training)
│       │   ├── results.png
│       │   ├── confusion_matrix.png
│       │   ├── BoxF1_curve.png
│       │   └── BoxPR_curve.png
│       │
│       └── rl_learning.png           # RL learning curve
│
├── notebooks/                        # Jupyter notebooks
│   └── yolo_training_colab.ipynb     # Colab training notebook (GPU)
│
├── scripts/                          # Training & experimentation scripts
│   ├── yolo_dataset_clean.py
│   ├── yolo_dataset_testing.py
│   ├── yolo_train_FAST.py            # Fast training version (<=4hrs completion>)
│   └── yolo_train_HP.py              # High-Performance requirement version (<=25hrs completion)
│
├── docs/                             # Project documentation/reports
│   └── Documentation.pdf
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation and guide
└── yolov8n.pt                        # Base pretrained YOLO weights
```
Note: The /runs, /venv, and /venv312 directories are excluded from the repository for cleanliness and reproducibility.

## ⚙️ Installation Guide

**1️. Clone Repository**

```
git clone <your-repo-link>
cd project-root
```

**2. Create Virtual Environment**

```
python -m venv venv
```

Activate:

Windows
```
venv\Scripts\activate
```
Mac/Linux
```
source venv/bin/activate
```

**3. Install Dependencies**

```
pip install -r requirements.txt
```

If no requirements file:
```
pip install ultralytics torch torchvision opencv-python numpy matplotlib pillow
```

# 📦 Dataset Setup

## YOLO Dataset Structure

```
data/dataset/
    train/
        images/
        labels/
    valid/
        images/
        labels/
    data.yaml
```

## 🧠 CNN Dataset Preparation

```
python cnn/prepare_cnn_dataset.py
```

Output:
```
data/cnn_dataset/train/
    head/
    helmet/
```

## 🧠 CNN Dataset Preparation

```
python cnn/prepare_cnn_dataset.py
```

Output:
```
data/cnn_dataset/train/
    head/
    helmet/
```

## 🏋️ CNN Training

```
python cnn/train.py
```

Output:
```
models/cnn_model.pth
```

## 🎯 YOLO Training (Google Colab)

- Training time: < 90 minutes
- mAP ≈ 0.95

Final model:
```
models/yolo_ppe_final.pt
```

## ▶️ Running the System

```
python main_system.py
```

## 🖥 GUI Options

**Webcam Mode**
- Real-time detection
- RL threshold optimization
- Live monitoring

**Image Mode**
- Upload images
- Useful for helmet testing

## 📊 RL Learning Curve

```
experiments/results/rl_learning.png
```

Expected Results Graph:
- Reward vs steps
- Learning convergence

## 🧠 Model Components

**YOLOv8**
- Detects objects (head regions)

**CNN (From Scratch)**
- 3 convolution layers
- Classifies: Helmet or No Helmet

**Reinforcement Learning**
- Q-learning agent
- Optimizes detection threshold

**NLP Module**
Generates:
- Safety warnings
- Compliance reports

## ⚖️ Ethics & Considerations

- Privacy: Avoid unauthorized surveillance
- Bias: Dataset limitations may affect accuracy
- Usage: Not for safety-critical enforcement

## 📊 Performance

| Metric         | Value                   |
| -------------- | ----------------------- |
| mAP@0.5        | ~0.95                   |
| Precision      | ~0.92                   |
| Recall         | ~0.93                   |
| RL Convergence | Stable (~0.5 threshold) |

## 🧪 Ablation Studies

- Epoch comparison (15 vs 50)
- Image size (512 vs 640)
- Augmentation effects

## 👥 Team Members and Roles

- Arcuino, Shan Harvey H.   - ...
- De Leon, Kim Alyson R.    - ...
- Palma, Jasmine Rose A.    - ...
- Ruiz, Eina Loux M.        - ...

## 📄 License

**Dataset:**
CC0 1.0 Universal (Hard Hat Workers Computer Vision Model)

In partial fulfillment of finals project requirements in Intelligent Systems (6INTELSY) and Data Analytics for Computer Science (6DANCS).
Strictly for academic use only.