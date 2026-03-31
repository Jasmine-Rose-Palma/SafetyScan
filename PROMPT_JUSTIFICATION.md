

# PROMPTING JUSTIFICATION REQUIREMENT

The following prompts were used, including only core development prompts and excluding debugging, error fixing and minor adjustments. All outputs are validated through actual program code execution and testing.

<br>

## **Section A - 1 :** Prompting was used in this problem

**The following prompt was used:**

```
"Provide a starter YOLOv8 training script optimized for CPU with proper hyperparameters and reproducibility."
```

**Justification:** Needed a structured and reproducible training pipeline for the PPE detection model, including appropriate hyperparameters, augmentation, and logging to meet project requirements.

**Validation:** Verified correctness by successfully training the model, achieving high mAP (~0.95), and ensuring outputs such as PR curves and training logs were generated.

<br>

## **Section A - 2 :** Prompting was used in this problem

**The following prompt was used:**

```
"Training guide for YOLOv8, efficiently using Google Colab GPU within a 90-minute runtime constraint."
```

**Justification:** Required an optimized training approach due to local hardware limitations and project constraints on runtime.

**Validation:** Successfully trained the model on Google Colab within ~15 minutes and confirmed performance metrics were comparable to longer CPU-based training.

<br>

## **Section A - 3 : Prompting was used in this problem**

**The following prompt was used:**

```
"Provide a reinforcement learning threshold agent that dynamically adjusts YOLO confidence threshold with learning curves."
```

**Justification:** Needed to implement the required RL component, including agent logic, reward function, and visualization of learning behavior.

**Validation:** Confirmed functionality through real-time webcam testing, observed adaptive threshold behavior, and generated a valid learning curve (rl_learning.png).

<br>

## **Section A - 4 :** Prompting was used in this problem

**The following prompt was used:**

```
"Generate a CNN starter plate for classifying helmet vs head and integrate it with the YOLO detection pipeline."
```

**Justification:** Required to fulfill the cnn model requirement and to enhance classification accuracy of detected regions.

**Validation:** Successfully trained the CNN model, observed decreasing loss during training, and verified correct classification outputs during system execution.

<br>

## **Section A - 5 :** Prompting was used in this problem

**The following prompt was used:**

```
"Create an NLP component that converts detection results into human-readable safety reports."
```

**Justification:** Needed to satisfy the NLP requirement and improve interpretability of the system outputs.

**Validation:** Verified correctness by observing meaningful and context-aware safety messages (e.g., violation vs compliance) displayed during both webcam and image modes.

<br>

## **Section A - 6 :** Prompting was used in this problem

**The following prompt was used:**

```
"Enhance the system interface to support both webcam and image input modes with a clean GUI."
```

**Justification:** Required to improve usability and allow testing with static images due to limited access to real-world PPE equipment.

**Validation:** Confirmed both modes function correctly, with accurate detections and consistent system outputs across webcam and uploaded images.

<br>

## **Section A - 7 :** Prompting was used in this problem

**The following prompt was used:**

```
"Provide a clean and reproducible project structure, setup instructions and dependency management."
```

**Justification:** Needed to ensure the project meets reproducibility and documentation requirements for submission.

**Validation:** Verified by successfully running the system from a fresh environment using only the README instructions and requirements.txt.