# CMI – Detect Behavior with Sensor Data

## Overview

This repository contains my solution for the Kaggle competition: [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)

The objective of the competition is to **detect body-focused repetitive behaviors (BFRBs)** such as hair pulling or skin picking using multimodal sensor data from a wrist-worn device.

This is a **multimodal time-series classification problem** involving:

* Motion (IMU)
* Thermal sensors
* Time-of-Flight (ToF) proximity sensors

## Problem Statement

Given a sequence of pre-segmented sensor readings, the task is to:

1. Distinguish **BFRB vs non-BFRB gestures**
2. Identify the specific type of the **BFRB gesture**

The evaluation metric is a **hierarchical F1 score**, combining:

* Binary F1 (BFRB vs non-BFRB)
* Macro F1 on all gestures (non-target classes are collapsed into a single class)

## Dataset

The data is collected from a wrist-worn device (Helios) equipped with:

* **IMU** (accelerometer + orientation)
* **5 Thermopile sensors** (body heat)
* **5 ToF sensors** (proximity via 8×8 grids)

Each sample is a **variable-length sequence** of sensor readings recorded while a subject performs a gesture.

Key challenges:

* Variable sequence lengths (up to ~700 timesteps)
* Sensor noise and known sensor communication failure
* 50% of test set is only recorded with the IMU sensor (deliberate choice by organizers)
* Device worn incorrectly by some subjects

## Solution Overview

### 1. Feature Engineering

* Removed gravity from accelerometer signals
* Converted quaternions → **relative rotations** (analogous to angular velocity)
* Handedness normalization (left/right wrist alignment)
* Sequence padding/truncation to fixed length

### 2. Model Architecture

A fully convolutional **multimodal fusion network**:

* **IMU Stems**: Separate temporal encoders for raw acceleration, linear acceleration, and rotation
* **Thermal Stem**: Temporal CNN encoder
* **ToF Stem**: Spatial CNN → Temporal encoder
* **Fusion Module** (additive gated fusion):
  * Projects modalities into shared space
  * Learns **gating weights** to adaptively fuse modalities
* **Temporal Encoder**: Residual Conv1D blocks
* **Attention Pooling Head**: Learns important timesteps

## Training Strategy

### Loss & Metric

* Soft-target cross entropy (supports MixUp)
* Hierarchical F1 (competition metric)

### Optimization

* Optimizer: AdamW
* LR Schedule: OneCycle policy

### Data Augmentation

* **Orientation Flip Augmentation:** Simulates incorrectly worn device
* **MixUp (time-series aware)**
* **Modality Dropout:** Simulates missing thermal/ToF sensors

### Validation Strategy

* **Stratified Group K-Fold** (grouped by subject)
* Prevents leakage across subjects

## Repository Structure

```text
.
├── dev/
│   └── data_analysis.ipynb    # important analysis for handedness transformation and missing value imputation 
├── gesture_recognition/
│   ├── dataset.py             # preprocessing, feature extraction, dataset splits
│   ├── model.py               # multimodal fusion model
│   ├── training.py            # train/validation loops
│   └── training_utils.py      # loss, metric, augmentation, checkpoint helpers
├── scripts/
│   └── train.py               # experiment entry point
├── pyproject.toml             # project metadata and uv configuration
├── uv.lock                    # locked dependencies
└── README.md
```

## Results

* **Private Leaderboard Score:** 0.815 (Hierarchical F1)
* **Leaderboard Position:** Top 35%

## Setup and Usage

This project uses **uv** for dependency management and environment setup.

**1. Create the environment**

CPU:

```bash
uv sync --extra cpu --group dev
```

CUDA 12.8:

```bash
uv sync --extra cu128 --group dev
```

**2. Activate the environment**

```bash
source .venv/bin/activate
```

**3. Train the model**

```bash
python scripts/train.py --sensor-dir data/processed --run baseline
```

**4. Run cross-validation**

```bash
python scripts/train.py --run-cv --run cv_experiment
```

## Acknowledgements

* [Child Mind Institute (CMI)](https://www.kaggle.com/organizations/childmindinstitute)
* [Healthy Brain Network (HBN)](https://healthybrainnetwork.org/)
* Kaggle community

## License

MIT
