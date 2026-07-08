# Breast Cancer Detection with MobileNetV2 + PSO Hyperparameter Tuning

A deep learning pipeline that classifies mammogram images from the **MIAS dataset** as **Benign** or **Malignant**, using a fine-tuned **MobileNetV2** CNN whose learning rate and dropout rate are optimized via **Particle Swarm Optimization (PSO)**.

## Overview

1. Parses the MIAS `Info.txt` label file and loads `.pgm` mammogram images.
2. Applies heavy data augmentation (rotation, shift, shear, zoom, flips, brightness, Gaussian noise) to balance and expand a small dataset (200 images total after augmentation, 100 per class).
3. Splits data into train / PSO-validation / final test sets.
4. Uses PSO (`pyswarm`) to search for the best `learning_rate` and `dropout_rate` for a MobileNetV2-based classifier, evaluating each candidate by training for 40 epochs and scoring on validation accuracy.
5. Trains a final model with the best PSO parameters using a triangular cyclic learning rate schedule, early stopping, and model checkpointing.
6. Evaluates on a held-out test set and runs inference on a sample image.

## Architecture

- **Base model:** MobileNetV2 (ImageNet weights, fully fine-tuned, input size 192×192×3)
- **Head:** GlobalAveragePooling2D → BatchNorm → Dense(256, L2 reg) → Dropout → Dense(128, L2 reg) → Dropout → Dense(64, L2 reg) → Dense(1, sigmoid)
- **Loss / metric:** Binary cross-entropy / accuracy
- **Optimizer:** Adam (learning rate tuned by PSO)

## Requirements

```
tensorflow
pyswarm
numpy
scikit-learn
matplotlib
pillow
```

Also requires the **MIAS mammography dataset**, expected at:

```
archive/
├── Info.txt
└── all-mias/
    ├── mdb001.pgm
    ├── mdb002.pgm
    └── ...
```

## Usage

Run the notebook cells in order:

1. Load libraries and configure GPU memory growth.
2. Load & augment data from `archive/`.
3. Train/test split (80/20), then further split train into PSO-train/PSO-val (80/20).
4. Run PSO search (`swarmsize=6`, `maxiter=5`) over `learning_rate ∈ [5e-6, 1e-4]` and `dropout_rate ∈ [0.05, 0.25]`.
5. Train the final model with the best parameters (cyclic LR, early stopping, checkpointing to `best_model.h5`).
6. Evaluate on the test set and predict on a sample image.
7. Final model saved as `mobilenetv2_mias_benign_malignant.h5`.

## Results (from this run)

- Best PSO parameters: `learning_rate ≈ 0.000041–0.000058` range explored, best validation accuracy of **0.75** reached during search.
- **Final test accuracy: 87.5%** (on a very small test set of ~40 images).
- Sample prediction on `mdb001.pgm`: **Benign**, 99.98% confidence.

## Caveats & Limitations

- **Very small dataset:** only 200 images (100 per class) after augmentation, with a test set of roughly 40 images — 87.5% test accuracy corresponds to only a handful of test samples, so the number should be read as a rough estimate, not a robust performance metric.
- **Augmentation-derived duplicates:** since augmented variants of the same source image can end up split across train/val/test, there's a risk of data leakage inflating accuracy.
- **Trivial class weights:** the computed class weights came out to `{0: 1.0, 1: 1.0}` — since the augmentation step already balances classes at 100/100, the class-weight step in this pipeline currently has no real effect.
- **Lightweight PSO search:** `swarmsize=6, maxiter=5` (30 candidate evaluations, each with a 40-epoch train) is a fairly coarse search; results may vary noticeably across reruns.
- **No cross-validation:** results are based on a single train/test split, so accuracy may not generalize.

## Disclaimer

This project is for research/educational purposes and is **not a validated diagnostic tool**. It should not be used for real clinical decision-making without rigorous validation on a much larger, clinically curated dataset.
