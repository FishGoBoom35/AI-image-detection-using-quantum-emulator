# Hybrid Quantum-Classical Image Classifier (CIFAKE)

## Overview

This project investigates the research question:

> **Do quantum-inspired feature mappings improve the classification of real vs AI-generated images?**

It implements a hybrid pipeline combining deep learning (CNN feature extraction), classical machine learning, and quantum-inspired transformations.

---

## Pipeline Summary

The system evaluates three conditions:

### **A. Strong Classical Baseline**

```
Image → CNN (ResNet18) → Logistic Regression
```

### **B. Classical Control**

```
Image → CNN → PCA → Logistic Regression
```

### **C. Quantum Feature Mapping (Test Condition)**

```
Image → CNN → PCA → Quantum Feature Map → Logistic Regression
```

This structure ensures a **fair comparison**, isolating the effect of the quantum transformation.

---

## Key Components

### 1. CNN Feature Extraction

* Uses pretrained **ResNet18**
* Extracts 512-dimensional feature vectors from images
* Operates in batches for efficiency

### 2. Dimensionality Reduction

* PCA reduces features to match number of qubits
* Prevents exponential growth in quantum simulation cost

### 3. Quantum Feature Mapping

* Implemented using **Cirq + qsim**
* Uses:

  * Data encoding via rotation gates (RY, RZ)
  * Entanglement (CZ gates)
  * Data re-uploading (second encoding layer)
* Outputs:

  * Single-qubit expectations (Z)
  * Two-qubit correlations (ZZ)

### 4. Classifier

* Logistic Regression used for consistency across conditions
* Evaluated using:

  * Accuracy
  * F1 Score
  * ROC-AUC

---

## Dataset Structure

Expected folder layout:

```
dataset/
│
├── train/
│   ├── REAL/
│   └── FAKE/
│
└── test/
    ├── REAL/
    └── FAKE/
```

Each folder should contain image files (`.jpg`, `.png`, etc.).

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision scikit-learn cirq qsimcirq pillow numpy
```

### 2. Place dataset

Put your dataset in:

```
./dataset/
```

### 3. Run the script

```bash
python your_script_name.py
```

---

## Experiment Design

The experiment sweeps across different numbers of qubits:

```python
QUBIT_SWEEP = [4, 6, 8, 10, 12]
```

For each value:

* PCA reduces features to `n_qubits`
* Quantum mapping is applied
* Performance is evaluated and compared

---

## Output

For each condition, the script prints:

* Accuracy
* F1 Score
* ROC-AUC
* Classification report
* Cross-validation AUC (quantum condition)

It also produces a final summary table for easy comparison.

---

## Important Notes

### 1. Dimensionality Bottleneck

Reducing 512 → small qubit counts (e.g., 6) is highly lossy.
This may limit performance of the quantum pipeline.

### 2. Quantum Model is Not Trainable

The quantum circuit is a **fixed feature map**, not a learned model.

### 3. Runtime Considerations

Quantum simulation is computationally expensive:

* Larger datasets will significantly increase runtime
* Increasing qubits increases cost exponentially

---
