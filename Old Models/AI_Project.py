"""
Improved hybrid quantum-classical AI image classifier using qsimcirq

Changes from the earlier version:
1. Uses 6 PCA features instead of 4
2. Uses a deeper 6-qubit circuit
3. Extracts quantum features from qsimcirq
4. Uses Logistic Regression on the quantum features instead of nearest-centroid

This is still a hybrid model:
- classical preprocessing
- quantum-emulated feature extraction
- classical final classifier
"""

import os
import cv2
import numpy as np
import cirq
import qsimcirq

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# File paths
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
real_folder = os.path.join(base_dir, "images", "real")
ai_folder = os.path.join(base_dir, "images", "ai")


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(path, size=(64, 64)):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    gray = gray.astype(np.float32) / 255.0
    return gray


# -----------------------------
# Build dataset
# -----------------------------
data = []
labels = []

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for filename in sorted(os.listdir(real_folder)):
    if not filename.lower().endswith(valid_ext):
        continue
    path = os.path.join(real_folder, filename)
    data.append(preprocess_image(path))
    labels.append(0)

for filename in sorted(os.listdir(ai_folder)):
    if not filename.lower().endswith(valid_ext):
        continue
    path = os.path.join(ai_folder, filename)
    data.append(preprocess_image(path))
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

# Flatten images: (n, 32, 32) -> (n, 1024)
X = data.reshape(len(data), -1)
y = labels


# -----------------------------
# Quantum feature transformer
# -----------------------------
class QuantumFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that:
    - takes PCA-reduced classical features
    - encodes them into a Cirq circuit
    - runs the circuit on qsimcirq
    - returns quantum-derived features
    """

    def __init__(self, n_qubits=6):
        self.n_qubits = n_qubits
        self.qubits = cirq.LineQubit.range(n_qubits)
        self.simulator = qsimcirq.QSimSimulator()

    def fit(self, X, y=None):
        return self

    def _quantum_features_one(self, x):
        if len(x) != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} input features, got {len(x)}")

        circuit = cirq.Circuit()

        # First encoding layer
        for i, q in enumerate(self.qubits):
            circuit.append(cirq.ry(float(x[i])).on(q))
            circuit.append(cirq.rz(float(x[i]) * 0.5).on(q))

        # First entangling layer
        for i in range(self.n_qubits - 1):
            circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))

        # Second encoding layer
        for i, q in enumerate(self.qubits):
            circuit.append(cirq.rx(float(x[i]) * 0.7).on(q))
            circuit.append(cirq.ry(float(x[i]) * 0.3).on(q))

        # Ring entanglement: connect last qubit back to first
        circuit.append(cirq.CZ(self.qubits[-1], self.qubits[0]))

        # Simulate the circuit
        result = self.simulator.simulate(circuit)
        state = result.final_state_vector

        # Extract <Z> expectation values
        features = []
        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        for q in self.qubits:
            z_val = cirq.Z(q).expectation_from_state_vector(
                state_vector=state,
                qubit_map=qubit_map
            ).real
            features.append(z_val)

        return np.array(features, dtype=np.float32)

    def transform(self, X):
        return np.array([self._quantum_features_one(row) for row in X], dtype=np.float32)


# -----------------------------
# Model pipeline
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=6)),
    ("quantum_features", QuantumFeatureTransformer(n_qubits=6)),
    ("clf", LogisticRegression(max_iter=2000))
])


# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
print("Predictions:", predictions)
print("Actual:     ", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))