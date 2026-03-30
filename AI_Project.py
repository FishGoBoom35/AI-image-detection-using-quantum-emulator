import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
 
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
 
import cirq
import qsimcirq
 
# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
QUBIT_SWEEP = [4, 6, 8, 10, 12]   # swept for the research question
 
# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
 
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
 
 
# -----------------------------
# LOAD DATASET PATHS
# -----------------------------
def load_dataset_split(base_path, split="train"):
    split_path = os.path.join(base_path, split)
    X_paths, y = [], []
    class_map = {"REAL": 0, "FAKE": 1}
 
    for class_name, label in class_map.items():
        folder = os.path.join(split_path, class_name)
        for filename in os.listdir(folder):
            if filename.lower().endswith(valid_ext):
                X_paths.append(os.path.join(folder, filename))
                y.append(label)
 
    return np.array(X_paths), np.array(y)
 
 
# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)
 
 
# -----------------------------
# CNN FEATURE EXTRACTOR (batched)
# -----------------------------
class CNNFeatureExtractor:
    def __init__(self):
        model = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(DEVICE)
        self.feature_extractor.eval()
 
    def extract(self, paths, batch_size=BATCH_SIZE):
        features = []
 
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                batch = torch.stack([load_image(p) for p in batch_paths]).to(DEVICE)
                feat = self.feature_extractor(batch)          # (B, 512, 1, 1)
                feat = feat.view(feat.size(0), -1)            # (B, 512)
                features.append(feat.cpu().numpy())
 
                if (i // batch_size) % 5 == 0:
                    print(f"  Extracted {min(i + batch_size, len(paths))}/{len(paths)} images...")
 
        return np.concatenate(features, axis=0).astype(np.float32)
 
 
# -----------------------------
# PREPROCESSING PIPELINE
# Fit on train, transform both — no leakage
# -----------------------------
class Preprocessor:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
 
    def fit_transform(self, X):
        X_pca = self.pca.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_pca)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"  PCA({self.pca.n_components_}): {explained:.1%} variance retained")
        return X_scaled
 
    def transform(self, X):
        return self.scaler.transform(self.pca.transform(X))
 
 
# -----------------------------
# QUANTUM FEATURE TRANSFORMER
# Fixed circuit (quantum kernel / feature map)
# -----------------------------
class QuantumFeatureTransformer:
    def __init__(self, n_qubits=6):
        self.n_qubits = n_qubits
        self.qubits = cirq.LineQubit.range(n_qubits)
        self.simulator = qsimcirq.QSimSimulator()
 
    def _build_circuit(self, x):
        """
        Two-layer data re-uploading circuit:
          Layer 1: RY(x) + RZ(0.5x) on each qubit
          Entanglement: CZ chain + CZ(last, first)
          Layer 2: RY(pi*x) re-upload
          Entanglement: Even-pair CZ
        Re-uploading increases expressivity of a fixed (non-trainable) map.
        """
        circuit = cirq.Circuit()
 
        # --- Layer 1: encode ---
        for i, q in enumerate(self.qubits):
            circuit.append(cirq.ry(float(x[i])).on(q))
            circuit.append(cirq.rz(float(x[i]) * 0.5).on(q))
 
        # Circular entanglement
        for i in range(self.n_qubits - 1):
            circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))
        circuit.append(cirq.CZ(self.qubits[-1], self.qubits[0]))
 
        # --- Layer 2: re-upload ---
        for i, q in enumerate(self.qubits):
            circuit.append(cirq.ry(float(x[i]) * np.pi).on(q))
 
        # Even-pair entanglement (different structure from layer 1)
        for i in range(0, self.n_qubits - 1, 2):
            circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))
 
        return circuit
 
    def _quantum_features_one(self, x):
        circuit = self._build_circuit(x)
        result = self.simulator.simulate(circuit)
        state = result.final_state_vector
        qubit_map = {q: i for i, q in enumerate(self.qubits)}
 
        features = []
 
        # Single-qubit Z expectations  →  n_qubits features
        for q in self.qubits:
            z_val = cirq.Z(q).expectation_from_state_vector(
                state_vector=state,
                qubit_map=qubit_map
            ).real
            features.append(z_val)
 
        # Two-qubit ZZ correlators  →  (n_qubits - 1) features
        # These capture entanglement structure in the feature map
        for i in range(self.n_qubits - 1):
            zz_op = cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[i + 1])
            zz_val = zz_op.expectation_from_state_vector(
                state_vector=state,
                qubit_map=qubit_map
            ).real
            features.append(zz_val)
 
        return np.array(features, dtype=np.float32)
 
    def transform(self, X):
        """X must already be PCA-reduced and scaled to [-pi, pi]."""
        assert X.shape[1] == self.n_qubits, (
            f"Expected {self.n_qubits} features, got {X.shape[1]}. "
            "Run Preprocessor first."
        )
        out = []
        for idx, row in enumerate(X):
            out.append(self._quantum_features_one(row))
            if idx % 50 == 0:
                print(f"  Quantum transform: {idx}/{len(X)}...")
        return np.array(out, dtype=np.float32)
 
 
# -----------------------------
# EVALUATION HELPER
# -----------------------------
def evaluate(name, clf, X_test, y_test):
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
 
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    auc  = roc_auc_score(y_test, proba)
 
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["REAL", "FAKE"]))
 
    return {"name": name, "accuracy": acc, "f1": f1, "auc": auc}
 
 
# -----------------------------
# MAIN
# -----------------------------
def main():
    results = []
 
    # ---- Load raw paths ----
    print("\n[1/5] Loading dataset paths...")
    X_train_paths, y_train = load_dataset_split(DATASET_PATH, "train")
    X_test_paths,  y_test  = load_dataset_split(DATASET_PATH, "test")
    print(f"  Train: {len(X_train_paths)} | Test: {len(X_test_paths)}")
    print(f"  Class balance (train) — REAL: {(y_train==0).sum()} | FAKE: {(y_train==1).sum()}")
 
    # ---- CNN features ----
    print("\n[2/5] Extracting CNN (ResNet18) features...")
    extractor = CNNFeatureExtractor()
    X_train_cnn = extractor.extract(X_train_paths)
    X_test_cnn  = extractor.extract(X_test_paths)
    print(f"  Feature shape: {X_train_cnn.shape}")
 
    # ---- Condition A: Full baseline (512-dim) ----
    print("\n[3/5] Condition A — Baseline (full 512-dim ResNet features)...")
    clf_full = LogisticRegression(max_iter=2000)
    clf_full.fit(X_train_cnn, y_train)
    results.append(evaluate("Baseline (512-dim ResNet)", clf_full, X_test_cnn, y_test))
 
    # ---- Sweep qubit counts ----
    print(f"\n[4/5] Sweeping n_qubits = {QUBIT_SWEEP}...")
 
    for n_qubits in QUBIT_SWEEP:
        print(f"\n--- n_qubits = {n_qubits} ---")
 
        # Shared preprocessing (fit once, used by both conditions B and C)
        prep = Preprocessor(n_components=n_qubits)
        X_train_pre = prep.fit_transform(X_train_cnn)
        X_test_pre  = prep.transform(X_test_cnn)
 
        # ---- Condition B: Classical control (PCA only) ----
        clf_pca = LogisticRegression(max_iter=2000)
        clf_pca.fit(X_train_pre, y_train)
        results.append(evaluate(
            f"Classical control (PCA-{n_qubits})",
            clf_pca, X_test_pre, y_test
        ))
 
        # ---- Condition C: Quantum feature map ----
        q_transformer = QuantumFeatureTransformer(n_qubits=n_qubits)
        print(f"  Transforming train set ({len(X_train_pre)} samples)...")
        X_train_q = q_transformer.transform(X_train_pre)
        print(f"  Transforming test set ({len(X_test_pre)} samples)...")
        X_test_q  = q_transformer.transform(X_test_pre)
        print(f"  Quantum feature shape: {X_train_q.shape}  "
              f"(Z expectations + ZZ correlators)")
 
        clf_q = LogisticRegression(max_iter=2000)
        clf_q.fit(X_train_q, y_train)
        results.append(evaluate(
            f"Quantum feature map (n_qubits={n_qubits})",
            clf_q, X_test_q, y_test
        ))
 
        # Cross-validation on quantum features (train set only)
        cv_scores = cross_val_score(
            LogisticRegression(max_iter=2000),
            X_train_q, y_train, cv=5, scoring="roc_auc"
        )
        print(f"  5-fold CV AUC (quantum, n={n_qubits}): "
              f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
 
    # ---- Summary table ----
    print("\n\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Condition':<40} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<40} {r['accuracy']:>6.4f} {r['f1']:>6.4f} {r['auc']:>6.4f}")
    print("="*60)
 
 
if __name__ == "__main__":
    main()