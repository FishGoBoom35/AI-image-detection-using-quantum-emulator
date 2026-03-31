##########################################################################
# CSC 320 Research Project - Hogs 2.0
# Team: Sean Bender, Michael Beehler, Adam Stafford, Anthony Soria
# Description: This program was made with the purpose of exploring our
# research on the usefulness of Quantum computing principles for detecting
# if an image is real or AI generated
##########################################################################
import os
import pickle
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
 
# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
DATASET_PATH  = "dataset"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 32
QUBIT_SWEEP   = [4, 6, 8, 10, 12]
 
# ---- Save / load settings ----
#
# SAVE_DIR holds everything the program produces:
#   cnn_features.npz           - ResNet18 features for train + test (most expensive)
#   preprocessor_n{k}.pkl      - fitted PCA + scaler for each qubit count
#   quantum_features_n{k}.npz  - quantum-transformed features for each qubit count
#   clf_baseline.pkl            - trained baseline classifier
#   clf_classical_n{k}.pkl     - trained classical-control classifier
#   clf_quantum_n{k}.pkl       - trained quantum classifier
#   cv_scores_n{k}.pkl         - cross-validation scores
#
# Set FORCE_RECOMPUTE = True to ignore all saved files and rerun everything.
# Set FORCE_RECOMPUTE = False (default) to skip any step whose output
# already exists on disk — second run completes in seconds.
#
SAVE_DIR        = "saved"
FORCE_RECOMPUTE = False
 
os.makedirs(SAVE_DIR, exist_ok=True)
 
 
# -----------------------------------------------------------------------
# SAVE / LOAD HELPERS
# -----------------------------------------------------------------------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved -> {path}")
 
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
 
def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)
    p = path if path.endswith(".npz") else path + ".npz"
    print(f"  Saved -> {p}")
 
def load_npz(path):
    p = path if path.endswith(".npz") else path + ".npz"
    return np.load(p)
 
def cached(path):
    """True if a file already exists (handles .npz suffix np.savez appends)."""
    return (not FORCE_RECOMPUTE) and (
        os.path.exists(path) or os.path.exists(path + ".npz")
    )
 
 
# -----------------------------------------------------------------------
# IMAGE TRANSFORM
# -----------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
 
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
 
 
# -----------------------------------------------------------------------
# LOAD DATASET PATHS
# -----------------------------------------------------------------------
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
 
 
# -----------------------------------------------------------------------
# LOAD IMAGE
# -----------------------------------------------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)
 
 
# -----------------------------------------------------------------------
# CNN FEATURE EXTRACTOR  (batched, frozen ResNet18)
# -----------------------------------------------------------------------
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
                feat = self.feature_extractor(batch)   # (B, 512, 1, 1)
                feat = feat.view(feat.size(0), -1)     # (B, 512)
                features.append(feat.cpu().numpy())
                if (i // batch_size) % 5 == 0:
                    print(f"    {min(i + batch_size, len(paths)):,}/{len(paths):,} images...")
        return np.concatenate(features, axis=0).astype(np.float32)
 
 
# -----------------------------------------------------------------------
# PREPROCESSING PIPELINE
# Fit on train, transform both — no leakage.
# -----------------------------------------------------------------------
class Preprocessor:
    def __init__(self, n_components):
        self.pca    = PCA(n_components=n_components)
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
 
    def fit_transform(self, X):
        X_pca    = self.pca.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_pca)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"    PCA({self.pca.n_components_}): {explained:.1%} variance retained")
        return X_scaled
 
    def transform(self, X):
        return self.scaler.transform(self.pca.transform(X))
 
 
# -----------------------------------------------------------------------
# QUANTUM FEATURE TRANSFORMER
# Fixed two-layer data re-uploading circuit.
# Output: (2 * n_qubits - 1) features per sample
#   n_qubits single-qubit Z expectations + (n_qubits - 1) ZZ correlators
# -----------------------------------------------------------------------
class QuantumFeatureTransformer:
    def __init__(self, n_qubits=6):
        self.n_qubits  = n_qubits
        self.qubits    = cirq.LineQubit.range(n_qubits)
        self.simulator = qsimcirq.QSimSimulator()
 
    def _build_circuit(self, x):
        circuit = cirq.Circuit()
 
        # Layer 1: RY(x_i) + RZ(0.5 * x_i)
        for i, q in enumerate(self.qubits):
            circuit.append(cirq.ry(float(x[i])).on(q))
            circuit.append(cirq.rz(float(x[i]) * 0.5).on(q))
 
        # Circular CZ entanglement
        for i in range(self.n_qubits - 1):
            circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))
        circuit.append(cirq.CZ(self.qubits[-1], self.qubits[0]))
 
        # Layer 2: re-upload RY(pi * x_i)
        for i, q in enumerate(self.qubits):
            circuit.append(cirq.ry(float(x[i]) * np.pi).on(q))
 
        # Even-pair CZ (different topology from layer 1)
        for i in range(0, self.n_qubits - 1, 2):
            circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))
 
        return circuit
 
    def _quantum_features_one(self, x):
        circuit   = self._build_circuit(x)
        result    = self.simulator.simulate(circuit)
        state     = result.final_state_vector
        qubit_map = {q: i for i, q in enumerate(self.qubits)}
        features  = []
 
        # Single-qubit Z expectations
        for q in self.qubits:
            z_val = cirq.Z(q).expectation_from_state_vector(
                state_vector=state, qubit_map=qubit_map
            ).real
            features.append(z_val)
 
        # Nearest-neighbour ZZ correlators
        for i in range(self.n_qubits - 1):
            zz_op  = cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[i + 1])
            zz_val = zz_op.expectation_from_state_vector(
                state_vector=state, qubit_map=qubit_map
            ).real
            features.append(zz_val)
 
        return np.array(features, dtype=np.float32)
 
    def transform(self, X):
        assert X.shape[1] == self.n_qubits, (
            f"Expected {self.n_qubits} input features, got {X.shape[1]}. "
            "Run Preprocessor(n_components=n_qubits) first."
        )
        out = []
        for idx, row in enumerate(X):
            out.append(self._quantum_features_one(row))
            if idx % 50 == 0:
                print(f"    Quantum transform: {idx:,}/{len(X):,}...")
        return np.array(out, dtype=np.float32)
 
 
# -----------------------------------------------------------------------
# EVALUATION HELPER
# -----------------------------------------------------------------------
def evaluate(name, clf, X_test, y_test):
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
 
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
 
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["REAL", "FAKE"]))
 
    return {"name": name, "accuracy": acc, "f1": f1, "auc": auc}
 
 
# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    results = []
 
    # ------------------------------------------------------------------
    # 1. Load paths
    # ------------------------------------------------------------------
    print("\n[1/5] Loading dataset paths...")
    X_train_paths, y_train = load_dataset_split(DATASET_PATH, "train")
    X_test_paths,  y_test  = load_dataset_split(DATASET_PATH, "test")
    print(f"  Train: {len(X_train_paths):,}  |  Test: {len(X_test_paths):,}")
    print(f"  Class balance (train) — REAL: {(y_train==0).sum():,} | FAKE: {(y_train==1).sum():,}")
 
    # ------------------------------------------------------------------
    # 2. CNN features  (most expensive — cached first)
    # ------------------------------------------------------------------
    cnn_cache = os.path.join(SAVE_DIR, "cnn_features")
 
    if cached(cnn_cache):
        print("\n[2/5] Loading cached CNN features...")
        data        = load_npz(cnn_cache)
        X_train_cnn = data["X_train"]
        X_test_cnn  = data["X_test"]
        print(f"  Loaded shape: {X_train_cnn.shape}")
    else:
        print("\n[2/5] Extracting CNN (ResNet18) features...")
        extractor   = CNNFeatureExtractor()
        X_train_cnn = extractor.extract(X_train_paths)
        X_test_cnn  = extractor.extract(X_test_paths)
        save_npz(cnn_cache, X_train=X_train_cnn, X_test=X_test_cnn)
        print(f"  Feature shape: {X_train_cnn.shape}")
 
    # ------------------------------------------------------------------
    # 3. Condition A: Full 512-dim baseline
    # ------------------------------------------------------------------
    baseline_cache = os.path.join(SAVE_DIR, "clf_baseline.pkl")
 
    print("\n[3/5] Condition A — Baseline (full 512-dim ResNet features)...")
    if cached(baseline_cache):
        print("  Loading cached baseline classifier...")
        clf_full = load_pickle(baseline_cache)
    else:
        clf_full = LogisticRegression(max_iter=2000)
        clf_full.fit(X_train_cnn, y_train)
        save_pickle(clf_full, baseline_cache)
 
    results.append(evaluate("Baseline (512-dim ResNet)", clf_full, X_test_cnn, y_test))
 
    # ------------------------------------------------------------------
    # 4. Qubit sweep — Conditions B (classical) and C (quantum)
    # ------------------------------------------------------------------
    print(f"\n[4/5] Sweeping n_qubits = {QUBIT_SWEEP}...")
 
    for n_qubits in QUBIT_SWEEP:
        print(f"\n--- n_qubits = {n_qubits} ---")
 
        # ---- Preprocessor (PCA + scaler) ----
        prep_cache = os.path.join(SAVE_DIR, f"preprocessor_n{n_qubits}.pkl")
 
        if cached(prep_cache):
            print("  Loading cached preprocessor...")
            prep        = load_pickle(prep_cache)
            X_train_pre = prep.transform(X_train_cnn)
            X_test_pre  = prep.transform(X_test_cnn)
            print(f"    PCA({n_qubits}): {prep.pca.explained_variance_ratio_.sum():.1%} variance retained")
        else:
            prep        = Preprocessor(n_components=n_qubits)
            X_train_pre = prep.fit_transform(X_train_cnn)
            X_test_pre  = prep.transform(X_test_cnn)
            save_pickle(prep, prep_cache)
 
        # ---- Condition B: Classical control ----
        clf_pca_cache = os.path.join(SAVE_DIR, f"clf_classical_n{n_qubits}.pkl")
 
        if cached(clf_pca_cache):
            print("  Loading cached classical classifier...")
            clf_pca = load_pickle(clf_pca_cache)
        else:
            clf_pca = LogisticRegression(max_iter=2000)
            clf_pca.fit(X_train_pre, y_train)
            save_pickle(clf_pca, clf_pca_cache)
 
        results.append(evaluate(
            f"Classical control (PCA-{n_qubits})",
            clf_pca, X_test_pre, y_test
        ))
 
        # ---- Quantum features ----
        qfeat_cache = os.path.join(SAVE_DIR, f"quantum_features_n{n_qubits}")
 
        if cached(qfeat_cache):
            print("  Loading cached quantum features...")
            qdata     = load_npz(qfeat_cache)
            X_train_q = qdata["X_train"]
            X_test_q  = qdata["X_test"]
            print(f"    Quantum feature shape: {X_train_q.shape}")
        else:
            q_transformer = QuantumFeatureTransformer(n_qubits=n_qubits)
            print(f"  Transforming train ({len(X_train_pre):,} samples)...")
            X_train_q = q_transformer.transform(X_train_pre)
            print(f"  Transforming test ({len(X_test_pre):,} samples)...")
            X_test_q  = q_transformer.transform(X_test_pre)
            print(f"    Quantum feature shape: {X_train_q.shape}  "
                  f"({n_qubits} Z + {n_qubits - 1} ZZ)")
            save_npz(qfeat_cache, X_train=X_train_q, X_test=X_test_q)
 
        # ---- Condition C: Quantum classifier ----
        clf_q_cache = os.path.join(SAVE_DIR, f"clf_quantum_n{n_qubits}.pkl")
 
        if cached(clf_q_cache):
            print("  Loading cached quantum classifier...")
            clf_q = load_pickle(clf_q_cache)
        else:
            clf_q = LogisticRegression(max_iter=2000)
            clf_q.fit(X_train_q, y_train)
            save_pickle(clf_q, clf_q_cache)
 
        results.append(evaluate(
            f"Quantum feature map (n_qubits={n_qubits})",
            clf_q, X_test_q, y_test
        ))
 
        # ---- Cross-validation ----
        cv_cache = os.path.join(SAVE_DIR, f"cv_scores_n{n_qubits}.pkl")
 
        if cached(cv_cache):
            cv_scores = load_pickle(cv_cache)
            print(f"  5-fold CV AUC (cached, n={n_qubits}): "
                  f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        else:
            cv_scores = cross_val_score(
                LogisticRegression(max_iter=2000),
                X_train_q, y_train, cv=5, scoring="roc_auc"
            )
            save_pickle(cv_scores, cv_cache)
            print(f"  5-fold CV AUC (n={n_qubits}): "
                  f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
 
    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)
    print(f"  {'Condition':<44} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print(f"  {'-'*44} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(
            f"  {r['name']:<44} "
            f"{r['accuracy']:>6.4f} "
            f"{r['f1']:>6.4f} "
            f"{r['auc']:>6.4f}"
        )
    print("=" * 64)
 
    print("\n  QUANTUM vs CLASSICAL CONTROL (AUC delta)")
    print(f"  {'n_qubits':<12} {'Classical AUC':>14} {'Quantum AUC':>12} {'delta':>8}")
    print(f"  {'-'*12} {'-'*14} {'-'*12} {'-'*8}")
    for n in QUBIT_SWEEP:
        cls_r = next(r for r in results if r["name"] == f"Classical control (PCA-{n})")
        qnt_r = next(r for r in results if r["name"] == f"Quantum feature map (n_qubits={n})")
        delta = qnt_r["auc"] - cls_r["auc"]
        arrow = "+" if delta >= 0 else "-"
        print(
            f"  {n:<12} {cls_r['auc']:>14.4f} {qnt_r['auc']:>12.4f} "
            f"  {arrow}{abs(delta):.4f}"
        )
 
    print(f"\n  All outputs saved to: ./{SAVE_DIR}/")
    print("  Set FORCE_RECOMPUTE = False (default) to load instantly on next run.\n")
 
 
if __name__ == "__main__":
    main()