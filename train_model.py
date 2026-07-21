"""
Train & Compare Anomaly Detection Models
==========================================
This script trains two RandomForestClassifiers for anomaly detection:

  • Baseline  — trained on real data only
  • Augmented — trained on real + synthetic (TimeGAN-generated) data

It prints a side-by-side comparison of accuracy, precision, recall, F1-score
and confusion matrices, then saves:
  - model_baseline.pkl       (trained Baseline model)
  - model_augmented.pkl      (trained Augmented model)
  - plots/comparison_metrics.png  (bar chart comparing the two models)
"""

import warnings
warnings.filterwarnings("ignore")        # suppress noisy sklearn / numpy warnings

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Load the windowed .npy files
# ────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Loading windowed data …")
print("=" * 65)

normal_windows            = np.load("normal_windows.npy")             # real normal
anomaly_windows           = np.load("anomaly_windows.npy")            # real anomaly
synthetic_normal_windows  = np.load("synthetic_normal_windows.npy")   # synthetic normal
synthetic_anomaly_windows = np.load("synthetic_anomaly_windows.npy")  # synthetic anomaly

print(f"  Real normal windows      : {normal_windows.shape}")
print(f"  Real anomaly windows     : {anomaly_windows.shape}")
print(f"  Synthetic normal windows : {synthetic_normal_windows.shape}")
print(f"  Synthetic anomaly windows: {synthetic_anomaly_windows.shape}")

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Flatten each window (24 timesteps × 3 features) → 72-D vector
# ────────────────────────────────────────────────────────────────────────────
# Scikit-learn classifiers expect 2-D input (samples, features),
# so we reshape each (N, 24, 3) array into (N, 72).

normal_flat            = normal_windows.reshape(normal_windows.shape[0], -1)
anomaly_flat           = anomaly_windows.reshape(anomaly_windows.shape[0], -1)
syn_normal_flat        = synthetic_normal_windows.reshape(synthetic_normal_windows.shape[0], -1)
syn_anomaly_flat       = synthetic_anomaly_windows.reshape(synthetic_anomaly_windows.shape[0], -1)

print(f"\n  Flattened feature vector length: {normal_flat.shape[1]}")

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Build "Baseline" and "Augmented" datasets
# ────────────────────────────────────────────────────────────────────────────
# Labels:  0 = normal,  1 = anomaly

# --- Baseline: real data only ---
X_base = np.vstack([normal_flat, anomaly_flat])
y_base = np.concatenate([
    np.zeros(len(normal_flat)),      # 0 → normal
    np.ones(len(anomaly_flat)),      # 1 → anomaly
])

# --- Augmented: real + synthetic data ---
X_aug = np.vstack([normal_flat, syn_normal_flat, anomaly_flat, syn_anomaly_flat])
y_aug = np.concatenate([
    np.zeros(len(normal_flat) + len(syn_normal_flat)),       # 0 → normal
    np.ones(len(anomaly_flat) + len(syn_anomaly_flat)),      # 1 → anomaly
])

print(f"\n  Baseline  dataset  →  {X_base.shape[0]:,} samples  "
      f"(normal: {int((y_base == 0).sum()):,}, anomaly: {int((y_base == 1).sum()):,})")
print(f"  Augmented dataset  →  {X_aug.shape[0]:,} samples  "
      f"(normal: {int((y_aug == 0).sum()):,}, anomaly: {int((y_aug == 1).sum()):,})")

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Train / test split (80 / 20, stratified)
# ────────────────────────────────────────────────────────────────────────────
X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
)

X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)

print(f"\n  Baseline  →  train: {len(X_base_train):,}   test: {len(X_base_test):,}")
print(f"  Augmented →  train: {len(X_aug_train):,}   test: {len(X_aug_test):,}")

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Train RandomForestClassifier on each dataset
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Training models …")
print("=" * 65)

# Baseline model
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
rf_baseline.fit(X_base_train, y_base_train)
print("  ✓ Baseline  RandomForest trained")

# Augmented model
rf_augmented = RandomForestClassifier(n_estimators=100, random_state=42)
rf_augmented.fit(X_aug_train, y_aug_train)
print("  ✓ Augmented RandomForest trained")

# ────────────────────────────────────────────────────────────────────────────
# Step 6: Evaluate both models on their respective test sets
# ────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    """Return a dict of standard classification metrics."""
    y_pred = model.predict(X_test)
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
        "Confusion": confusion_matrix(y_test, y_pred),
    }

base_metrics = evaluate(rf_baseline,  X_base_test, y_base_test)
aug_metrics  = evaluate(rf_augmented, X_aug_test,  y_aug_test)

# ────────────────────────────────────────────────────────────────────────────
# Step 7: Print a clean comparison table
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Model Comparison — Baseline vs Augmented")
print("=" * 65)
print(f"  {'Metric':<12}  {'Baseline':>10}  {'Augmented':>10}")
print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}")
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    print(f"  {metric:<12}  {base_metrics[metric]:>10.4f}  {aug_metrics[metric]:>10.4f}")

print(f"\n  Baseline Confusion Matrix:")
print(f"  {base_metrics['Confusion']}")
print(f"\n  Augmented Confusion Matrix:")
print(f"  {aug_metrics['Confusion']}")

# ────────────────────────────────────────────────────────────────────────────
# Step 8: Save trained models as .pkl files
# ────────────────────────────────────────────────────────────────────────────
with open("model_baseline.pkl", "wb") as f:
    pickle.dump(rf_baseline, f)
print("\n  ✓ Saved model_baseline.pkl")

with open("model_augmented.pkl", "wb") as f:
    pickle.dump(rf_augmented, f)
print("  ✓ Saved model_augmented.pkl")

# ────────────────────────────────────────────────────────────────────────────
# Step 9: Save comparison bar chart
# ────────────────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

metrics_to_plot = ["Precision", "Recall", "F1-Score"]
base_vals = [base_metrics[m] for m in metrics_to_plot]
aug_vals  = [aug_metrics[m]  for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
bar_width = 0.30

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - bar_width / 2, base_vals, bar_width, label="Baseline",  color="#4C72B0")
bars2 = ax.bar(x + bar_width / 2, aug_vals,  bar_width, label="Augmented", color="#55A868")

# Add value labels on top of each bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Score")
ax.set_title("Anomaly Detection — Baseline vs Augmented Model")
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/comparison_metrics.png", dpi=150)
print("  ✓ Saved plots/comparison_metrics.png")

print("\n" + "=" * 65)
print("  Done! Both models trained, evaluated, and saved.")
print("=" * 65)
