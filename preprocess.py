"""
Preprocessing for TimeGAN Training
====================================
This script takes the raw IoT sensor CSV and produces windowed NumPy arrays
ready for TimeGAN:

  1. Drop timestamp, scale features with MinMaxScaler (saved as scaler.pkl)
  2. Split into normal (label=0) and anomaly (label=1) groups
  3. Create sliding windows of length 24 (step=1) for each group
  4. Save as normal_windows.npy and anomaly_windows.npy
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Load the CSV and separate the timestamp
# ────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("iot_sensor_data.csv", parse_dates=["timestamp"])

# Save timestamp separately (useful later for plotting generated data)
timestamps = df["timestamp"]
timestamps.to_csv("timestamps.csv", index=False)
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Scale features to [0, 1] using MinMaxScaler
# ────────────────────────────────────────────────────────────────────────────
feature_cols = ["temperature", "vibration", "pressure"]

scaler = MinMaxScaler(feature_range=(0, 1))
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Save the fitted scaler so we can inverse-transform generated data later
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaled features to [0, 1] and saved scaler → scaler.pkl")
print(f"  Min values after scaling: {df[feature_cols].min().values}")
print(f"  Max values after scaling: {df[feature_cols].max().values}")

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Split into normal and anomaly groups
# ────────────────────────────────────────────────────────────────────────────
normal_df  = df[df["label"] == 0][feature_cols].reset_index(drop=True)
anomaly_df = df[df["label"] == 1][feature_cols].reset_index(drop=True)

print(f"\nNormal  rows: {len(normal_df):,}")
print(f"Anomaly rows: {len(anomaly_df):,}")

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Create sliding windows
#   - Window length : 24 time steps (24 consecutive minutes)
#   - Step size     : 1 (slide one step at a time)
#   - Output shape  : (num_windows, 24, 3)
# ────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 24   # Number of consecutive time steps per window
STEP    = 1    # How many steps to slide forward each time

def create_windows(data, seq_len, step):
    """
    Slide a window of `seq_len` rows over `data` with the given `step`.
    Returns a 3-D NumPy array of shape (num_windows, seq_len, num_features).
    """
    windows = []
    for start in range(0, len(data) - seq_len + 1, step):
        window = data[start : start + seq_len]    # shape: (seq_len, 3)
        windows.append(window)
    return np.array(windows)

# Convert DataFrames to NumPy arrays before windowing
normal_windows  = create_windows(normal_df.values,  SEQ_LEN, STEP)
anomaly_windows = create_windows(anomaly_df.values, SEQ_LEN, STEP)

print(f"\n── Sliding-window results (seq_len={SEQ_LEN}, step={STEP}) ──")
print(f"  Normal  windows: {normal_windows.shape}")
print(f"  Anomaly windows: {anomaly_windows.shape}")

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Save windowed arrays as .npy files
# ────────────────────────────────────────────────────────────────────────────
np.save("normal_windows.npy",  normal_windows)
np.save("anomaly_windows.npy", anomaly_windows)

print("\n✅ Saved: normal_windows.npy")
print("✅ Saved: anomaly_windows.npy")
print("✅ Saved: scaler.pkl")
print("\nPreprocessing complete — data is ready for TimeGAN training!")
