"""
Generate Synthetic IoT Sensor Data for Anomaly Detection
=========================================================
This script simulates a machine with 3 sensors (temperature, vibration,
pressure) recording every minute for 30 days. Normal behaviour follows
smooth sine-wave patterns with small noise; anomalies are randomly
injected as short windows of abnormal spikes or drops.

Output: iot_sensor_data.csv
Columns: timestamp, temperature, vibration, pressure, label (0/1)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42                       # For reproducibility
DAYS = 30                       # Duration of simulation
SAMPLE_INTERVAL_MIN = 1         # One reading per minute
NUM_ANOMALY_WINDOWS = 25        # Number of anomaly windows (between 20-30)
ANOMALY_MIN_LEN = 10            # Shortest anomaly window (minutes)
ANOMALY_MAX_LEN = 60            # Longest  anomaly window (minutes)

# Sensor baseline parameters (centre value, amplitude of daily cycle, noise σ)
SENSOR_PARAMS = {
    "temperature": {"center": 70,  "amplitude": 5,   "noise_std": 0.5},
    "vibration":   {"center": 0.5, "amplitude": 0.1,  "noise_std": 0.02},
    "pressure":    {"center": 30,  "amplitude": 2,    "noise_std": 0.3},
}

# ── Step 1: Create timestamps ───────────────────────────────────────────────
np.random.seed(SEED)

total_minutes = DAYS * 24 * 60                         # 30 days in minutes
start_time = datetime(2026, 1, 1)                      # Arbitrary start date
timestamps = [start_time + timedelta(minutes=i) for i in range(total_minutes)]

print(f"Generating {total_minutes:,} data points ({DAYS} days × 1440 min/day)…")

# ── Step 2: Generate normal (healthy) sensor readings ────────────────────────
# Each signal = slow sine wave (daily cycle) + tiny random noise
time_index = np.arange(total_minutes)                  # 0, 1, 2, … minutes

def generate_normal_signal(params):
    """Return a smooth, slightly noisy signal simulating healthy behaviour."""
    daily_cycle = params["amplitude"] * np.sin(2 * np.pi * time_index / 1440)
    noise = np.random.normal(0, params["noise_std"], total_minutes)
    return params["center"] + daily_cycle + noise

temperature = generate_normal_signal(SENSOR_PARAMS["temperature"])
vibration   = generate_normal_signal(SENSOR_PARAMS["vibration"])
pressure    = generate_normal_signal(SENSOR_PARAMS["pressure"])

# ── Step 3: Inject anomaly windows ──────────────────────────────────────────
# Pick random start positions and lengths, then distort readings in those windows.
labels = np.zeros(total_minutes, dtype=int)             # 0 = normal everywhere

anomaly_starts = np.random.randint(0, total_minutes - ANOMALY_MAX_LEN,
                                   size=NUM_ANOMALY_WINDOWS)
anomaly_lengths = np.random.randint(ANOMALY_MIN_LEN, ANOMALY_MAX_LEN + 1,
                                    size=NUM_ANOMALY_WINDOWS)

for start, length in zip(anomaly_starts, anomaly_lengths):
    end = min(start + length, total_minutes)            # Don't exceed array
    labels[start:end] = 1                               # Mark as anomaly

    # Choose a random fault type for variety
    fault_type = np.random.choice(["spike", "drop", "erratic"])

    if fault_type == "spike":
        # Sudden upward shift
        temperature[start:end] += np.random.uniform(10, 25)
        vibration[start:end]   += np.random.uniform(0.3, 0.8)
        pressure[start:end]    += np.random.uniform(5, 15)

    elif fault_type == "drop":
        # Sudden downward shift
        temperature[start:end] -= np.random.uniform(10, 20)
        vibration[start:end]   -= np.random.uniform(0.2, 0.5)
        pressure[start:end]    -= np.random.uniform(5, 12)

    else:  # erratic
        # Wild random fluctuations
        n = end - start
        temperature[start:end] += np.random.normal(0, 8, n)
        vibration[start:end]   += np.random.normal(0, 0.4, n)
        pressure[start:end]    += np.random.normal(0, 6, n)

# ── Step 4: Build DataFrame and save to CSV ─────────────────────────────────
df = pd.DataFrame({
    "timestamp":   timestamps,
    "temperature": np.round(temperature, 2),
    "vibration":   np.round(vibration, 4),
    "pressure":    np.round(pressure, 2),
    "label":       labels,
})

output_file = "iot_sensor_data.csv"
df.to_csv(output_file, index=False)

# ── Step 5: Print summary statistics ────────────────────────────────────────
normal_count  = (df["label"] == 0).sum()
anomaly_count = (df["label"] == 1).sum()

print(f"\n✅ Dataset saved to '{output_file}'")
print(f"   Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Normal  rows   : {normal_count:,}  ({normal_count/len(df)*100:.1f}%)")
print(f"   Anomaly rows   : {anomaly_count:,}  ({anomaly_count/len(df)*100:.1f}%)")
print(f"   Anomaly windows: {NUM_ANOMALY_WINDOWS}")
print(f"\nFirst 5 rows:")
print(df.head().to_string(index=False))
