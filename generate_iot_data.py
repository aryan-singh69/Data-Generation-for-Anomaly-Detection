import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42                       # For reproducibility
DAYS = 30                       # Duration of simulation
SAMPLE_INTERVAL_MIN = 1         # One reading per minute
NUM_ANOMALY_WINDOWS = 18        # Number of anomaly windows (15-20 range)
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

# ── Step 3: Inject SUBTLE anomaly windows ───────────────────────────────────
# Anomalies deviate only 1.5x–2.5x above the normal noise level, and each
# window affects a random subset of sensors (1, 2, or all 3). This creates
# realistic overlap between normal and anomaly distributions so a classifier
# can't trivially separate them without enough training data.
labels = np.zeros(total_minutes, dtype=int)             # 0 = normal everywhere

# Noise-relative deviation ranges  (multiplier × sensor noise_std)
# A 2× multiplier means the shift is only twice the normal noise — subtle!
SUBTLE_SHIFT = {
    "temperature": SENSOR_PARAMS["temperature"]["noise_std"],  # 0.5
    "vibration":   SENSOR_PARAMS["vibration"]["noise_std"],    # 0.02
    "pressure":    SENSOR_PARAMS["pressure"]["noise_std"],     # 0.3
}

anomaly_starts = np.random.randint(0, total_minutes - ANOMALY_MAX_LEN,
                                   size=NUM_ANOMALY_WINDOWS)
anomaly_lengths = np.random.randint(ANOMALY_MIN_LEN, ANOMALY_MAX_LEN + 1,
                                    size=NUM_ANOMALY_WINDOWS)

sensor_arrays = {"temperature": temperature,
                 "vibration":   vibration,
                 "pressure":    pressure}

for start, length in zip(anomaly_starts, anomaly_lengths):
    end = min(start + length, total_minutes)            # Don't exceed array
    labels[start:end] = 1                               # Mark as anomaly
    n = end - start

    # Randomly choose which sensors are affected (1, 2, or 3)
    num_affected = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
    affected = np.random.choice(list(sensor_arrays.keys()),
                                size=num_affected, replace=False)

    # Choose a random fault type for variety
    fault_type = np.random.choice(["spike", "drop", "erratic"])

    for sensor_name in affected:
        sigma = SUBTLE_SHIFT[sensor_name]               # Normal noise level
        arr = sensor_arrays[sensor_name]

        # Shift magnitude: 1.5× – 2.5× the normal noise standard deviation
        multiplier = np.random.uniform(1.5, 2.5)

        if fault_type == "spike":
            arr[start:end] += sigma * multiplier
        elif fault_type == "drop":
            arr[start:end] -= sigma * multiplier
        else:  # erratic — per-point random jitter at 1.5-2.5× noise
            arr[start:end] += np.random.normal(0, sigma * multiplier, n)

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
