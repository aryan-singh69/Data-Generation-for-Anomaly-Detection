"""
Exploratory Data Analysis (EDA) for IoT Sensor Dataset
=======================================================
This script loads iot_sensor_data.csv and produces:
  1. Basic dataset info (shape, missing values, class distribution)
  2. Time-series plots with anomaly regions shaded in red
  3. Correlation heatmap between the 3 sensor signals
  4. Descriptive statistics split by normal vs anomaly rows
  5. All plots saved as PNGs in the plots/ folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# ────────────────────────────────────────────────────────────────────────────
# Step 0: Setup — create the output folder for plots
# ────────────────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# Use a clean style for all plots
sns.set_style("whitegrid")

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Load the CSV and parse timestamps
# ────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("iot_sensor_data.csv", parse_dates=["timestamp"])

print("=" * 60)
print("STEP 1 — Dataset loaded")
print("=" * 60)

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Print basic info
# ────────────────────────────────────────────────────────────────────────────

# 2a. Shape
print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 2b. Data types
print(f"\nColumn types:\n{df.dtypes.to_string()}")

# 2c. Missing values
missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing.to_string()}")

# 2d. Class distribution (label = 0 normal, label = 1 anomaly)
label_counts = df["label"].value_counts().sort_index()
label_pcts   = df["label"].value_counts(normalize=True).sort_index() * 100

print("\nClass distribution:")
print(f"  Normal  (0): {label_counts[0]:>6,} rows  ({label_pcts[0]:.1f}%)")
print(f"  Anomaly (1): {label_counts[1]:>6,} rows  ({label_pcts[1]:.1f}%)")

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Time-series plots with anomaly regions highlighted in red
# ────────────────────────────────────────────────────────────────────────────

# Helper: find contiguous anomaly windows so we can shade them
def get_anomaly_windows(labels):
    """Return list of (start_idx, end_idx) for contiguous label==1 regions."""
    windows = []
    in_anomaly = False
    for i, val in enumerate(labels):
        if val == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif val == 0 and in_anomaly:
            windows.append((start, i - 1))
            in_anomaly = False
    if in_anomaly:                         # Handle anomaly at the very end
        windows.append((start, len(labels) - 1))
    return windows

anomaly_windows = get_anomaly_windows(df["label"].values)

# The 3 sensor columns we want to plot
sensors = ["temperature", "vibration", "pressure"]
units   = ["°F", "mm/s", "PSI"]

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

for ax, sensor, unit in zip(axes, sensors, units):
    # Draw the signal line
    ax.plot(df["timestamp"], df[sensor], linewidth=0.4,
            color="#1f77b4", label=sensor.capitalize())

    # Shade each anomaly window in red
    for start, end in anomaly_windows:
        ax.axvspan(df["timestamp"].iloc[start],
                   df["timestamp"].iloc[end],
                   color="red", alpha=0.25, label="_nolegend_")

    ax.set_ylabel(f"{sensor.capitalize()} ({unit})", fontsize=11)
    ax.legend(loc="upper right")

# Format the shared x-axis with readable dates
axes[-1].set_xlabel("Time", fontsize=12)
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.autofmt_xdate(rotation=30)

fig.suptitle("IoT Sensor Readings Over 30 Days  (red = anomaly windows)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/timeseries_with_anomalies.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: plots/timeseries_with_anomalies.png")

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Correlation heatmap
# ────────────────────────────────────────────────────────────────────────────
corr = df[sensors].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
            linewidths=1, square=True, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Sensor Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: plots/correlation_heatmap.png")

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Descriptive statistics — Normal vs Anomaly
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Descriptive Statistics (Normal vs Anomaly)")
print("=" * 60)

normal_df  = df[df["label"] == 0][sensors]
anomaly_df = df[df["label"] == 1][sensors]

# Choose the stats we care about
stats_to_show = ["mean", "std", "min", "max"]

print("\n── Normal rows ────────────────────────────────────────────")
print(normal_df.describe().loc[stats_to_show].round(3).to_string())

print("\n── Anomaly rows ───────────────────────────────────────────")
print(anomaly_df.describe().loc[stats_to_show].round(3).to_string())

# ────────────────────────────────────────────────────────────────────────────
# Bonus: Box-plots comparing Normal vs Anomaly for each sensor
# ────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, sensor, unit in zip(axes, sensors, units):
    sns.boxplot(x="label", y=sensor, hue="label", data=df, ax=ax,
                palette=["#4CAF50", "#F44336"], width=0.5, legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("")
    ax.set_ylabel(f"{sensor.capitalize()} ({unit})", fontsize=11)
    ax.set_title(sensor.capitalize(), fontsize=12, fontweight="bold")

fig.suptitle("Sensor Distributions: Normal vs Anomaly",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/boxplots_normal_vs_anomaly.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: plots/boxplots_normal_vs_anomaly.png")

print("\n✅ All done! Check the plots/ folder for saved figures.")
