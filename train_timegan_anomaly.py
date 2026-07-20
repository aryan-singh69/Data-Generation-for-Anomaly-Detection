"""
Train TimeGAN on Anomaly IoT Sensor Windows
=============================================
This script uses the ydata-synthetic library to train a *second* TimeGAN
model — this time on the preprocessed **anomaly** sensor windows — then
generates 500 synthetic anomaly sequences.

Because anomalies are rare, the anomaly dataset will usually be *much*
smaller than the normal dataset. The script prints a clear warning when
the dataset might be too small for stable TimeGAN training, but still
attempts to train anyway.

Prerequisites:
    pip install ydata-synthetic

Input : anomaly_windows.npy  (shape: num_windows, 24, 3)
Output: synthetic_anomaly_windows.npy, timegan_anomaly_model/ folder
"""

import sys
import os
import warnings
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Step 0: Check that ydata-synthetic is installed
# ────────────────────────────────────────────────────────────────────────────
try:
    # Suppress the deprecation warning from ydata-synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
        from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    print("✅ ydata-synthetic imported successfully.\n")
except ImportError as e:
    print("=" * 60)
    print("❌ ERROR: Could not import ydata-synthetic.")
    print("=" * 60)
    print(f"\nDetails: {e}\n")
    print("To fix this, run:")
    print("    pip install ydata-synthetic")
    print("\nydata-synthetic requires TensorFlow 2.x. If TensorFlow is")
    print("missing, pip will install it automatically. On Windows with")
    print("Python 3.10 this usually works out of the box.\n")
    print("If you keep seeing errors, try creating a fresh venv:")
    print("    python -m venv venv")
    print("    venv\\Scripts\\activate")
    print("    pip install ydata-synthetic")
    sys.exit(1)

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Load the preprocessed anomaly windows
# ────────────────────────────────────────────────────────────────────────────
data_path = "anomaly_windows.npy"
if not os.path.exists(data_path):
    print(f"❌ File not found: {data_path}")
    print("   Run preprocess.py first to generate this file.")
    sys.exit(1)

anomaly_windows = np.load(data_path)
print(f"Loaded {data_path}")
print(f"  Shape: {anomaly_windows.shape}")
print(f"  (num_windows={anomaly_windows.shape[0]}, "
      f"seq_len={anomaly_windows.shape[1]}, "
      f"n_features={anomaly_windows.shape[2]})\n")

# ────────────────────────────────────────────────────────────────────────────
# Step 1b: Warn if the anomaly dataset is very small
# ────────────────────────────────────────────────────────────────────────────
# TimeGAN (like most GANs) struggles when it has very few training samples.
# We check here and print a clear warning so the user knows what to expect.
MIN_RECOMMENDED = 100  # Rough guideline; fewer samples → noisier results

num_windows = anomaly_windows.shape[0]
if num_windows < MIN_RECOMMENDED:
    print("=" * 60)
    print("⚠️  WARNING: Very small anomaly dataset detected!")
    print("=" * 60)
    print(f"  Number of anomaly windows : {num_windows}")
    print(f"  Recommended minimum       : {MIN_RECOMMENDED}")
    print()
    print("  TimeGAN (and GANs in general) need a reasonable number of")
    print("  training samples to learn meaningful patterns. With very")
    print("  few samples the model may:")
    print("    • Overfit to the training data (memorisation)")
    print("    • Produce low-quality / noisy synthetic sequences")
    print("    • Show unstable training losses")
    print()
    print("  The script will still attempt training, but be aware that")
    print("  the synthetic output quality may be limited.")
    print("  Consider collecting more anomaly data or using data")
    print("  augmentation techniques to improve results.")
    print("=" * 60)
    print()

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Configure TimeGAN hyper-parameters
# ────────────────────────────────────────────────────────────────────────────
# We use the same architecture as the normal model but adapt the batch size
# to the smaller anomaly dataset.  If the dataset is smaller than the default
# batch size, we reduce it to avoid empty batches.
SEQ_LEN    = anomaly_windows.shape[1]  # 24 time steps
N_FEATURES = anomaly_windows.shape[2]  # 3 features (temp, vibration, pressure)
HIDDEN_DIM = 24                        # Hidden / latent dimension for GAN nets
GAMMA      = 1                         # Weight for the supervised loss
NOISE_DIM  = 32                        # Dimension of the random noise input
LAYERS_DIM = 128                       # Units per layer
LR         = 5e-4                      # Learning rate (Adam)
EPOCHS     = 200                       # Training epochs (200 — small dataset)

# Adaptive batch size: use 128 for large datasets, but clamp to the number
# of available windows so we never request a batch bigger than the dataset.
BATCH_SIZE = min(128, num_windows)

print("── TimeGAN Configuration (Anomaly) ──")
print(f"  Sequence length : {SEQ_LEN}")
print(f"  Num features    : {N_FEATURES}")
print(f"  Hidden dim      : {HIDDEN_DIM}")
print(f"  Batch size      : {BATCH_SIZE}  (adapted to dataset size)")
print(f"  Learning rate   : {LR}")
print(f"  Epochs          : {EPOCHS}")
print()

# ModelParameters is a namedtuple with fields:
#   batch_size, lr, betas, layers_dim, noise_dim, ..., latent_dim, ..., gamma
gan_args = ModelParameters(
    batch_size=BATCH_SIZE,
    lr=LR,
    noise_dim=NOISE_DIM,
    layers_dim=LAYERS_DIM,
    latent_dim=HIDDEN_DIM,
    gamma=GAMMA
)

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Build and train TimeGAN on anomaly data
# ────────────────────────────────────────────────────────────────────────────
print("Initialising TimeGAN model (anomaly)…")

# TimeSeriesSynthesizer is a factory: passing modelname='timegan'
# returns a TimeGAN instance internally.
timegan = TimeSeriesSynthesizer(
    modelname="timegan",
    model_parameters=gan_args
)

# Set sequence length and number of features so the model knows the shape.
timegan.seq_len = SEQ_LEN
timegan.n_seq   = N_FEATURES

print("Starting training — this may take a while…\n")
print("TimeGAN trains in 3 phases:")
print("  Phase 1: Embedding network (autoencoder)")
print("  Phase 2: Supervised network")
print("  Phase 3: Joint adversarial training\n")

timegan.train(data=anomaly_windows, train_steps=EPOCHS)
print("\n✅ Training complete!\n")

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Generate 500 synthetic anomaly sequences
# ────────────────────────────────────────────────────────────────────────────
N_SYNTHETIC = 500
print(f"Generating {N_SYNTHETIC} synthetic anomaly sequences…")
synthetic_data_list = timegan.sample(N_SYNTHETIC)

# timegan.sample() returns a list of DataFrames; stack into a 3-D array
synthetic_data = np.array([df.values for df in synthetic_data_list])
print(f"  Generated shape: {synthetic_data.shape}")
print(f"  Expected shape : ({N_SYNTHETIC}, {SEQ_LEN}, {N_FEATURES})")

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Save generated data and model
# ────────────────────────────────────────────────────────────────────────────

# 5a. Save synthetic anomaly sequences
output_file = "synthetic_anomaly_windows.npy"
np.save(output_file, synthetic_data)
print(f"\n✅ Saved synthetic anomaly data → {output_file}")

# 5b. Save the trained model to the timegan_anomaly_model/ folder
model_dir = "timegan_anomaly_model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "timegan_anomaly.pkl")
from joblib import dump
dump(timegan, model_path)
print(f"✅ Saved trained anomaly model → {model_path}")

# ────────────────────────────────────────────────────────────────────────────
# Step 6: Quick sanity check
# ────────────────────────────────────────────────────────────────────────────
print("\n── Sanity Check ──")
print(f"  Real anomaly data  — min: {anomaly_windows.min():.4f}, "
      f"max: {anomaly_windows.max():.4f}, "
      f"mean: {anomaly_windows.mean():.4f}")
print(f"  Synth anomaly data — min: {synthetic_data.min():.4f}, "
      f"max: {synthetic_data.max():.4f}, "
      f"mean: {synthetic_data.mean():.4f}")

# Print a final summary of everything that was saved
print("\n── Summary ──")
print(f"  Real anomaly windows   : {anomaly_windows.shape}")
print(f"  Synthetic anomaly data : {synthetic_data.shape}")
print(f"  Saved synthetic data   : {output_file}")
print(f"  Saved model            : {model_path}")
print("\n✅ All done! Synthetic anomaly data is ready for evaluation.")
