"""
Train TimeGAN on Normal IoT Sensor Windows
============================================
This script uses the ydata-synthetic library to train a TimeGAN model
on the preprocessed normal sensor windows, then generates 500 synthetic
normal sequences.

Prerequisites:
    pip install ydata-synthetic

Input : normal_windows.npy  (shape: num_windows, 24, 3)
Output: synthetic_normal_windows.npy, timegan_model/ folder
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
# Step 1: Load the preprocessed normal windows
# ────────────────────────────────────────────────────────────────────────────
data_path = "normal_windows.npy"
if not os.path.exists(data_path):
    print(f"❌ File not found: {data_path}")
    print("   Run preprocess.py first to generate this file.")
    sys.exit(1)

normal_windows = np.load(data_path)
print(f"Loaded {data_path}")
print(f"  Shape: {normal_windows.shape}")
print(f"  (num_windows={normal_windows.shape[0]}, "
      f"seq_len={normal_windows.shape[1]}, "
      f"n_features={normal_windows.shape[2]})\n")

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Configure TimeGAN hyper-parameters
# ────────────────────────────────────────────────────────────────────────────
SEQ_LEN    = normal_windows.shape[1]   # 24 time steps
N_FEATURES = normal_windows.shape[2]   # 3 features (temp, vibration, pressure)
HIDDEN_DIM = 24                        # Hidden / latent dimension for GAN nets
GAMMA      = 1                         # Weight for the supervised loss
NOISE_DIM  = 32                        # Dimension of the random noise input
LAYERS_DIM = 128                       # Units per layer
BATCH_SIZE = 128                       # Training batch size
LR         = 5e-4                      # Learning rate (Adam)
EPOCHS     = 200                       # Training epochs (200 for quick demo)

print("── TimeGAN Configuration ──")
print(f"  Sequence length : {SEQ_LEN}")
print(f"  Num features    : {N_FEATURES}")
print(f"  Hidden dim      : {HIDDEN_DIM}")
print(f"  Batch size      : {BATCH_SIZE}")
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
# Step 3: Build and train TimeGAN
# ────────────────────────────────────────────────────────────────────────────
print("Initialising TimeGAN model…")

# TimeSeriesSynthesizer is a factory: passing modelname='timegan'
# returns a TimeGAN instance internally.
timegan = TimeSeriesSynthesizer(
    modelname="timegan",
    model_parameters=gan_args
)

# The low-level train() method expects:
#   data  : 3-D numpy array (num_windows, seq_len, n_features)
#   train_steps : number of epochs
# It sets seq_len and n_seq from the data shape automatically.
timegan.seq_len = SEQ_LEN
timegan.n_seq   = N_FEATURES

print("Starting training — this will take a while…\n")
print("TimeGAN trains in 3 phases:")
print("  Phase 1: Embedding network (autoencoder)")
print("  Phase 2: Supervised network")
print("  Phase 3: Joint adversarial training\n")

timegan.train(data=normal_windows, train_steps=EPOCHS)
print("\n✅ Training complete!\n")

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Generate 500 synthetic normal sequences
# ────────────────────────────────────────────────────────────────────────────
N_SYNTHETIC = 500
print(f"Generating {N_SYNTHETIC} synthetic normal sequences…")
synthetic_data_list = timegan.sample(N_SYNTHETIC)

# timegan.sample() returns a list of DataFrames; stack into a 3-D array
synthetic_data = np.array([df.values for df in synthetic_data_list])
print(f"  Generated shape: {synthetic_data.shape}")
print(f"  Expected shape : ({N_SYNTHETIC}, {SEQ_LEN}, {N_FEATURES})")

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Save generated data and model
# ────────────────────────────────────────────────────────────────────────────

# Save synthetic sequences
output_file = "synthetic_normal_windows.npy"
np.save(output_file, synthetic_data)
print(f"\n✅ Saved synthetic data → {output_file}")

# Save the trained model (joblib pickle)
model_path = "timegan_model.pkl"
from joblib import dump
dump(timegan, model_path)
print(f"✅ Saved trained model → {model_path}")

# ────────────────────────────────────────────────────────────────────────────
# Step 6: Quick sanity check
# ────────────────────────────────────────────────────────────────────────────
print("\n── Sanity Check ──")
print(f"  Real data  — min: {normal_windows.min():.4f}, "
      f"max: {normal_windows.max():.4f}, "
      f"mean: {normal_windows.mean():.4f}")
print(f"  Synth data — min: {synthetic_data.min():.4f}, "
      f"max: {synthetic_data.max():.4f}, "
      f"mean: {synthetic_data.mean():.4f}")
print("\n✅ All done! Synthetic normal data is ready for evaluation.")
