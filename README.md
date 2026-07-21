 # Synthetic Time-Series Data Generation for Anomaly Detection using TimeGAN (IoT/Industry)

> **6-Week AI/ML Internship Project**

## Objective

This project uses **TimeGAN** (Time-series Generative Adversarial Network) to generate realistic synthetic IoT sensor data and augment a small, imbalanced dataset for anomaly detection. The core idea is simple: real-world anomaly data is rare, so we train a generative model to create *more* of it synthetically. By combining real + synthetic data, we aim to improve the performance of downstream anomaly detection models — especially recall and F1-score on the minority (anomaly) class.

## Dataset

The dataset simulates a **factory machine with 3 sensors** recording once per minute over **30 days** (~43,200 data points):

| Sensor        | Unit | Normal Behaviour            |
|---------------|------|-----------------------------|
| Temperature   | °C   | Smooth sine wave + noise    |
| Vibration     | mm/s | Smooth sine wave + noise    |
| Pressure      | psi  | Smooth sine wave + noise    |

**Anomaly injection:** Random short windows (bursts of abnormal spikes or drops) are injected into the normal signal to simulate real equipment faults. Each row has a binary `label` column — `0` for normal, `1` for anomaly. Anomalies make up a small fraction of the total data (class imbalance).

## Tech Stack

| Tool / Library       | Purpose                                       |
|----------------------|-----------------------------------------------|
| Python 3.10+         | Core language                                 |
| pandas               | Data loading and manipulation                 |
| NumPy                | Numerical arrays, windowing                   |
| scikit-learn         | MinMaxScaler, train/test split, metrics        |
| ydata-synthetic      | TimeGAN implementation                        |
| TensorFlow 2.x       | Backend for TimeGAN (installed with ydata)    |
| matplotlib           | Time-series and comparison plots              |
| seaborn              | Heatmaps, box plots, styled visualisations    |
| joblib / pickle      | Model and scaler serialisation                |

## Project Workflow

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6
```

| Step | Script                      | Description                                                                 |
|------|-----------------------------|-----------------------------------------------------------------------------|
| 1    | `generate_iot_data.py`      | Simulate 30 days of IoT sensor data with injected anomalies → `iot_sensor_data.csv` |
| 2    | `eda.py`                    | Exploratory data analysis: plots, stats, class distribution → `plots/`     |
| 3    | `preprocess.py`             | Scale features, create sliding windows (length 24) → `.npy` files          |
| 4    | `train_timegan.py`          | Train TimeGAN on **normal** windows → 500 synthetic normal sequences       |
| 5    | `train_timegan_anomaly.py`  | Train TimeGAN on **anomaly** windows → 500 synthetic anomaly sequences     |
| 6    | `train_model.py`            | Train & compare baseline vs augmented anomaly detectors *(upcoming)*       |

## Folder Structure

```
Data-Generation-for-Anomaly-Detection/
│
├── generate_iot_data.py           # Step 1: Simulate IoT sensor data
├── eda.py                         # Step 2: Exploratory data analysis
├── preprocess.py                  # Step 3: Scaling + sliding windows
├── train_timegan.py               # Step 4: TimeGAN on normal data
├── train_timegan_anomaly.py       # Step 5: TimeGAN on anomaly data
├── train_model.py                 # Step 6: Model training & comparison (upcoming)
│
├── iot_sensor_data.csv            # Raw simulated dataset
├── timestamps.csv                 # Extracted timestamps
├── normal_windows.npy             # Preprocessed normal windows (N, 24, 3)
├── anomaly_windows.npy            # Preprocessed anomaly windows (N, 24, 3)
├── scaler.pkl                     # Fitted MinMaxScaler
│
├── synthetic_normal_windows.npy   # Generated after Step 4
├── synthetic_anomaly_windows.npy  # Generated after Step 5
│
├── timegan_model.pkl              # Saved TimeGAN (normal)
├── timegan_anomaly_model/         # Saved TimeGAN (anomaly)
│   └── timegan_anomaly.pkl
│
├── plots/                         # EDA visualisations
│   ├── timeseries_with_anomalies.png
│   ├── boxplots_normal_vs_anomaly.png
│   └── correlation_heatmap.png
│
└── README.md                      # You are here
```

## Results

> **Note:** Fill in the table below with actual numbers after running `train_model.py`.

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Baseline      | —        | —         | —      | —        |
| + Augmented   | —        | —         | —      | —        |

**Baseline** = trained on real data only.
**Augmented** = trained on real + synthetic (TimeGAN-generated) data.

The key metric to watch is **Recall** and **F1-Score** on the anomaly class — these should improve with synthetic augmentation since the model sees more examples of the rare anomaly patterns.

## How to Run

### 1. Set up the environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn ydata-synthetic joblib
```

### 2. Run scripts in order

```bash
# Step 1: Generate the simulated IoT dataset
python generate_iot_data.py

# Step 2: Run exploratory data analysis (saves plots to plots/)
python eda.py

# Step 3: Preprocess data into sliding windows
python preprocess.py

# Step 4: Train TimeGAN on normal windows (~5-15 min depending on hardware)
python train_timegan.py

# Step 5: Train TimeGAN on anomaly windows
python train_timegan_anomaly.py

# Step 6: Train and compare anomaly detection models (upcoming)
python train_model.py
```

> **Tip:** Steps 4 and 5 (TimeGAN training) are the most time-consuming. On a CPU-only machine, each can take 5–15 minutes. A GPU (TensorFlow with CUDA) speeds this up significantly.

## Key Learnings / Conclusion

- **Synthetic data augmentation helps with class imbalance.** By generating realistic synthetic anomaly sequences with TimeGAN, the anomaly detection model gets more examples of the rare class to learn from, improving recall and F1-score.

- **GANs need sufficient training data.** TimeGAN works well on the normal data (thousands of windows), but struggles with very small anomaly datasets (dozens of windows). The quality of synthetic data is directly tied to how much real data the GAN has to learn from.

- **TimeGAN preserves temporal dynamics.** Unlike simpler augmentation methods (e.g., random noise injection or SMOTE), TimeGAN learns the *temporal correlations* between time steps, producing sequences that respect the underlying time-series structure.

- **End-to-end pipeline matters.** Building the full pipeline — from data generation through EDA, preprocessing, synthetic generation, to model evaluation — gave hands-on experience with a realistic ML workflow, not just isolated model training.

---

*Built as part of a 6-week AI/ML internship project. Feedback and contributions welcome!*
