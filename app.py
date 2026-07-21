"""
IoT Anomaly Detection — Streamlit Dashboard
============================================
A professional analytics tool for the TimeGAN-based synthetic data
augmentation project.  Three pages:

  1. Overview          — KPIs, pipeline flow, and EDA plots
  2. Model Performance — Baseline vs Augmented metrics, confusion matrices
  3. Live Prediction   — Upload sensor CSV → detect anomalies → export results

Run with:
    streamlit run app.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be the very first Streamlit command
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IoT Anomaly Detection — Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# PATH HELPER  — resolve file paths relative to this script
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(relative_path):
    """Return the absolute path for a file relative to the project root."""
    return os.path.join(BASE_DIR, relative_path)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — injected once to style the entire app
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global typography ───────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, 'Helvetica Neue', sans-serif;
}

/* ── Main container ──────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Hide default Streamlit branding ─────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}

/* ── Dark sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #cbd5e1;
}

/* ── KPI metric card ─────────────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid #e2e8f0;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.10);
}
.mc-icon  { font-size: 1.4rem; margin-bottom: 2px; }
.mc-value {
    font-size: 1.7rem; font-weight: 700; color: #0d9488; line-height: 1.3;
}
.mc-value.accent-blue   { color: #0ea5e9; }
.mc-value.accent-purple { color: #7c3aed; }
.mc-value.accent-green  { color: #059669; }
.mc-label {
    font-size: 0.73rem; font-weight: 600; color: #64748b;
    margin-top: 5px; text-transform: uppercase; letter-spacing: 0.06em;
}

/* ── Pipeline step card ──────────────────────────────────────────────── */
.step-card {
    background: linear-gradient(135deg, #f0fdfa 0%, #ffffff 100%);
    border-radius: 14px;
    padding: 22px 14px;
    border: 1px solid #ccfbf1;
    text-align: center;
    min-height: 175px;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px; border-radius: 50%;
    background: #0d9488; color: #fff; font-weight: 700; font-size: 0.8rem;
}
.step-icon  { font-size: 1.6rem; margin: 6px 0 4px 0; }
.step-title { font-weight: 600; color: #0f172a; font-size: 0.9rem; }
.step-desc  { font-size: 0.78rem; color: #64748b; line-height: 1.45; margin-top: 4px; }

/* ── Callout / takeaway box ──────────────────────────────────────────── */
.callout-box {
    background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
    border-left: 4px solid #0d9488;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 10px 0;
}
.callout-title { font-weight: 600; color: #0d9488; margin-bottom: 4px; }
.callout-text  { color: #334155; font-size: 0.9rem; line-height: 1.5; }

/* ── Confusion-matrix table ──────────────────────────────────────────── */
.cm-table { border-collapse: collapse; margin: 0 auto; font-size: 0.85rem; }
.cm-table th, .cm-table td {
    padding: 10px 20px; text-align: center; border: 1px solid #e2e8f0;
}
.cm-table th { background: #f8fafc; font-weight: 600; color: #334155; }
.cm-correct  { background: #ecfdf5; color: #065f46; font-weight: 600; }
.cm-error    { background: #fef2f2; color: #991b1b; font-weight: 600; }

/* ── Page header helpers ─────────────────────────────────────────────── */
.page-title {
    font-size: 1.65rem; font-weight: 700; color: #0f172a; margin-bottom: 0;
}
.page-sub {
    font-size: 0.95rem; color: #64748b; margin-bottom: 1.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# CACHED LOADERS  — models and scaler loaded once, then reused
#
# Each loader returns (object, None) on success or (None, error_string) on
# failure.  This prevents the app from crashing when files are missing or
# when there is a sklearn version mismatch on the deployment server.
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_augmented_model():
    """Load the trained Augmented RandomForest from model_augmented.pkl."""
    model_path = _path("model_augmented.pkl")
    if not os.path.exists(model_path):
        return None, "model_augmented.pkl not found."
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, f"Failed to load model_augmented.pkl: {e}"


@st.cache_resource
def load_baseline_model():
    """Load the trained Baseline RandomForest from model_baseline.pkl."""
    model_path = _path("model_baseline.pkl")
    if not os.path.exists(model_path):
        return None, "model_baseline.pkl not found."
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, f"Failed to load model_baseline.pkl: {e}"


@st.cache_resource
def load_scaler():
    """Load the fitted MinMaxScaler from scaler.pkl."""
    scaler_path = _path("scaler.pkl")
    if not os.path.exists(scaler_path):
        return None, "scaler.pkl not found."
    try:
        with open(scaler_path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, f"Failed to load scaler.pkl: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# LOAD METRICS  — prefer the lightweight metrics.json produced by
# train_model.py.  Fall back to recomputing from .npy + .pkl files only
# if the JSON is missing.  This makes deployment much lighter because the
# large .npy arrays are no longer required on the server.
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_metrics_from_json():
    """
    Read metrics.json and reshape it into the dict format the pages expect:
      { "baseline": { "Accuracy", "Precision", "Recall", "F1-Score", "Confusion" },
        "augmented": { ... },
        "counts":    { "total_real", "total_syn", ... } }
    Returns (metrics_dict, None) on success, or (None, error_string).
    """
    json_path = _path("metrics.json")
    if not os.path.exists(json_path):
        return None, "metrics.json not found."
    try:
        with open(json_path, "r") as f:
            raw = json.load(f)

        # Reshape into the format the dashboard pages expect
        def _reshape(section):
            return {
                "Accuracy":  section["accuracy"],
                "Precision": section["precision"],
                "Recall":    section["recall"],
                "F1-Score":  section["f1"],
                "Confusion": section["confusion_matrix"],
            }

        counts = raw.get("sample_counts", {})
        metrics = {
            "baseline":  _reshape(raw["baseline"]),
            "augmented": _reshape(raw["augmented"]),
            "counts": {
                "real_normal":  counts.get("real_normal_count", 0),
                "real_anomaly": counts.get("real_anomaly_count", 0),
                "syn_normal":   counts.get("synthetic_normal_count", 0),
                "syn_anomaly":  counts.get("synthetic_anomaly_count", 0),
                "total_real":   counts.get("real_normal_count", 0)
                                + counts.get("real_anomaly_count", 0),
                "total_syn":    counts.get("synthetic_normal_count", 0)
                                + counts.get("synthetic_anomaly_count", 0),
            },
        }
        return metrics, None
    except Exception as e:
        return None, f"Failed to read metrics.json: {e}"


@st.cache_data
def compute_metrics_from_npy():
    """
    Fallback: reproduce the exact evaluation from train_model.py using the
    .npy arrays and .pkl models.  This is only called when metrics.json is
    missing (e.g. during local development before train_model.py is re-run).

    Returns (metrics_dict, None) on success, or (None, error_string).
    """
    # Late import — only needed for this fallback path
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )
    except ImportError as e:
        return None, f"scikit-learn is required but could not be imported: {e}"

    # ── Check that every required file exists before loading ──────────────
    required_files = [
        "normal_windows.npy",
        "anomaly_windows.npy",
        "synthetic_normal_windows.npy",
        "synthetic_anomaly_windows.npy",
        "model_baseline.pkl",
        "model_augmented.pkl",
    ]
    missing = [f for f in required_files if not os.path.exists(_path(f))]
    if missing:
        return None, (
            "Cannot compute metrics — missing files: "
            + ", ".join(missing)
        )

    try:
        # ── Load windowed arrays ──────────────────────────────────────────
        normal_w  = np.load(_path("normal_windows.npy"))
        anomaly_w = np.load(_path("anomaly_windows.npy"))
        syn_norm  = np.load(_path("synthetic_normal_windows.npy"))
        syn_anom  = np.load(_path("synthetic_anomaly_windows.npy"))

        # ── Flatten: (N, 24, 3) → (N, 72) ────────────────────────────────
        normal_flat   = normal_w.reshape(len(normal_w), -1)
        anomaly_flat  = anomaly_w.reshape(len(anomaly_w), -1)
        syn_norm_flat = syn_norm.reshape(len(syn_norm), -1)
        syn_anom_flat = syn_anom.reshape(len(syn_anom), -1)

        # ── Baseline dataset (real only) ──────────────────────────────────
        X_base = np.vstack([normal_flat, anomaly_flat])
        y_base = np.concatenate(
            [np.zeros(len(normal_flat)), np.ones(len(anomaly_flat))]
        )

        # ── Augmented dataset (real + synthetic) ──────────────────────────
        X_aug = np.vstack(
            [normal_flat, syn_norm_flat, anomaly_flat, syn_anom_flat]
        )
        y_aug = np.concatenate(
            [
                np.zeros(len(normal_flat) + len(syn_norm_flat)),
                np.ones(len(anomaly_flat) + len(syn_anom_flat)),
            ]
        )

        # ── 80 / 20 stratified split (same seed as train_model.py) ────────
        _, X_base_test, _, y_base_test = train_test_split(
            X_base, y_base, test_size=0.2, random_state=42, stratify=y_base,
        )
        _, X_aug_test, _, y_aug_test = train_test_split(
            X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug,
        )

        # ── Load both models ─────────────────────────────────────────────
        with open(_path("model_baseline.pkl"), "rb") as f:
            baseline_model = pickle.load(f)
        with open(_path("model_augmented.pkl"), "rb") as f:
            augmented_model = pickle.load(f)

        # ── Evaluate ──────────────────────────────────────────────────────
        def _eval(model, X_test, y_test):
            y_pred = model.predict(X_test)
            return {
                "Accuracy":  accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall":    recall_score(y_test, y_pred, zero_division=0),
                "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
                # .tolist() so the result is JSON-serialisable for cache_data
                "Confusion": confusion_matrix(y_test, y_pred).tolist(),
            }

        base_m = _eval(baseline_model, X_base_test, y_base_test)
        aug_m  = _eval(augmented_model, X_aug_test, y_aug_test)

        # ── Sample counts for the Overview KPI cards ──────────────────────
        counts = {
            "real_normal":  int(len(normal_w)),
            "real_anomaly": int(len(anomaly_w)),
            "syn_normal":   int(len(syn_norm)),
            "syn_anomaly":  int(len(syn_anom)),
            "total_real":   int(len(normal_w) + len(anomaly_w)),
            "total_syn":    int(len(syn_norm) + len(syn_anom)),
        }

        return {"baseline": base_m, "augmented": aug_m, "counts": counts}, None

    except Exception as e:
        return None, f"Error computing metrics from .npy files: {e}"


def get_metrics():
    """
    Return the metrics dict or None.  Tries metrics.json first, then falls
    back to recomputing from .npy files.  Returns (metrics_dict, error_msg).
    """
    metrics, err = load_metrics_from_json()
    if metrics is not None:
        return metrics, None

    # Fallback: compute from raw data files
    metrics, err2 = compute_metrics_from_npy()
    if metrics is not None:
        return metrics, None

    # Both paths failed — combine error messages
    return None, f"{err}  |  Fallback also failed: {err2}"


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

# Styled project logo / name
st.sidebar.markdown(
    """
    <div style="text-align:center; padding:12px 0 18px 0;">
        <span style="font-size:2.4rem;">📡</span><br>
        <span style="font-size:1.05rem; font-weight:700; color:#f1f5f9;">
            IoT Anomaly Detection
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

# Navigation radio
page = st.sidebar.radio(
    "Navigate",
    ["🏠  Overview", "📊  Model Performance", "🔮  Live Prediction"],
    index=0,
    label_visibility="collapsed",
)

# About blurb at sidebar bottom
st.sidebar.markdown("---")
st.sidebar.caption(
    "**About** · TimeGAN-augmented anomaly detection for IoT sensor streams. "
    "Built with TimeGAN · RandomForest · Streamlit."
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: custom HTML metric card
# ═══════════════════════════════════════════════════════════════════════════
def metric_card(icon, value, label, accent=""):
    """
    Render a styled KPI card.

    Parameters
    ----------
    icon   : str  — emoji displayed above the value
    value  : str  — the main number / text
    label  : str  — small uppercase caption below the value
    accent : str  — optional CSS class for colour (accent-blue, accent-purple, accent-green)
    """
    cls = f"mc-value {accent}" if accent else "mc-value"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="mc-icon">{icon}</div>
            <div class="{cls}">{value}</div>
            <div class="mc-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: find contiguous anomaly segments efficiently using NumPy
# ═══════════════════════════════════════════════════════════════════════════
def find_anomaly_segments(labels_array):
    """
    Given a 1-D array/Series of 0s and 1s, return a list of (start_idx, end_idx)
    tuples marking contiguous runs of 1s.  Uses NumPy diff for speed instead
    of a Python-level row-by-row loop.
    """
    if len(labels_array) == 0:
        return []

    arr = np.asarray(labels_array, dtype=int)
    # Pad with 0 at both ends so transitions at boundaries are detected
    padded = np.concatenate([[0], arr, [0]])
    diff = np.diff(padded)
    # Rising edges (0→1) mark segment starts; falling edges (1→0) mark ends
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1  # inclusive end index
    return list(zip(starts.tolist(), ends.tolist()))


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="page-title">📡 IoT Anomaly Detection with TimeGAN</p>'
        '<p class="page-sub">'
        "Synthetic time-series augmentation to improve anomaly detection "
        "in IoT sensor data"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── KPI cards row ─────────────────────────────────────────────────────
    metrics, metrics_err = get_metrics()

    if metrics is not None:
        counts = metrics["counts"]
        base   = metrics["baseline"]
        aug    = metrics["augmented"]

        # Percentage-point improvements (guard against zero-division)
        base_recall = base["Recall"]
        base_f1     = base["F1-Score"]
        recall_imp  = (aug["Recall"] - base_recall) * 100
        f1_imp      = (aug["F1-Score"] - base_f1) * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(
                "📊", f'{counts["total_real"]:,}', "Real Samples", "accent-blue",
            )
        with c2:
            metric_card(
                "🧬", f'{counts["total_syn"]:,}', "Synthetic Samples", "accent-purple",
            )
        with c3:
            metric_card(
                "🎯", f"+{recall_imp:.2f} pp", "Recall Improvement", "accent-green",
            )
        with c4:
            metric_card(
                "📈", f"+{f1_imp:.2f} pp", "F1-Score Improvement", "accent-green",
            )
    else:
        st.warning(
            "⚠️ Metrics data not available.  Run `train_model.py` to generate "
            "`metrics.json` (or ensure the `.npy` / `.pkl` files are present).  "
            f"Detail: {metrics_err}"
        )

    st.divider()

    # ── How It Works — 4-step pipeline ────────────────────────────────────
    st.markdown("### 🔄 How It Works")

    steps = [
        ("📥", "Data Generation",
         "Generate realistic IoT sensor data with normal and anomaly patterns"),
        ("🧬", "TimeGAN Augmentation",
         "Train TimeGAN to synthesise additional normal & anomaly time-series"),
        ("🤖", "Model Training",
         "Train RandomForest on real-only (Baseline) vs real + synthetic (Augmented)"),
        ("📊", "Evaluation",
         "Compare both models on Accuracy, Precision, Recall & F1-Score"),
    ]

    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{i + 1}</div><br>
                    <div class="step-icon">{icon}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── EDA plots in a 2-column layout ────────────────────────────────────
    st.markdown("### 🔎 Exploratory Data Analysis")

    left, right = st.columns(2)

    ts_path  = _path("plots/timeseries_with_anomalies.png")
    box_path = _path("plots/boxplots_normal_vs_anomaly.png")

    with left:
        if os.path.exists(ts_path):
            st.image(
                ts_path,
                caption="Sensor readings over 30 days  (red = anomaly windows)",
                use_container_width=True,
            )
        else:
            st.info("Run `eda.py` to generate this plot.")

    with right:
        if os.path.exists(box_path):
            st.image(
                box_path,
                caption="Normal vs Anomaly distributions per sensor",
                use_container_width=True,
            )
        else:
            st.info("Run `eda.py` to generate this plot.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":

    st.markdown(
        '<p class="page-title">📊 Model Performance: Baseline vs Augmented</p>'
        '<p class="page-sub">'
        "Side-by-side comparison of the real-only and TimeGAN-augmented "
        "RandomForest classifiers"
        "</p>",
        unsafe_allow_html=True,
    )

    metrics, metrics_err = get_metrics()

    if metrics is not None:
        base = metrics["baseline"]
        aug  = metrics["augmented"]

        # ── st.metric cards with delta ────────────────────────────────────
        st.markdown("#### Key Metrics  *(Augmented model shown, delta vs Baseline)*")

        m1, m2, m3, m4 = st.columns(4)
        for col, name in zip(
            [m1, m2, m3, m4],
            ["Accuracy", "Precision", "Recall", "F1-Score"],
        ):
            delta_val = aug[name] - base[name]
            with col:
                st.metric(
                    label=name,
                    value=f"{aug[name]:.4f}",
                    delta=f"{delta_val:+.4f} vs baseline",
                )

        st.divider()

        # ── Comparison bar chart ──────────────────────────────────────────
        st.markdown("#### 📊 Metric Comparison Chart")
        comp_path = _path("plots/comparison_metrics.png")
        if os.path.exists(comp_path):
            st.image(comp_path, use_container_width=True)
        else:
            st.warning("Run `train_model.py` to generate the comparison chart.")

        st.divider()

        # ── Confusion matrices side by side ───────────────────────────────
        st.markdown("#### 🔢 Confusion Matrices")

        def _render_cm(cm, title):
            """Return an HTML string for a styled 2×2 confusion matrix."""
            try:
                tn, fp = int(cm[0][0]), int(cm[0][1])
                fn, tp = int(cm[1][0]), int(cm[1][1])
            except (IndexError, TypeError, ValueError):
                return (
                    f'<p style="color:#991b1b;">Could not render confusion '
                    f'matrix for {title}.</p>'
                )
            return f"""
            <div style="text-align:center;">
                <p style="font-weight:600; color:#334155; margin-bottom:10px;">
                    {title}
                </p>
                <table class="cm-table">
                    <tr>
                        <th></th>
                        <th>Pred Normal</th>
                        <th>Pred Anomaly</th>
                    </tr>
                    <tr>
                        <th>Actual Normal</th>
                        <td class="cm-correct">{tn:,}</td>
                        <td class="cm-error">{fp:,}</td>
                    </tr>
                    <tr>
                        <th>Actual Anomaly</th>
                        <td class="cm-error">{fn:,}</td>
                        <td class="cm-correct">{tp:,}</td>
                    </tr>
                </table>
            </div>
            """

        cm_left, cm_right = st.columns(2)
        with cm_left:
            st.markdown(
                _render_cm(base["Confusion"], "Baseline (Real Only)"),
                unsafe_allow_html=True,
            )
        with cm_right:
            st.markdown(
                _render_cm(aug["Confusion"], "Augmented (Real + Synthetic)"),
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Key Takeaway callout ──────────────────────────────────────────
        recall_pp = (aug["Recall"] - base["Recall"]) * 100
        f1_pp     = (aug["F1-Score"] - base["F1-Score"]) * 100

        st.markdown(
            f"""
            <div class="callout-box">
                <div class="callout-title">💡 Key Takeaway</div>
                <div class="callout-text">
                    Adding TimeGAN-generated synthetic data improved
                    <b>Recall</b> by <b>{recall_pp:+.2f} pp</b> and
                    <b>F1-Score</b> by <b>{f1_pp:+.2f} pp</b>.
                    The augmented model catches more real anomalies while
                    maintaining high precision — exactly the goal of data
                    augmentation for imbalanced classes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.error(
            "Required data files are missing.  Run `preprocess.py` and "
            "`train_model.py` to generate them.  "
            f"Detail: {metrics_err}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮  Live Prediction":

    st.markdown(
        '<p class="page-title">🔮 Run Anomaly Detection on Your Data</p>'
        '<p class="page-sub">'
        "Upload a CSV of IoT sensor readings to detect anomalies with the "
        "trained Augmented model"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Expected-format hint ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="callout-box" style="margin-bottom:1rem;">
            <div class="callout-title">📄 Expected CSV Format</div>
            <div class="callout-text">
                <code>timestamp, temperature, vibration, pressure</code><br>
                Each row = one minute of sensor readings.  At least <b>24 rows</b>
                are needed to form one sliding window.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pre-check: are the model and scaler available? ────────────────────
    model_obj, model_err = load_augmented_model()
    scaler_obj, scaler_err = load_scaler()

    if model_err:
        st.error(f"❌ Cannot run predictions — {model_err}")
    if scaler_err:
        st.error(f"❌ Cannot run predictions — {scaler_err}")

    # Only show the uploader if the essential files are loadable
    if model_obj is not None and scaler_obj is not None:

        # ── File uploader ─────────────────────────────────────────────────
        uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])

        if uploaded is not None:

            # 1. Read the CSV ──────────────────────────────────────────────
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"❌ Could not parse the uploaded file: {e}")
                st.stop()

            # 2. Validate columns ─────────────────────────────────────────
            required = {"timestamp", "temperature", "vibration", "pressure"}
            if not required.issubset(set(df.columns)):
                missing = required - set(df.columns)
                st.error(
                    f"❌ Missing columns: **{', '.join(sorted(missing))}**.  "
                    "The CSV needs: `timestamp`, `temperature`, `vibration`, "
                    "`pressure`."
                )
                st.stop()

            # Parse timestamps safely (after column validation)
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception as e:
                st.error(
                    f"❌ Could not parse the `timestamp` column: {e}.  "
                    "Make sure it contains valid date/time values."
                )
                st.stop()

            # 3. Check minimum rows ───────────────────────────────────────
            SEQ_LEN = 24  # must match the window length used during training
            if len(df) < SEQ_LEN:
                st.error(
                    f"❌ Only **{len(df)}** rows — at least **{SEQ_LEN}** are "
                    "needed to create one sliding window."
                )
                st.stop()

            # Validate that sensor columns are numeric
            feature_cols = ["temperature", "vibration", "pressure"]
            for col_name in feature_cols:
                if not np.issubdtype(df[col_name].dtype, np.number):
                    try:
                        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                    except Exception:
                        pass
                if df[col_name].isna().all():
                    st.error(
                        f"❌ Column `{col_name}` contains no valid numeric data."
                    )
                    st.stop()

            # Drop rows with NaN in sensor columns (warn if any)
            nan_count = df[feature_cols].isna().sum().sum()
            if nan_count > 0:
                st.warning(
                    f"⚠️ Found {nan_count} NaN value(s) in sensor columns — "
                    "those rows will be dropped."
                )
                df = df.dropna(subset=feature_cols).reset_index(drop=True)
                if len(df) < SEQ_LEN:
                    st.error(
                        f"❌ After dropping NaN rows, only **{len(df)}** rows "
                        f"remain — at least **{SEQ_LEN}** are required."
                    )
                    st.stop()

            st.success(f"✅ Loaded **{len(df):,}** rows successfully.")

            # 4. Scale features with the saved scaler ─────────────────────
            try:
                df_scaled = df.copy()
                df_scaled[feature_cols] = scaler_obj.transform(
                    df[feature_cols].values
                )
            except Exception as e:
                st.error(f"❌ Scaling failed: {e}")
                st.stop()

            # 5. Create sliding windows (same logic as preprocess.py) ─────
            def create_windows(data, seq_len, step=1):
                """
                Slide a window of `seq_len` rows over `data`.
                Returns a 3-D NumPy array: (num_windows, seq_len, num_features).
                """
                windows = []
                for start in range(0, len(data) - seq_len + 1, step):
                    windows.append(data[start : start + seq_len])
                return np.array(windows)

            windows = create_windows(
                df_scaled[feature_cols].values, SEQ_LEN
            )
            # Flatten each window to 1-D: (N, 24, 3) → (N, 72)
            windows_flat = windows.reshape(len(windows), -1)

            # 6. Run predictions ──────────────────────────────────────────
            try:
                preds = model_obj.predict(windows_flat)  # 0 = normal, 1 = anomaly
            except Exception as e:
                st.error(f"❌ Model prediction failed: {e}")
                st.stop()

            total    = len(preds)
            n_anom   = int(np.sum(preds))
            n_norm   = total - n_anom
            anom_pct = (n_anom / total * 100) if total else 0.0

            # 7. Summary metric cards ─────────────────────────────────────
            st.divider()
            st.markdown("#### 📋 Prediction Summary")

            s1, s2, s3 = st.columns(3)
            with s1:
                metric_card("🔍", f"{total:,}", "Windows Analyzed", "accent-blue")
            with s2:
                metric_card("🚨", f"{n_anom:,}", "Anomalies Detected")
            with s3:
                metric_card("📉", f"{anom_pct:.1f}%", "Anomaly Rate", "accent-purple")

            # 8. Map window predictions → per-row labels ──────────────────
            #    A row is flagged anomalous if ANY window containing it was
            #    predicted as anomaly.
            row_labels = np.zeros(len(df), dtype=int)
            for i, p in enumerate(preds):
                if p == 1:
                    row_labels[i : i + SEQ_LEN] = 1
            df["predicted"] = row_labels

            # 9. Interactive Plotly chart with anomaly shading ─────────────
            st.divider()
            st.markdown("#### 📉 Sensor Readings with Predicted Anomalies")

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    "Temperature (°F)",
                    "Vibration (mm/s)",
                    "Pressure (PSI)",
                ),
            )

            line_colors = {
                "temperature": "#0ea5e9",
                "vibration":   "#f59e0b",
                "pressure":    "#10b981",
            }

            # Pre-compute anomaly segments once (vectorised, fast)
            anomaly_segments = find_anomaly_segments(df["predicted"].values)

            for idx, sensor in enumerate(feature_cols, start=1):
                # Draw the sensor signal
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df[sensor],
                        mode="lines",
                        name=sensor.capitalize(),
                        line=dict(color=line_colors[sensor], width=1.2),
                    ),
                    row=idx,
                    col=1,
                )

                # Shade contiguous anomaly regions in red
                for seg_start, seg_end in anomaly_segments:
                    fig.add_vrect(
                        x0=df["timestamp"].iloc[seg_start],
                        x1=df["timestamp"].iloc[seg_end],
                        fillcolor="red",
                        opacity=0.15,
                        line_width=0,
                        row=idx,
                        col=1,
                    )

            fig.update_layout(
                height=680,
                hovermode="x unified",
                template="plotly_white",
                margin=dict(t=40, b=30),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 10. Download results as CSV ─────────────────────────────────
            st.divider()
            st.markdown("#### 📥 Export Results")

            # Build a per-window results table
            window_results = pd.DataFrame(
                {
                    "window_index":    range(total),
                    "start_timestamp": [
                        df["timestamp"].iloc[i] for i in range(total)
                    ],
                    "end_timestamp": [
                        df["timestamp"].iloc[i + SEQ_LEN - 1]
                        for i in range(total)
                    ],
                    "prediction": [
                        "Anomaly" if p == 1 else "Normal" for p in preds
                    ],
                }
            )

            csv_bytes = window_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download Prediction Results (CSV)",
                data=csv_bytes,
                file_name="anomaly_predictions.csv",
                mime="text/csv",
            )

            # Expandable raw data table
            with st.expander("🗂️ View per-window results"):
                st.dataframe(
                    window_results, use_container_width=True, hide_index=True,
                )

        else:
            # Prompt when no file is uploaded yet
            st.info(
                "👆 Upload a CSV to get started.  You can use the project's own "
                "`iot_sensor_data.csv` as a quick test."
            )

    else:
        # Model or scaler could not be loaded — upload form hidden
        st.info(
            "The file uploader will appear once the model and scaler files "
            "are available."
        )
