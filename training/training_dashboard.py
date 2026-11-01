"""
Jaanch AI Model Real-Time Training Progress Dashboard
A live Streamlit application displaying:
1. Live Training Progress & Status
2. Current Epoch & Total Epochs
3. Completed Batches & Total Batches
4. Interactive Loss Curves (Train Loss vs Validation Loss)
5. Interactive Metric Curves (Val Accuracy, Precision, Recall, F1-Score)
6. Model Performance & Best Checkpoint Details
7. Live Logs Table from actual CSV logs
8. Execution & Resumption Controls

Strictly relies on actual training execution logs (ml/reports/training_logs.csv & checkpoint_state.json).
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Jaanch AI Training Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-running {
        color: #2ca02c;
        font-weight: bold;
    }
    .status-completed {
        color: #1f77b4;
        font-weight: bold;
    }
    .status-idle {
        color: #7f7f7f;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_CSV_PATH = BASE_DIR / "ml" / "reports" / "training_logs.csv"
STATE_JSON_PATH = BASE_DIR / "ml" / "models" / "checkpoints" / "checkpoint_state.json"
CHECKPOINTS_DIR = BASE_DIR / "ml" / "models" / "checkpoints"


def load_training_state():
    """Loads real-time state from checkpoint_state.json."""
    if STATE_JSON_PATH.exists():
        try:
            with open(STATE_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_training_logs():
    """Loads actual CSV log file."""
    if LOG_CSV_PATH.exists():
        try:
            df = pd.read_csv(LOG_CSV_PATH)
            if not df.empty:
                return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def format_seconds(seconds: float) -> str:
    """Formats seconds into readable string."""
    if seconds is None or np.isnan(seconds) or seconds < 0:
        return "00:00"
    return str(timedelta(seconds=int(seconds)))


# -----------------------------------------------------------------------------
# Sidebar: Control Panel & Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Training Controls")
st.sidebar.markdown("Real-time pipeline monitoring & execution")

auto_refresh = st.sidebar.checkbox("Auto-Refresh Dashboard", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=1, max_value=10, value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Launch Local Training")
epochs_input = st.sidebar.number_input("Total Epochs", min_value=1, max_value=50, value=5, step=1)
batch_size_input = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
training_mode = st.sidebar.selectbox("Training Engine", ["SGD / Logistic Regression (Fast)", "Transformer (HF Trainer)"], index=0)

mode_flag = "sklearn" if "SGD" in training_mode else "transformer"

col_btn1, col_btn2 = st.sidebar.columns(2)

can_resume = (CHECKPOINTS_DIR / "checkpoint_latest.joblib").exists() or (CHECKPOINTS_DIR / "checkpoint_latest.pt").exists()

if col_btn1.button("▶️ Start Training", use_container_width=True):
    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "train_local.py"),
        "--mode", mode_flag,
        "--epochs", str(epochs_input),
        "--batch-size", str(batch_size_input)
    ]
    subprocess.Popen(cmd, cwd=str(BASE_DIR))
    st.sidebar.success("Training process launched in background!")
    time.sleep(1)
    st.rerun()

if col_btn2.button("🔄 Resume", disabled=not can_resume, use_container_width=True):
    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "train_local.py"),
        "--mode", mode_flag,
        "--epochs", str(epochs_input),
        "--batch-size", str(batch_size_input),
        "--resume"
    ]
    subprocess.Popen(cmd, cwd=str(BASE_DIR))
    st.sidebar.info("Resuming training from latest checkpoint!")
    time.sleep(1)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**Data Integrity Invariant**:
All curves, metrics, and progress bars come exclusively from actual training execution logs (`training_logs.csv`).
No simulated values are generated.
""")

# -----------------------------------------------------------------------------
# Main Dashboard View
# -----------------------------------------------------------------------------
st.title("🛡️ Jaanch AI Model Training Monitor")
st.markdown("Real-time tracking of epochs, batches, loss convergence, validation metrics, and model checkpoints.")

state = load_training_state()
logs_df = load_training_logs()

# Determine overall status
raw_status = state.get("status", "IDLE")
if raw_status in ["TRAINING", "RUNNING", "IN_PROGRESS"]:
    status_display = "🟢 TRAINING IN PROGRESS"
    status_class = "status-running"
elif raw_status == "COMPLETED":
    status_display = "🔵 TRAINING COMPLETED"
    status_class = "status-completed"
else:
    status_display = "⚪ IDLE / READY"
    status_class = "status-idle"

# Top Metric KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

current_epoch = state.get("current_epoch", 0)
total_epochs = state.get("total_epochs", epochs_input)
current_batch = state.get("current_batch", 0)
total_batches = state.get("total_batches", 0)
elapsed_sec = state.get("elapsed_seconds", 0)
eta_sec = state.get("eta_seconds", 0)
best_metric_val = state.get("best_metric_value", 0.0)
best_epoch = state.get("best_epoch", 0)
best_metric_name = state.get("best_metric_name", "F1").upper()

with col1:
    st.metric("Status", status_display.split(" ")[1])
with col2:
    st.metric("Epoch Progress", f"{current_epoch} / {total_epochs}")
with col3:
    st.metric("Batch Progress", f"{current_batch} / {total_batches}")
with col4:
    st.metric("Elapsed Time", format_seconds(elapsed_sec))
with col5:
    st.metric("Estimated ETA", format_seconds(eta_sec))

# Overall Progress Bar
if total_epochs > 0 and total_batches > 0:
    progress_frac = min(1.0, max(0.0, ((current_epoch - 1) * total_batches + current_batch) / (total_epochs * total_batches)))
elif total_epochs > 0 and current_epoch > 0:
    progress_frac = min(1.0, max(0.0, current_epoch / total_epochs))
else:
    progress_frac = 0.0

st.progress(progress_frac, text=f"Total Pipeline Progress: {progress_frac * 100:.1f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# Visual Analytics: Loss and Accuracy Curves
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Loss & Accuracy Curves", "🏆 Model Performance & Checkpoints", "📋 Real-Time Log Records"])

with tab1:
    if logs_df.empty:
        st.info("ℹ️ No training epochs logged yet. Launch training from the sidebar to view live loss and accuracy curves.")
    else:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Training vs Validation Loss")
            loss_df = logs_df[["epoch", "train_loss", "val_loss"]].set_index("epoch")
            st.line_chart(loss_df)

        with col_c2:
            st.subheader("Validation Metrics (Accuracy, Recall, Precision, F1)")
            metrics_chart_df = logs_df[["epoch", "val_accuracy", "precision", "recall", "f1"]].set_index("epoch")
            st.line_chart(metrics_chart_df)

with tab2:
    st.subheader("Checkpoint Governance & Best Model Selection")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Best Model Checkpoint</h4>
            <p><b>Target Metric:</b> {best_metric_name}</p>
            <p><b>Peak Score:</b> {best_metric_val:.4f}</p>
            <p><b>Achieved at Epoch:</b> {best_epoch}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        latest_ckpt = CHECKPOINTS_DIR / "checkpoint_latest.joblib"
        best_ckpt = CHECKPOINTS_DIR / "checkpoint_best.joblib"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Artifact Storage</h4>
            <p><b>Latest Checkpoint:</b> {'✅ Ready' if latest_ckpt.exists() else '❌ None'}</p>
            <p><b>Best Checkpoint:</b> {'✅ Ready' if best_ckpt.exists() else '❌ None'}</p>
            <p><b>Resume Available:</b> {'Yes' if can_resume else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_p3:
        if not logs_df.empty:
            last_row = logs_df.iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h4>Latest Epoch Summary</h4>
                <p><b>Val Accuracy:</b> {last_row.get('val_accuracy', 0.0):.4f}</p>
                <p><b>Scam F1-Score:</b> {last_row.get('f1', 0.0):.4f}</p>
                <p><b>Scam Recall:</b> {last_row.get('recall', 0.0):.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Latest Epoch Summary</h4>
                <p>Waiting for first epoch completion...</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.subheader("Live Training Log Stream (`reports/training_logs.csv`)")
    if logs_df.empty:
        st.text("No training logs recorded yet.")
    else:
        st.dataframe(
            logs_df.sort_values(by="epoch", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# Auto-refresh loop
if auto_refresh and raw_status in ["TRAINING", "RUNNING", "IN_PROGRESS"]:
    time.sleep(refresh_interval)
    st.rerun()

