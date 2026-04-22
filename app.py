
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="WWTP Intelligence Dashboard",
    page_icon="💧",
    layout="wide"
)

st.title("💧 WWTP Intelligence Dashboard")
st.markdown(
    "**Eastern Treatment Plant, Melbourne** — "
    "COD Soft-Sensor + Operational Anomaly Detection"
)
st.markdown("---")

FEATURE_COLS = [
    "avg_outflow", "avg_inflow", "total_grid", "Am", "TN",
    "T", "H", "PP", "VV", "VM",
    "year", "month_sin", "month_cos",
    "temp_humidity", "inflow_rolling7",
    "BOD_lag1", "COD_lag1"
]

@st.cache_resource
def load_all_models():
    xgb_cod     = joblib.load("xgb_cod.pkl")
    scaler_X    = joblib.load("scaler_X.pkl")
    iso_forest  = joblib.load("iso_forest.pkl")
    scaler_iso  = joblib.load("scaler_iso.pkl")
    lstm_config = joblib.load("lstm_ae_config.pkl")
    autoencoder = load_model("lstm_autoencoder.keras")
    return xgb_cod, scaler_X, iso_forest, scaler_iso, lstm_config, autoencoder

xgb_cod, scaler_X, iso_forest, scaler_iso, lstm_config, autoencoder = load_all_models()
SEQ_LEN   = lstm_config["seq_len"]
threshold = lstm_config["threshold"]
scaler_lstm = lstm_config["scaler"]

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("📋 Daily Plant Inputs")

with st.sidebar:
    st.subheader("🏭 Plant Operational")
    avg_outflow     = st.number_input("Avg Outflow (m³/s)",    0.0,  10.0,  3.7)
    avg_inflow      = st.number_input("Avg Inflow (m³/s)",     0.0,  20.0,  4.5)
    total_grid      = st.number_input("Total Grid (kWh)", 100000, 400000, 275000, step=1000)
    Am              = st.number_input("Ammonia — Am (mg/L)",    0.0, 100.0, 39.0)
    TN              = st.number_input("Total Nitrogen (mg/L)", 40.0,  95.0, 62.5)
    BOD_lag1        = st.number_input("Yesterday BOD (mg/L)",100.0, 900.0,382.0)
    COD_lag1        = st.number_input("Yesterday COD (mg/L)",300.0,1800.0,846.0)
    inflow_rolling7 = st.number_input("7-day Avg Inflow",      0.0,  20.0,  4.5)
    st.subheader("🌤️ Weather")
    T    = st.number_input("Mean Temp °C",       0.0, 40.0, 15.0)
    H    = st.number_input("Humidity %",           0,  100,   64)
    PP   = st.number_input("Precipitation (mm)",  0.0, 20.0,  0.0)
    VV   = st.number_input("Visibility (km)",     0.0, 15.0,  9.0)
    VM   = st.number_input("Max Wind (km/h)",     0.0, 90.0, 35.0)
    st.subheader("📅 Date")
    year  = st.selectbox("Year",  list(range(2014, 2030)), index=4)
    month = st.selectbox("Month", list(range(1, 13)),      index=5)

month_sin     = np.sin(2 * np.pi * month / 12)
month_cos     = np.cos(2 * np.pi * month / 12)
temp_humidity = T * H

input_dict = {
    "avg_outflow": avg_outflow, "avg_inflow": avg_inflow,
    "total_grid": total_grid,   "Am": Am,
    "TN": TN,                   "T": T,
    "H": H,                     "PP": PP,
    "VV": VV,                   "VM": VM,
    "year": year,               "month_sin": month_sin,
    "month_cos": month_cos,     "temp_humidity": temp_humidity,
    "inflow_rolling7": inflow_rolling7,
    "BOD_lag1": BOD_lag1,       "COD_lag1": COD_lag1
}
input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

tab1, tab2, tab3 = st.tabs([
    "🧪 COD Prediction",
    "🚨 Anomaly Detection",
    "📊 Historical Overview"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.header("🧪 Effluent COD Soft-Sensor")
    X_scaled = scaler_X.transform(input_df)
    cod_pred = xgb_cod.predict(X_scaled)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted COD",  f"{cod_pred:.1f} mg/L")
    col2.metric("Plant Mean COD", "846 mg/L")
    col3.metric("vs Plant Mean",  f"{cod_pred - 846:+.1f} mg/L",
                delta_color="inverse")

    if cod_pred > 1173:
        st.error("⚠️ ALERT: Predicted COD exceeds upper fence (1173 mg/L)")
    elif cod_pred > 1000:
        st.warning("⚡ Elevated COD — monitor closely.")
    else:
        st.success("✅ COD within normal operating range.")

    st.subheader("🔍 SHAP Prediction Explanation")
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    explainer   = shap.TreeExplainer(xgb_cod)
    shap_vals   = explainer(X_scaled_df)
    fig_shap, ax = plt.subplots(figsize=(9, 5))
    plt.sca(ax)
    shap.plots.waterfall(shap_vals[0], show=False)
    ax.set_title(f"SHAP Waterfall — Predicted COD: {cod_pred:.1f} mg/L",
                 fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.header("🚨 Operational Anomaly Detection")

    X_iso    = scaler_iso.transform(input_df)
    if_score = iso_forest.decision_function(X_iso)[0]
    if_label = iso_forest.predict(X_iso)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌲 Isolation Forest")
        st.metric("Anomaly Score", f"{if_score:.4f}")
        if if_label == -1:
            st.error("🔴 ANOMALY — Point/instantaneous deviation detected")
        else:
            st.success("🟢 Normal — No point anomaly detected")

        fig_if, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Score"], [if_score],
                color="red" if if_label == -1 else "steelblue", alpha=0.7)
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
        ax.set_xlim(-0.15, 0.15)
        ax.set_title("IF Score (negative = anomaly)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_if)
        plt.close()

    with col2:
        st.subheader("🧠 LSTM Autoencoder")
        st.info(
            "Upload a CSV with **7 rows** (last 7 days) and "
            "17 feature columns for temporal anomaly detection."
        )
        uploaded = st.file_uploader("Upload 7-day sequence CSV", type="csv")
        if uploaded is not None:
            seq_df = pd.read_csv(uploaded)
            if seq_df.shape == (7, len(FEATURE_COLS)):
                seq_scaled = scaler_lstm.transform(seq_df[FEATURE_COLS])
                seq_input  = seq_scaled.reshape(1, SEQ_LEN, len(FEATURE_COLS))
                seq_recon  = autoencoder.predict(seq_input, verbose=0)
                recon_err  = float(np.mean((seq_input - seq_recon) ** 2))
                st.metric("Reconstruction Error", f"{recon_err:.5f}")
                st.metric("Threshold",            f"{threshold:.5f}")
                if recon_err > threshold:
                    st.error(f"🔴 TEMPORAL ANOMALY — Error {recon_err:.5f} > threshold {threshold:.5f}")
                else:
                    st.success("🟢 Normal temporal pattern")
            else:
                st.error(f"CSV must have exactly 7 rows and {len(FEATURE_COLS)} columns.")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Historical Anomaly Overview")
    try:
        adf = pd.read_csv("anomaly_results.csv", parse_dates=["date"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days",     len(adf))
        c2.metric("IF Anomalies",   int(adf["iso_anomaly"].sum()),
                  f"{100*adf['iso_anomaly'].mean():.1f}%")
        c3.metric("LSTM Anomalies", int(adf["lstm_anomaly"].sum()),
                  f"{100*adf['lstm_anomaly'].mean():.1f}%")
        c4.metric("Both Methods",
                  int((adf["iso_anomaly"] & adf["lstm_anomaly"]).sum()))

        import matplotlib.dates as mdates
        fig_h, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(adf["date"], adf["COD"], color="coral",
                     linewidth=0.7, alpha=0.7)
        axes[0].scatter(adf.loc[adf["iso_anomaly"]==1, "date"],
                        adf.loc[adf["iso_anomaly"]==1, "COD"],
                        color="red", s=15, marker="x", zorder=5,
                        label="IF Anomaly")
        axes[0].set_ylabel("COD (mg/L)")
        axes[0].set_title("COD with IF Anomalies", fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[1].plot(adf["date"], adf["lstm_error"],
                     color="darkorange", linewidth=0.8, alpha=0.8)
        axes[1].axhline(threshold, color="red", linestyle="--", linewidth=1.5)
        axes[1].set_ylabel("LSTM Recon. Error")
        axes[1].set_title("LSTM Reconstruction Error Over Time", fontweight="bold")
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close()
    except FileNotFoundError:
        st.warning("anomaly_results.csv not found.")

st.markdown("---")
st.caption("Raja Chouhan | 230107055 | CL653 | Eastern Treatment Plant, Melbourne")
