"""EpisodeForge: Enterprise Robotics Data Operations & Quality Dashboard."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EpisodeForge | Operations & Quality Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 EpisodeForge: Robotics Data Platform")
st.markdown(
    "**Enterprise Data Transformation Engine:** Raw Recordings $\\rightarrow$ Validation $\\rightarrow$ Client Schema Transforms $\\rightarrow$ Isolated Loader Verification $\\rightarrow$ Cloud Delivery"
)

# Connect to DuckDB Lineage Catalog
db_path = Path("data/lineage_catalog.duckdb")
conn = None
if db_path.exists():
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception as e:
        st.warning(f"Connecting in read-write mode: {e}")
        conn = duckdb.connect(str(db_path))

# Top Metric KPI Row
st.subheader("📊 Operational Telemetry & Fleet Health")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

total_runs = 0
total_eps = 0
avg_throughput = 0.0
overall_pass_rate = 100.0
total_gb_delivered = 0.0

if conn is not None:
    try:
        runs_df = conn.execute("SELECT * FROM runs ORDER BY timestamp_utc DESC").df()
        if not runs_df.empty:
            total_runs = len(runs_df)
            total_eps = int(runs_df["episodes_released"].sum())
            avg_throughput = float(runs_df["throughput_eps_hr"].mean())
            pass_count = (runs_df["validation_status"] == "PASS").sum()
            overall_pass_rate = (pass_count / total_runs) * 100.0

        deliveries_df = conn.execute("SELECT * FROM deliveries").df()
        if not deliveries_df.empty:
            total_gb_delivered = float(deliveries_df["total_bytes"].sum()) / (1024**3)
    except Exception as e:
        st.error(f"Error querying telemetry: {e}")

m_col1.metric("Total Execution Runs", f"{total_runs:,}")
m_col2.metric("Delivered Episodes", f"{total_eps:,}")
m_col3.metric("Fleet Throughput", f"{avg_throughput:,.0f} eps/hr")
m_col4.metric("Client Loader Pass Rate", f"{overall_pass_rate:.1f}%")
m_col5.metric("Cloud Egress / Delivery", f"{total_gb_delivered:.2f} GB")

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🚀 Recent Pipeline Execution Runs")
    if conn is not None:
        try:
            df = conn.execute(
                "SELECT run_id, client, dataset_name, format, episodes_released, throughput_eps_hr, validation_status, timestamp_utc FROM runs ORDER BY timestamp_utc DESC LIMIT 10"
            ).df()
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No pipeline execution runs logged yet. Execute `forge run` to process datasets."
                )
        except Exception as e:
            st.error(f"Catalog query failed: {e}")
    else:
        st.info("Lineage catalog database initializing...")

with col_right:
    st.subheader("📦 Published Dataset Releases")
    releases = sorted(
        Path("data/releases").glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if releases:
        rel_data = []
        for r in releases:
            client_name = r.parent.name
            rel_name = r.name
            manifest_file = r / "delivery_manifest.json"
            status = "Ready" if manifest_file.exists() else "Local"
            file_count = len(list(r.glob("**/*")))
            rel_data.append(
                {
                    "Client": client_name,
                    "Release Package": rel_name,
                    "Files": file_count,
                    "Manifest": status,
                }
            )
        st.dataframe(pd.DataFrame(rel_data), use_container_width=True, hide_index=True)
    else:
        st.info("No releases in `data/releases/` yet.")

st.divider()

st.subheader("⚡ Quick Launch Navigation")
n_col1, n_col2, n_col3, n_col4 = st.columns(4)
with n_col1:
    st.markdown("### 🔍 [DLQ Triage](/DLQ_Triage)")
    st.caption("Inspect quarantined episodes, kinematic errors, and failure taxonomy.")
with n_col2:
    st.markdown("### 🗂️ [Dataset Explorer](/Dataset_Explorer)")
    st.caption("Inspect delivered LeRobot / robomimic datasets, metadata, and video previews.")
with n_col3:
    st.markdown("### 💰 [Cost Forecaster](/Cost_Forecaster)")
    st.caption("Interactive quota calculator, cloud rates, and resource planning.")
with n_col4:
    st.markdown("### 📜 [Lineage Explorer](/Lineage_Explorer)")
    st.caption("Browse DuckDB provenance graphs, deliveries, and data recall notices.")
