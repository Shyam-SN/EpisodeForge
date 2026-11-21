"""Interactive Pipeline Financial & Resource Forecaster."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from episodeforge.config.client_spec import ClientSpec
from episodeforge.config.cost_config import CostRates
from episodeforge.metrics.cost import forecast_pipeline_cost, load_cost_rates

st.set_page_config(page_title="Cost Forecaster | EpisodeForge", page_icon="💰", layout="wide")
st.title("💰 Production Capacity & Financial Forecaster")
st.markdown(
    "Forecast cloud compute, object storage, network egress, and delivery quotes based on measured pipeline throughput."
)

# Discover available client specs
spec_files = sorted(Path("configs/clients").glob("*.yaml"))
spec_options = {p.name: p for p in spec_files}

st.sidebar.header("Configuration & Sliders")
selected_spec_name = st.sidebar.selectbox("Target Client Specification", list(spec_options.keys()))
spec_path = spec_options[selected_spec_name]

with open(spec_path, encoding="utf-8") as f:
    spec_data = yaml.safe_load(f)
spec_model = ClientSpec.model_validate(spec_data)

episodes_volume = st.sidebar.slider(
    "Volume Quota (Episodes)",
    min_value=50,
    max_value=100000,
    value=5000,
    step=50,
)

worker_count = st.sidebar.slider(
    "Parallel Worker Allocation", min_value=1, max_value=32, value=4, step=1
)

# Cost Rate Overrides
with st.sidebar.expander("Cloud Provider Pricing Overrides", expanded=False):
    default_rates = load_cost_rates()
    c_rate = st.number_input(
        "Compute ($/vCPU-hr)", value=default_rates.compute_core_hour_usd, format="%.4f"
    )
    s_rate = st.number_input(
        "S3 Storage ($/GB-mo)", value=default_rates.s3_standard_storage_gb_month_usd, format="%.4f"
    )
    e_rate = st.number_input(
        "S3 Egress ($/GB)", value=default_rates.s3_egress_gb_usd, format="%.4f"
    )
    custom_rates = CostRates(
        compute_core_hour_usd=c_rate,
        s3_standard_storage_gb_month_usd=s_rate,
        s3_egress_gb_usd=e_rate,
    )

forecast = forecast_pipeline_cost(
    episodes=episodes_volume,
    spec=spec_model,
    rates=custom_rates,
    worker_count=worker_count,
)

# Display Key Forecast Metrics
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Robot Demo Hours", f"{forecast.total_robot_hours:.1f} hrs")
kpi2.metric("Wall-Clock Turnaround", f"{forecast.projected_wall_clock_hours:.2f} hrs")
kpi3.metric("Data Footprint", f"{forecast.total_dataset_gb:.2f} GB")
kpi4.metric("Total Project Cost", f"${forecast.total_cost_usd:.2f}")
kpi5.metric("Cost / Robot Hour", f"${forecast.cost_per_delivered_hour_usd:.2f} / hr")

st.divider()

col_breakdown, col_tradeoffs = st.columns([1, 1])

with col_breakdown:
    st.subheader("💵 Financial Quote Breakdown")
    fin_data = {
        "Cost Dimension": [
            "Cloud Compute (vCPU-hrs)",
            "S3 Standard Storage (Monthly)",
            "S3 Egress Transfer",
            "API PUT / GET Requests",
        ],
        "Subtotal ($ USD)": [
            round(forecast.compute_cost_usd, 4),
            round(forecast.storage_monthly_cost_usd, 4),
            round(forecast.egress_cost_usd, 4),
            round(forecast.api_requests_cost_usd, 4),
        ],
    }
    st.table(fin_data)
    st.markdown(f"**Unit Cost:** `${forecast.cost_per_episode_usd:.4f}` per validated episode.")

with col_tradeoffs:
    st.subheader("⚡ Speed vs. Cost Tradeoff")
    st.write(
        f"Processing **{episodes_volume:,}** episodes of `{spec_model.format}` requires **{forecast.total_compute_core_hours:.1f}** core-hours."
    )
    st.write(
        f"At **{worker_count}** workers, pipeline throughput is approximately **{worker_count * 1.5 * 3600:,.0f} eps/hr**."
    )
    st.info(
        "💡 **Optimization Insight:** Video compression (AV1 vs raw HDF5 arrays) yields a ~10x reduction in cloud storage and transfer costs with zero perceptual loss for robot policies."
    )
