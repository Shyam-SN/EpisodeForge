"""Dataset Release Explorer & Feature Inspector."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset Explorer | EpisodeForge", page_icon="🗂️", layout="wide")
st.title("🗂️ Dataset Release Explorer")
st.markdown(
    "Browse delivered immutable client packages, inspect metadata schemas, and verify file integrity."
)

releases_dir = Path("data/releases")
if not releases_dir.exists():
    st.info("No releases found in `data/releases/`.")
    st.stop()

release_folders = sorted(releases_dir.glob("*/*"), key=lambda p: str(p))
if not release_folders:
    st.info("No releases generated yet.")
    st.stop()

rel_names = [f"{p.parent.name}/{p.name}" for p in release_folders]
selected_rel_name = st.sidebar.selectbox("Select Release Package", rel_names)
selected_path = releases_dir / selected_rel_name

st.subheader(f"Package: `{selected_rel_name}`")
col_info, col_manifest = st.columns([1, 1])

manifest_file = selected_path / "delivery_manifest.json"
receipt_file = selected_path / "delivery_receipt.json"
report_file = selected_path / "delivery_report.md"

with col_info:
    st.markdown("### 📋 Release Files & Footprint")
    all_files = sorted(selected_path.glob("**/*"))
    file_records = []
    for fp in all_files:
        if fp.is_file():
            rel_p = str(fp.relative_to(selected_path))
            sz_kb = fp.stat().st_size / 1024.0
            file_records.append({"File": rel_p, "Size (KB)": round(sz_kb, 2)})
    df_files = pd.DataFrame(file_records)
    st.dataframe(df_files, use_container_width=True, hide_index=True)

with col_manifest:
    st.markdown("### 🔒 Cryptographic Manifest")
    if manifest_file.exists():
        with open(manifest_file, encoding="utf-8") as mf:
            m_data = json.load(mf)
        st.json(m_data)
    elif receipt_file.exists():
        with open(receipt_file, encoding="utf-8") as rf:
            r_data = json.load(rf)
        st.json(r_data)
    else:
        st.info("No delivery manifest present in this release.")

st.divider()

# Parquet / LeRobot Metadata Inspection
meta_episodes = selected_path / "meta" / "episodes.parquet"
if meta_episodes.exists():
    st.subheader("📊 LeRobot Episode Index (`meta/episodes.parquet`)")
    import pyarrow.parquet as pq

    table = pq.read_table(meta_episodes)
    st.dataframe(table.to_pandas(), use_container_width=True)

# Robomimic HDF5 Inspection
hdf5_file = selected_path / "dataset.hdf5"
if hdf5_file.exists():
    st.subheader("📊 Robomimic HDF5 Structure (`dataset.hdf5`)")
    try:
        import h5py

        with h5py.File(hdf5_file, "r") as hf:
            demos = list(hf["data"].keys())
            st.write(f"Total Demonstrations: **{len(demos)}**")
            if demos:
                first_demo = demos[0]
                obs_keys = list(hf["data"][first_demo]["obs"].keys())
                st.write(f"Sample Demo (`{first_demo}`) Observations:", obs_keys)
                act_shape = hf["data"][first_demo]["actions"].shape
                st.write(f"Action Array Shape: `{act_shape}`")
    except Exception as e:
        st.error(f"Error inspecting HDF5: {e}")

# Delivery Report Tab
if report_file.exists():
    with st.expander("📄 View Client Delivery Report (Markdown)", expanded=False):
        with open(report_file, encoding="utf-8") as rf:
            st.markdown(rf.read())
