"""Dead-Letter Queue (DLQ) & Quarantine Triage Station."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DLQ Triage | EpisodeForge", page_icon="🚨", layout="wide")
st.title("🚨 Dead-Letter Queue (DLQ) & Quarantine Triage")
st.markdown(
    "Automated intake QA isolates non-compliant, unsafe, or corrupted robot demonstrations before transformation."
)

rejected_dir = Path("data/rejected")
if not rejected_dir.exists():
    st.info("No quarantined episodes found. Dead-Letter Queue is empty.")
    st.stop()

# Discover quarantined episode directories
quarantined_dirs = sorted([d for d in rejected_dir.glob("ep_*") if d.is_dir()])
if not quarantined_dirs:
    st.success("Dead-Letter Queue is completely clean! Zero defective episodes in quarantine.")
    st.stop()

st.sidebar.header("Filter & Inspector")
selected_ep = st.sidebar.selectbox("Select Quarantined Episode", [d.name for d in quarantined_dirs])

# Aggregate defect taxonomy
summary_rows = []
for ep_d in quarantined_dirs:
    log_file = ep_d / "rejection_reason.json"
    reason = "Unknown Quality Failure"
    failed_checks = []
    if log_file.exists():
        try:
            with open(log_file, encoding="utf-8") as lf:
                d = json.load(lf)
                reason = d.get("reason", "Validation Failure")
                failed_checks = d.get("failed_checks", [])
        except Exception:
            pass
    summary_rows.append(
        {
            "Episode ID": ep_d.name,
            "Primary Rejection Reason": reason,
            "Failed Checks": ", ".join(failed_checks) if failed_checks else "Schema / Kinematics",
        }
    )

df_summary = pd.DataFrame(summary_rows)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Defect Distribution")
    reason_counts = df_summary["Primary Rejection Reason"].value_counts()
    st.bar_chart(reason_counts)

with col2:
    st.subheader(f"Quarantined Episodes ({len(quarantined_dirs)} Total)")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.divider()

# Deep Dive Inspection of Selected Episode
if selected_ep:
    ep_path = rejected_dir / selected_ep
    st.subheader(f"🔍 Forensic Audit: `{selected_ep}`")

    log_file = ep_path / "rejection_reason.json"
    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            details = json.load(f)
        st.json(details)
    else:
        st.warning("No structured rejection_reason.json manifest found in episode directory.")

    col_files, col_actions = st.columns([2, 1])
    with col_files:
        st.markdown("**Quarantined Files On Disk:**")
        files = [p.name for p in ep_path.glob("**/*") if p.is_file()]
        st.code("\n".join(files))

    with col_actions:
        st.markdown("**Operator Remediation Playbook:**")
        st.button("Mark for Resimulation / Re-capture")
        st.button("Dismiss / Permanent Purge")
