"""DuckDB Lineage & Provenance Catalog Explorer."""

from __future__ import annotations

from pathlib import Path

import duckdb
import streamlit as st

st.set_page_config(page_title="Lineage Explorer | EpisodeForge", page_icon="📜", layout="wide")
st.title("📜 Data Lineage & Audit Catalog")
st.markdown(
    "Track provenance from raw MuJoCo / RigSim seed through transformations to delivered S3 release artifacts."
)

db_path = Path("data/lineage_catalog.duckdb")
if not db_path.exists():
    st.info("Lineage catalog database not created yet. Execute a pipeline run to log provenance.")
    st.stop()

try:
    conn = duckdb.connect(str(db_path), read_only=True)
except Exception:
    conn = duckdb.connect(str(db_path))

st.sidebar.header("Query Presets")
preset = st.sidebar.radio(
    "Select Entity Table",
    ["runs", "episodes", "deliveries", "recalls", "Custom SQL"],
)

default_queries = {
    "runs": "SELECT * FROM runs ORDER BY timestamp_utc DESC LIMIT 50",
    "episodes": "SELECT * FROM episodes ORDER BY created_at DESC LIMIT 50",
    "deliveries": "SELECT * FROM deliveries ORDER BY completed_at_utc DESC LIMIT 50",
    "recalls": "SELECT * FROM recalls ORDER BY timestamp_utc DESC LIMIT 50",
    "Custom SQL": "SELECT client, COUNT(*) as count, AVG(throughput_eps_hr) as avg_throughput FROM runs GROUP BY client",
}

sql_input = st.text_area("SQL Execution Query", value=default_queries[preset], height=100)

if st.button("Execute Query", type="primary"):
    try:
        df = conn.execute(sql_input).df()
        st.write(f"Returned **{len(df)}** rows:")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"SQL Execution Error: {e}")
else:
    try:
        df = conn.execute(sql_input).df()
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"SQL Execution Error: {e}")
