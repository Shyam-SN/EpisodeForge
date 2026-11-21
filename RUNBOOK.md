# EpisodeForge: Operations Runbook & Incident Response

> **Audience:** Data Operations Engineers, Platform Engineers, Support On-Call  
> **Status:** Active / Production Runbook

---

## 1. Quick Operator Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Lint Spec** | `forge spec lint configs/clients/client_a_lerobot.v1.yaml` |
| **Diff Specs** | `forge spec diff configs/clients/client_a_lerobot.v1.yaml configs/clients/client_a_lerobot.v2.yaml` |
| **Execute Pipeline** | `forge run --input-dir data/raw/m1_episodes --spec configs/clients/client_a_lerobot.v1.yaml --workers 4` |
| **Validate Release** | `forge validate-out data/releases/client_a/panda_tabletop_lerobot_v1 --spec configs/clients/client_a_lerobot.v1.yaml` |
| **Deliver to MinIO / S3** | `forge deliver data/releases/client_a/panda_tabletop_lerobot_v1 --spec configs/clients/client_a_lerobot.v1.yaml` |
| **Execute Recall** | `forge recall --release-dir data/releases/client_a/panda_tabletop_lerobot_v1 --episodes ep_000002 --reason "Grasp slip"` |
| **Financial Forecast** | `forge plan --episodes 10000 --spec configs/clients/client_a_lerobot.v1.yaml` |
| **Generate Report** | `forge report data/releases/client_a/panda_tabletop_lerobot_v1` |
| **Query Lineage** | `forge lineage runs` / `forge lineage episodes` |
| **Launch Dashboard** | `forge dashboard` (opens on port 8501) |

---

## 2. Standard Production Execution

### Step 1: Verify Storage & Services
Ensure local MinIO or remote S3 is operational:
```bash
docker compose up -d minio
curl -I http://localhost:9000/minio/health/live
```

### Step 2: Running a Production Batch
Run the pipeline with appropriate worker count for your CPU hardware (on Apple Silicon M-series, `--workers 4` or `--workers 8` is recommended):
```bash
forge run \
  --input-dir data/raw/m1_episodes \
  --spec configs/clients/client_a_lerobot.v1.yaml \
  --workers 4
```

### Step 3: Resuming an Interrupted Run
EpisodeForge uses a **content-addressable cache** (`data/cache/`). If a run is killed mid-flight (`SIGINT`, out-of-memory, network drop), rerun the identical command. The runner will automatically detect completed episode artifacts and achieve near 100% cache hits, resuming immediately from the point of failure.

---

## 3. Dead-Letter Queue (DLQ) Triage

When episodes fail input validation (kinematic limit violations, NaN values, excessive timestamp jitter), they are quarantined to `data/rejected/ep_<id>/`.

### Inspecting Quarantined Episodes
1. List all quarantined episodes:
   ```bash
   ls -la data/rejected/
   ```
2. Inspect the forensic reason:
   ```bash
   cat data/rejected/ep_000003/rejection_reason.json
   ```
   Example output:
   ```json
   {
     "episode_id": "ep_000003",
     "timestamp_utc": "2026-11-13T22:30:00Z",
     "reason": "Joint velocity limit exceeded on joint panda_joint4 (max=3.82 rad/s, limit=2.17 rad/s)",
     "failed_checks": ["kinematics_velocity_bounds"]
   }
   ```
3. Use the Streamlit Dashboard for visual triage:
   ```bash
   forge dashboard
   # Navigate to the 'DLQ Triage' tab
   ```

### Remediation Actions
- **Scripted Defect / Simulation Artifact:** If the defect was caused by IK solver singularity, trigger targeted re-simulation with a different seed.
- **Hardware Sensor Glitch (Real Rig):** If the defect is from a physical camera frame drop, discard the recording and flag the camera USB bus.

---

## 4. Data Recall & Patch Procedures

If a client reports defective episodes in a previously accepted release:

1. Identify the list of episode IDs to recall: `ep_000012, ep_000045`.
2. Execute surgical recall:
   ```bash
   forge recall \
     --release-dir data/releases/client_a/panda_tabletop_lerobot_v1 \
     --episodes ep_000012,ep_000045 \
     --reason "Undetected gripper collision during terminal phase" \
     --patch-version v1.0.1
   ```
3. Verify the generated patch:
   - Check `data/releases/client_a/panda_tabletop_lerobot_v1.0.1/recall_notice.json`.
   - Verify `delivery_manifest.json` matches the pruned episode set.
4. Upload patch to cloud:
   ```bash
   forge deliver data/releases/client_a/panda_tabletop_lerobot_v1.0.1 --spec configs/clients/client_a_lerobot.v1.yaml
   ```

---

## 5. Adding a New Client Target Format

To onboard a new ML format (e.g. ROS2 Bag or WebDataset):
1. **Define ClientSpec**: Add target options to `src/episodeforge/config/client_spec.py`.
2. **Implement Writer**: Add a dedicated writer under `src/episodeforge/writers/<format_name>.py`.
3. **Register Dispatcher**: Add format handling in `src/episodeforge/writers/__init__.py`.
4. **Create Isolated Validator**: Create `validators/<format_name>/` with an isolated virtual environment and `validate.py` returning JSON verification status.
5. **Run Integration Test**: Add a test in `tests/test_writers.py` to ensure official loader round-trip compatibility.

---

## 6. Common Incidents & Troubleshooting

### Incident 1: "PyAV VideoToolbox Hardware Encoder Sandbox Error"
- **Symptom:** `av.AVError: Cannot allocate memory` or IPC permission error on macOS.
- **Root Cause:** macOS WindowServer sandbox restricts hardware GPU encoder IPC within sandboxed command runners.
- **Resolution:** EpisodeForge automatically falls back to software encoders (`libsvtav1`, `libx264`). Ensure your spec uses `codec: av1` or `codec: h264`.

### Incident 2: "Official Loader Incompatible Import Error"
- **Symptom:** Dependency conflicts between PyTorch (LeRobot) and TensorFlow (RLDS).
- **Root Cause:** Upstream libraries pin incompatible protobuf or NumPy versions.
- **Resolution:** EpisodeForge executes all official client loaders in dedicated subprocesses under `validators/<client>/`. Never install `lerobot` or `tensorflow` directly into the core pipeline environment.

### Incident 3: "MinIO S3 Connection Refused"
- **Symptom:** `EndpointConnectionError: Could not connect to the endpoint URL: http://localhost:9000`
- **Resolution:** Start or restart Docker Compose:
  ```bash
  docker compose down && docker compose up -d minio
  ```
  Alternatively, test with `--dry-run` to generate cryptographic receipts without remote upload.

