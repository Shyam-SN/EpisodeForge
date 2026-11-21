# Target Client Formats Specification & Compatibility Matrix

This document defines the exact specifications, versions, directory layouts, and field-by-field mappings targeted by EpisodeForge for external client data deliveries.

---

## 1. Format Version Summary

| Client | Format Target | Specification Version | Reference Library / Loader | Target Environment |
| :--- | :--- | :--- | :--- | :--- |
| **Client A (v1 & v2)** | **LeRobot** | `v3.0` (with `v2.1` fallback compat) | `lerobot>=0.3.0` (`LeRobotDataset`) | `validators/lerobot` |
| **Client B** | **RLDS / Open X** | `open_x_v0.1` (TFDS format) | `tensorflow-datasets>=4.9.0`, `rlds` | `validators/rlds` |
| **Client C** | **robomimic HDF5** | `robomimic_v0.2` | `robomimic>=0.3.0` / `h5py>=3.10` | `validators/robomimic` |
| **Client D** | **WebDataset** | `wds_v1` | `webdataset>=0.2.86` / `tarfile` | Standard / Core |

---

## 2. LeRobot Dataset Format (`v3.0` & `v2.1`)

LeRobot (by Hugging Face) standardizes imitation learning datasets. EpisodeForge targets the **v3.0 chunked dataset specification** (with backward compatibility for v2.1 per-episode layouts).

### 2.1 Directory Layout (`v3.0`)
```
<dataset_root>/
├── meta/
│   ├── info.json                      # Dataset schema, features, fps, total_episodes, codebase_version
│   ├── stats.json                     # Feature-wise normalisation statistics (mean, std, min, max, q01, q99)
│   ├── tasks.parquet                  # Mapping of task_index -> natural language task description
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet       # Episode index table: episode_index, length, start/end timestamps
├── data/
│   └── chunk-000/
│       └── file-000.parquet           # High-rate numerical features: observation.state, action, timestamp
└── videos/
    └── chunk-000/
        ├── observation.images.top/
        │   └── file-000.mp4           # Video stream chunk for front camera
        └── observation.images.wrist/
            └── file-000.mp4           # Video stream chunk for wrist camera
```

### 2.2 Field-by-Field Schema Mapping (`meta/info.json`)
```json
{
  "codebase_version": "v3.0",
  "fps": 30,
  "robot_type": "franka_emika_panda",
  "total_episodes": 1000,
  "total_frames": 180000,
  "total_tasks": 3,
  "total_chunks": 1,
  "chunks_size": 1000,
  "features": {
    "action": {
      "dtype": "float32",
      "shape": [8],
      "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper"]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [8],
      "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper"]
    },
    "observation.images.top": {
      "dtype": "video",
      "shape": [256, 256, 3],
      "names": ["height", "width", "channels"],
      "info": {"video.fps": 30.0, "video.codec": "av1", "video.pix_fmt": "yuv420p"}
    },
    "observation.images.wrist": {
      "dtype": "video",
      "shape": [256, 256, 3],
      "names": ["height", "width", "channels"],
      "info": {"video.fps": 30.0, "video.codec": "av1", "video.pix_fmt": "yuv420p"}
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [1],
      "names": null
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "episode_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "task_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    }
  }
}
```

### 2.3 Verification Loader Code
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(root=dataset_path)
assert len(dataset) > 0
item = dataset[0]
assert "observation.state" in item
assert "action" in item
```

---

## 3. RLDS / Open X-Embodiment Dataset Format (`open_x_v0.1`)

RLDS (Reinforcement Learning Datasets) encodes datasets as a sequence of nested steps inside standard TensorFlow Datasets (TFDS).

### 3.1 Directory Layout
```
<dataset_root>/
├── panda_tabletop_rlds/
│   └── 0.1.0/
│       ├── dataset_info.json          # TFDS metadata, citation, splits, description
│       ├── features.json             # Serialized tfds.features.FeaturesDict definition
│       ├── panda_tabletop_rlds-train.tfrecord-00000-of-00004
│       ├── panda_tabletop_rlds-train.tfrecord-00001-of-00004
│       ├── panda_tabletop_rlds-val.tfrecord-00000-of-00001
│       └── metadata.json
```

### 3.2 Step-Level Feature Dictionary
Each episode record contains an `episode_metadata` dict and a `steps` dataset:
- `steps`:
  - `observation`:
    - `image`: `tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg')`
    - `state`: `tfds.features.Tensor(shape=(8,), dtype=tf.float32)`
  - `action`: `tfds.features.Tensor(shape=(7,), dtype=tf.float32)` [dx, dy, dz, roll, pitch, yaw, gripper]
  - `discount`: `tf.float32` (1.0 for ongoing, 0.0 at terminal)
  - `reward`: `tf.float32` (0.0 intermediate, 1.0 on success)
  - `is_first`: `tf.bool`
  - `is_last`: `tf.bool`
  - `is_terminal`: `tf.bool`
  - `language_instruction`: `tf.string`

### 3.3 Verification Loader Code
```python
import tensorflow_datasets as tfds

builder = tfds.builder_from_directory(dataset_dir)
ds = builder.as_dataset(split='train')
for episode in ds.take(1):
    for step in episode['steps']:
        obs = step['observation']
        act = step['action']
        break
```

---

## 4. robomimic HDF5 Dataset Format (`robomimic_v0.2`)

The robomimic format is an HDF5-based storage layout heavily adopted in robot imitation learning benchmarks.

### 4.1 HDF5 Internal Hierarchy
```
panda_tabletop_robomimic.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions: (T, 7) float32        # [dx, dy, dz, ax, ay, az, gripper]
│   │   ├── dones: (T,) int64               # 0 until final step (1)
│   │   ├── rewards: (T,) float32           # 0.0 or 1.0
│   │   ├── states: (T, 8) float32          # joint states + gripper width
│   │   ├── obs/
│   │   │   ├── agentview_image: (T, 84, 84, 3) uint8
│   │   │   └── robot0_eye_in_hand_image: (T, 84, 84, 3) uint8
│   │   └── attrs:
│   │       ├── num_samples: int (T)
│   │       └── model_file: str (MJCF XML)
│   ├── demo_1/
│   │   └── ...
│   └── attrs:
│       ├── total: int (sum of num_samples across all demos)
│       └── env_args: JSON string {"env_name": "PandaTabletop", "type": 1, "env_kwargs": {...}}
└── mask/
    ├── train: array of string keys ["demo_0", "demo_2", ...]
    ├── valid: array of string keys ["demo_1", ...]
    └── successful: array of string keys (episodes where task succeeded)
```

### 4.2 Verification Loader Code
```python
import h5py

with h5py.File(hdf5_path, "r") as f:
    assert "data" in f
    assert "mask/train" in f
    total_samples = f["data"].attrs["total"]
    demo_keys = [k for k in f["data"].keys()]
    assert len(demo_keys) > 0
    demo = f["data"][demo_keys[0]]
    assert "obs/agentview_image" in demo
    assert "actions" in demo
    assert demo.attrs["num_samples"] == demo["actions"].shape[0]
```

---

## 5. WebDataset Format (`wds_v1`)

WebDataset formats multi-sensor data as sequential `.tar` archives containing identically keyed files for rapid streaming.

### 5.1 Archive File Contents per Clip
For episode index `000042`:
- `000042.mp4`: Lossless / high-efficiency HEVC or H.264 video stream (1280x720)
- `000042.json`: Clip metadata (duration, frame count, camera intrinsics, QA verdict)
- `000042.imu.json`: Resampled or synchronized IMU accelerometry and gyroscopy time-series
- `000042.calib.json`: Extrinsic calibration matrix to reference body frame

