# Repository Guidelines

## Self-update rule
After major code changes, update this AGENTS.md to reflect:
- architecture
- conventions
- dependencies

## Project Structure & Module Organization
ROS2 (ament_python) workspace for Unity-integrated BCI workflows (SSVEP, P300).

| Package | Purpose |
|---|---|
| `src/eeg_processing/` | Core EEG/BCI logic, controllers, FBCCA decoding. Entry points in `setup.py`. |
| `src/publisher_test/` | Utility publishers, UDP trigger sender, TCP listener, reasoner test nodes. |
| `src/ROS-TCP-Endpoint/` | Unity ROS TCP bridge (`ros_tcp_endpoint`). |
| `src/AgenticReasoner/` | AI agent framework for ROS (no console_scripts registered). |
| `src/EEG_Analysis/` | EEG analysis library (no console_scripts registered). |
| `data/` | Recorded trials, mappings, generated datasets/plots. |
| `dev_logs/` | Development notes. |

Keep node code inside each package module and package metadata in `package.xml`, `setup.py`, `setup.cfg`.

### Two Node4 variants (do not confuse)
- `central_controller_ssvep_node4` (`CentralControllerSSVEPNode4.py`): original monolithic controller; README claims it auto-trains eTRCA models in pretrain mode.
- `ssvep_communication_node4_test` (`SSVEP_Communication_Node4_test.py`): **modular FBCCA-only controller**. Pretrain is data-recording only (no model training); decode uses runtime FBCCA via `ssvep_runtime_fbcca.py`.

### Module separation rules
Do not move behavior logic back into main node files.
- Node2: `SSVEP_Communication_Node2.py` + `ssvep_communication_node2_config.py`.
- Node3: wiring only in `SSVEP_Communication_Node3.py`; behavior in `decode.py` / `pretrain.py` / `reasoner.py`.
- Node3_1: wiring only in `SSVEP_Communication_Node3_1.py`; behavior in `decode_1.py` / `pretrain_1.py` / `reasoner_1.py`.
- Node4_test: wiring only in `SSVEP_Communication_Node4_test.py`; behavior in `decode_2_test.py` / `pretrain_2_test.py` / `reasoner_2_test.py`.

### Shared utilities
- `eeg_processing/utils.py`: `CircularEEGBuffer`, `NodeState` enum, `TrialState`. Prefer these over redefining helpers.
- `ssvep_communication_node4_test_config.py`: `FBCCARuntimeConfig` holds all filterbank/preprocessing defaults for Node4_test.

### Stale / unregistered files
- `reasoner_publish_test_3_test.py` exists in the tree but is **not registered in `publisher_test/setup.py`**. The active multi-stage reasoner entry points are `reasoner_publish_test_2` and `reasoner_publish_test_2_local_llm`.

## Build, Test, and Development Commands
Run from workspace root:

- `colcon build --symlink-install`
- `colcon build --packages-select eeg_processing`
- `source install/setup.bash`
- `colcon test --packages-select eeg_processing ros_tcp_endpoint AgenticReasoner EEG_Analysis`
- `colcon test-result --verbose`

Focused verification:
- `python -m py_compile src/eeg_processing/eeg_processing/<module>.py` (minimum pre-build check).
- `python3 src/eeg_processing/eeg_processing/validate_ssvep3_npy.py`
- `python3 src/eeg_processing/eeg_processing/validate_ssvep4_npy.py`

Runtime examples:
- `ros2 launch ros_tcp_endpoint endpoint.py`
- `ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0`
- `ros2 run eeg_processing ssvep_communication_node4_test --ros-args -p run_mode:=decode -p reasoner_mode_enabled:=true`
- `ros2 run publisher_test reasoner_publish_test_2`

**Critical:** `eeg_bypass_debug:=true` completely disables EEG TCP reception and trigger sending. In decode mode this causes `_consume_reasoner_selection()` to always return -1, and the node never leaves `REASONER_WAIT_SELECTION` (effectively an infinite stall). Only use bypass in pretrain mode for UI/logic testing, or ensure `mock_selected_index` is set to a valid slot.

### Image coordinate convention
ROS/OpenCV origin is top-left (Y down). Unity origin is bottom-left (Y up). All images published to Unity must be vertically flipped (`np.flipud`). This convention is applied in every decode module.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, PEP 257 docstrings.
- Lint gates: `ament_flake8` and `ament_pep257` (see each package `test/`).
- ROS nodes: descriptive `snake_case` files, `CamelCase` classes.
- Static defaults belong in the corresponding `ssvep_communication_node*_config.py`. Only high-frequency runtime toggles should be ROS parameters: `run_mode`, `reasoner_mode_enabled`, `mock_selected_index`, `save_dir`, `image_dir`, `decode_max_trials`, `eeg_bypass_debug`.

## Protocol & Slot Conventions

### Node4_test slots and indexing
- Full UI slots `0..7`: image slots `0,1,2,4,5,6`, confirm `3`, undo/rollback `7`.
- Decode image metadata is 0-based end-to-end: `img=0..5`, `image_id=0..N-1`, reasoner frame `index=0..5`.
- Preprocessing chain: demean -> detrend -> bandpass (default 6-100Hz) -> notch 50/100Hz at acquisition rate, then resample to decode target srate (default 256Hz).
- Manual bad-channel exclusion via `FBCCARuntimeConfig.manual_bad_channels` with fixed-name mapping from `channel_name_order` (default 8-channel: `O1,O2,Oz,PO3,PO4,Pz,P3,P4`). No per-epoch automatic bad-channel detection.
- Decode timing is ROS-driven: ROS sends decode trigger `1/2` immediately before publishing Unity `stim/stop`; Unity no longer sends `trial_started` UDP back.
- Pretrain completion saves dataset/metadata CSV+NPY only; must not auto-train or emit model sidecars.
- Node4_test connects to the EEG TCP server directly (IP `192.168.56.3:8712` by default), **not** via `eeg_tcp_listener_node`. The `eeg_tcp_listener_node` is a standalone utility for raw EEG streaming. Dependencies include `numpy`, `scipy`, `PIL`, and `metabci` (for FBCCA/SCCA algorithms).

### Decode v1 batch protocol (Node3_1 -> Unity)
Publish `cmd=batch_start;trial=...;target=...;count=...` on `/ssvep_decode_cmd`, then image packets on `/image_seg`, then `cmd=batch_end;...`. Unity flashes only active dynamic slots implied by `count` (max 6).

### Reasoner v2 staged protocol (`reasoner_publish_test_2` -> Node3_1 / Node4_test)
- Image frame metadata carries `stage/page/item_uid/item_label`.
- Commands:
  - `confirm`: page/stage forward navigation (ignored if no next page).
  - `rollback`: page backward based on confirm history (cross-stage allowed), clears selections only on the rollback target page.
  - `undo_selection`: cancel a prior staged selection by `stage/page/item_uid` and republish that page.
  - `reuse_page`: keep current page, restart decode `prepare` without re-publishing images (used by A-stage object selection).
  - `done`: close session.
- LLM stream: JSON events on `/reasoner/llm_stream` using `std_msgs/String` (`type=reset|append|done|error`, `stage=activity`, optional `text`); Node3_1 forwards to `/llm_output_stream` for Unity display.
- Camera conflict policy is code-level (`PREFERRED_CAMERA`), default `camera2` priority.
- LLM settings are code-level constants (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TIMEOUT_S`). Fallback activity candidates must remain available when LLM output is missing/invalid.

### Node3_1 reasoner history and undo
- Appends/publishes history images for staged selections except immediate duplicate re-selections (same stage + same `item_uid` as stack top), which are skipped and directly restart decode `prepare`.
- Slot `7` undoes the latest action from an action stack: `confirm` -> publish `rollback`; `selection` -> remove matching history item + publish `undo_selection` + send history UDP `delete_last`.

## Testing Guidelines
- Framework: `pytest` with ROS ament linters (`test_flake8.py`, `test_pep257.py`, `test_copyright.py` in each package `test/`).
- For communication-node refactors, at minimum validate `python -m py_compile` on the touched module plus `colcon build --packages-select eeg_processing`.
- Before PRs, run `colcon build` and `colcon test`; include `colcon test-result --verbose` output when fixing failures.

## Commit & Pull Request Guidelines
- Short, focused commit subjects (often concise Chinese/English). One logical change per commit.
- Preferred format: imperative summary, optionally scoped (example: `eeg_processing: refine UDP listener timeout`).
- PRs should include: purpose, impacted packages, how to run/verify, and sample logs or screenshots when Unity-facing behavior changes.
- Link related issue/task IDs and note any parameter/port changes (for example UDP `9999`, TCP `10000`).
