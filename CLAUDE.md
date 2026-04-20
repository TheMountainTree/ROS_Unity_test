# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS2 workspace for EEG-based Brain-Computer Interface (BCI) systems, integrating with Unity for visual stimuli presentation. Supports SSVEP and P300 paradigms. The system runs on Ubuntu (ROS2 core) communicating with Windows (Unity + EEG amplifier software via Neuracle).

## Build Commands

```bash
colcon build --symlink-install                           # Development build (recommended)
colcon build --packages-select eeg_processing            # Build single package
colcon test --packages-select eeg_processing             # Run tests
colcon test-result --verbose                             # Inspect test failures
source install/setup.bash                                # Source workspace after build
python3 src/eeg_processing/eeg_processing/validate_ssvep4_npy.py  # Validate Node4_test output
python3 src/eeg_processing/eeg_processing/validate_ssvep3_npy.py  # Validate Node3 output
```

For refactors, validate with `python -m py_compile` on touched modules plus `colcon build --packages-select eeg_processing`.

## Key Nodes

```bash
# Unity communication bridge (must start first)
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# Node4_test — current main version (FBCCA runtime decoding)
ros2 run eeg_processing ssvep_communication_node4_test --ros-args -p run_mode:=pretrain
ros2 run eeg_processing ssvep_communication_node4_test --ros-args -p run_mode:=decode -p reasoner_mode_enabled:=true

# Reasoner image batch test (multi-stage: object -> category -> activity)
ros2 run publisher_test reasoner_publish_test_3_test

# History image sender for Unity display
ros2 run eeg_processing history_sender_node
```

**Warning:** `eeg_bypass_debug:=true` completely disables EEG TCP reception and trigger sending. In decode mode this causes `_consume_reasoner_selection()` to always return -1, creating an infinite retry loop. Only use bypass in pretrain mode for UI/logic testing.

## Architecture

### System Topology

```
EEG Amplifier (Windows) ──TCP 8712──▶ Node4_test (EEG TCP client) ──▶ CircularEEGBuffer
                                                  │
                           ┌──────────────────────┼───────────────────────┐
                           │              ROS2 Network                    │
                           │  Central Controller (Node4_test)             │
                           │  ┌─ DecodeModule  (FBCCA decode state machine)
                           │  ├─ PretrainModule (EEG TCP + epoch capture)
                           │  └─ ReasonerModule (image batch + selection)
                           └──────────────┬──────────────────────────────┘
                                          │ ROS-TCP-Endpoint (Port 10000)
                                          ▼
                           Unity Frontend ◀────▶ UDP Triggers
                           (Visual Stimulus)     Ports 9999/10000/10001/12001
```

Note: Node4_test connects to EEG TCP server directly (not via `eeg_tcp_listener_node`). The `eeg_tcp_listener_node` is a standalone utility for raw EEG UDP/TCP streaming.

### Node4_test Architecture (Mixin Composition)

```python
class CentralControllerSSVEPNode4Test(DecodeModule, PretrainModule, ReasonerModule, Node):
```

**Module files (all in `src/eeg_processing/eeg_processing/`):**
- `SSVEP_Communication_Node4_test.py` — Main node, init, timer loop, cleanup
- `decode_2_test.py` — `DecodeModule`: decode state machine, FBCCA invocation, label-to-slot mapping
- `pretrain_2_test.py` — `PretrainModule`: EEG TCP polling, trigger processing, epoch capture, dataset save
- `reasoner_2_test.py` — `ReasonerModule`: reasoner handshake, image batch protocol, selection/confirm/undo
- `ssvep_communication_node4_test_config.py` — All static config via dataclasses (`FBCCARuntimeConfig`, etc.)
- `ssvep_runtime_fbcca.py` — `SSVEPFBCCARuntime`: preprocessing + FBCCA decode wrapper (uses metabci)
- `utils.py` — `CircularEEGBuffer` (ring buffer with absolute indexing), `NodeState`, `TrialState`

**Maintenance rule:** Keep only wiring/parameters in `SSVEP_Communication_Node4_test.py`; edit behavior in the `_2_test` modules. All static defaults belong in `ssvep_communication_node4_test_config.py`; only high-frequency runtime toggles (`run_mode`, `reasoner_mode_enabled`, `mock_selected_index`, `save_dir`, `image_dir`, `decode_max_trials`, `eeg_bypass_debug`) are ROS parameters.

### Decode Mode State Machine

```
INIT_WAIT → DECODE_PUBLISHING → DECODE_HOLD → DECODE_STIMULATING
  → DECODE_WAIT_CAPTURE → REASONER_WAIT_SELECTION
  → (REASONER_WAIT_BATCH or WAITING, then back to DECODE_PUBLISHING)
```

Note: `DECODE_WAIT_START` is defined in `NodeState` enum but not used in the current Node4_test decode flow. The state `WAITING` is used for non-reasoner inter-trial intervals.

### EEG Decode Pipeline (End-to-End)

```
1. TCP frames (8ch + trigger, float32) → _poll_eeg_tcp() → CircularEEGBuffer
2. Trigger detection: trigger=1 records stim_start_abs, trigger=2 calls _capture_epoch()
3. _capture_epoch() → eeg_ring.get_range() → raw epoch (n_channels, ~4000 samples at 1000Hz)
4. _perform_eeg_decoding() → _decode_epoch(epoch)
5. SSVEPFBCCARuntime.decode_epoch():
   a. _apply_manual_channel_exclusion(): drop channels listed in manual_bad_channels
   b. preprocess_epoch(): demean → detrend → bandpass 6-100Hz → notch 50/100Hz → resample to 256Hz
   c. _ensure_estimator(): lazy-create FBSCCA with filterbank + sine/cosine references
   d. estimator.predict() → predicted class index (0-based) → mapped to label (1-8)
   e. Filter against active_ui_slots (reject inactive image slot predictions)
6. _map_predicted_to_slot(): label-1 → UI slot (0-7); image slots also checked against current_active_ui_image_slots
7. _handle_reasoner_selection(slot): image(0,1,2,4,5,6) / confirm(3) / undo(7)
```

**Important:** Bad channel handling in Node4_test runtime is manual-only (configured via `manual_bad_channels` in `FBCCARuntimeConfig`). There is no per-epoch automatic MAD-based bad channel detection at runtime — that exists in offline analysis scripts only.

**Important:** Active slot filtering happens twice — once inside `decode_epoch()` (rejects predictions for inactive image slots) and again in `_map_predicted_to_slot()`. This is redundant but harmless.

**Important:** In non-reasoner decode mode, the state machine goes `WAITING → _prepare_decode_trial` and never enters `REASONER_WAIT_SELECTION`, so FBCCA decode results are never consumed. Non-reasoner decode is effectively an EEG data recording mode.

### Frequency and Slot Mapping

**SSVEP frequencies (8 targets, Node4_test config):**
`[8.684, 9.706, 11.0, 11.786, 12.692, 13.75, 15.0, 18.333]` Hz

| Label | Frequency | UI Slot | Type |
|-------|-----------|---------|------|
| 1 | 8.684 Hz | 0 | image |
| 2 | 9.706 Hz | 1 | image |
| 3 | 11.0 Hz | 2 | image |
| 4 | 11.786 Hz | 3 | **confirm** |
| 5 | 12.692 Hz | 4 | image |
| 6 | 13.75 Hz | 5 | image |
| 7 | 15.0 Hz | 6 | image |
| 8 | 18.333 Hz | 7 | **undo** |

**Slot layout:**
```
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │   Row 0: images + confirm(✓)
├─────┼─────┼─────┼─────┤
│  4  │  5  │  6  │  7  │   Row 1: images + undo(✗)
└─────┴─────┴─────┴─────┘
```

Image slots: 0, 1, 2, 4, 5, 6. Function slots: 3 (confirm), 7 (undo). The mapping is label-1 = slot index, so each frequency maps directly to its corresponding UI slot.

**Slot-to-group-image mapping** (reasoner_2_test.py): `{0:0, 1:1, 2:2, 4:3, 5:4, 6:5}` — maps UI slot to 0-based index within the 6-item reasoner batch. Must match `SLOT_TO_ITEM_INDEX` in `reasoner_publish_test_2.py`.

### Reasoner Protocol

Handshake: node publishes `ssvep_ready` → reasoner responds `reasoner_ready`.

Multi-stage flow (via `reasoner_publish_test_2`/`reasoner_publish_test_3_test`): object → category → activity.

Selection actions:
- Image slot (0,1,2,4,5,6): record to history, publish `cmd=selection` to reasoner; if same stage+item_uid as top of history stack, skip as duplicate and restart flashing
- Confirm (3): publish `cmd=confirm` (page-forward/stage-forward)
- Undo (7): pop last action from action stack — if confirm, publish `cmd=rollback`; if selection, remove history item + publish `cmd=undo_selection` + send history UDP `delete_last`
- Invalid (-1): restart flashing current images (`_start_next_decode_trial_with_current_images`)

Reasoner LLM stream: `reasoner_publish_test_2` publishes JSON events (`type=reset|append|done|error`) on `/reasoner/llm_stream`; Node4_test forwards to `/llm_output_stream` for Unity display.

### ROS Topics

- `/image_seg` — Image batch (6 images/trial, frame_id carries trial/target/freq metadata)
- `/ssvep_decode_cmd` — Decode commands (prepare/stim/stop/done/batch_start/batch_end)
- `/ssvep_train_cmd` — Pretrain commands (cue/stim/rest/done)
- `/reasoner/images` — External image batch input (from reasoner_publish_test_2)
- `/reasoner/feedback` — Feedback to reasoner (selection/confirm/undo/rollback)
- `/history_image` — History thumbnails for Unity (default 140x140)
- `/reasoner/llm_stream` → `/llm_output_stream` — LLM stream forwarding

### Network Configuration (Node4_test defaults)

- Unity decode trigger UDP: `127.0.0.1:9999` (byte marker: 100+target=start, 200+target=end)
- Unity decode ack UDP: `0.0.0.0:10000`
- Unity pretrain trigger UDP: `0.0.0.0:10001`
- Ubuntu trigger sender: `192.168.56.103:5006`
- Windows COM forwarder: `192.168.56.3:8888`
- Windows EEG TCP: `192.168.56.3:8712`
- History UDP: `127.0.0.1:12001`

Note: Unity decode markers (port 9999) and EEG trigger injection (port 8888) are separate systems. Trigger=1/2 via port 8888 is what drives epoch alignment; decode markers on 9999 are for logging only.

### EEG Data Format

TCP stream per frame (little-endian float32):
```
Ch1(4B) → Ch2(4B) → ... → Ch8(4B) → Trigger(4B) = 36 bytes/frame
```
Last float is trigger channel: 1 = stim start, 2 = stim end. Default: 8 channels, 1000 Hz, 9 floats/frame.

### FBCCA Runtime Parameters (Node4_test actual defaults)

Preprocessing: target_srate=256Hz, bandpass=6-100Hz (order 4), notch=[50,100]Hz (Q=35)
Filterbank: 3 subbands, wp=[(6,50),(14,50),(22,50)], ws=[(4,52),(12,52),(20,52)]
Decode: n_components=1, n_harmonics=4, filter weights=(n+1)^(-1.25)+0.25
Bad channel: manual only, default drops ["P4","PO4","O2"] from ["O1","O2","Oz","PO3","PO4","Pz","P3","P4"]

### Unity Side (SSVEP_Stimulus4.cs)

`ROS2SSVEPStimulator2` subscribes to `/image_seg`, `/ssvep_decode_cmd`, `/ssvep_train_cmd`. It renders SSVEP flashing patterns using frame-counter-based on/off patterns built from `ssvepFrequencies` and the detected refresh rate. Decode image routing: `decodeImageIndices = {0,1,2,4,5,6}` maps batch indices 0-5 to UI slots. Confirm/rollback slots (3/7) always show default icons during decode.

## Data Output

- `data/central_controller_ssvep_node4_test/` — Node4_test outputs
- `data/central_controller_ssvep3/` — Node3 outputs
- Files: `*_dataset_*.npy` (EEG epochs), `*_trials_*.csv` (trial info), `*_metadata_*.csv`, `*_mapping_*.csv`, `*_eeg_trials_*.csv`

## Dependencies

- ROS2 (ament_python build system, ament_flake8, ament_pep257 linters)
- numpy, scipy, PIL
- metabci (`metabci.brainda.algorithms` for FBSCCA, generate_filterbank, generate_cca_references)
- MNE (optional, for P300 data loading)

## Image Coordinate Convention

ROS/OpenCV origin: top-left (Y down). Unity origin: bottom-left (Y up). Images published to Unity must be vertically flipped (`np.flipud`).

## Coding Conventions

- Python: PEP 8, 4-space indentation. Lint gates: `ament_flake8` + `ament_pep257`.
- Node/module naming: descriptive snake_case files with CamelCase class names.
- New SSVEP controllers should use shared helpers from `utils.py` (`CircularEEGBuffer`, `NodeState`, `TrialState`) rather than redefining them.
- Static config defaults belong in `ssvep_communication_node*_config.py` dataclasses; only high-frequency runtime toggles are ROS parameters.
- Node4_test decode slot conventions: full UI slots `0..7`; image slots `0,1,2,4,5,6`; confirm=3, undo=7. Decode label mapping: label-1 = slot index, kept aligned with reasoner slot semantics.
- Node4_test decode image metadata is 0-based end-to-end: `img=0..5`, `image_id=0..N-1`, reasoner frame `index=0..5`.
- Node4_test is fixed to runtime FBCCA (no eTRCA path). Algorithm internals stay in `ssvep_runtime_fbcca.py`; controllers call only unified runtime APIs.
- Commit format: short imperative summary, often in Chinese/English. One logical change per commit.
