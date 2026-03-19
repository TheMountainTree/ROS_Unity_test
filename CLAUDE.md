# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS2 workspace for EEG-based Brain-Computer Interface (BCI) systems, integrating with Unity for visual stimuli presentation. The system supports SSVEP (Steady-State Visual Evoked Potential) and P300 paradigms for BCI applications.

## Build Commands

```bash
# Build all packages
colcon build

# Build specific package
colcon build --packages-select eeg_processing
colcon build --packages-select publisher_test
colcon build --packages-select ros_tcp_endpoint

# Build with symlink for development (recommended)
colcon build --symlink-install

# Source the workspace after building
source install/setup.bash
```

## Running Nodes

```bash
# Start ROS-TCP-Endpoint (Unity bridge)
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# EEG data TCP receiver (hardware interface)
ros2 run publisher_test eeg_tcp_listener_node

# SSVEP controllers - Node3 and Node4 are current main versions
ros2 run eeg_processing central_controller_ssvep_node3 --ros-args -p run_mode:=pretrain
ros2 run eeg_processing central_controller_ssvep_node4 --ros-args -p run_mode:=decode

# Reasoner communication node2 (external image batch driver for decode/pretrain)
ros2 run eeg_processing ssvep_communication_node2 --ros-args \
  -p run_mode:=decode \
  -p reasoner_mode_enabled:=true

# Reasoner image batch test publisher
ros2 run publisher_test reasoner_publish_test

# UDP trigger sender (for EEG synchronization testing)
ros2 run publisher_test udp_sender_node --ros-args -p trigger_value:=1 -p remote_ip:=192.168.56.3

# Image publishers (for testing)
ros2 run publisher_test image_publisher
ros2 run publisher_test seg_image_publisher

# History image sender (for Unity history display)
ros2 run eeg_processing history_sender_node
```

## Architecture

### System Topology

```
┌─────────────────┐     TCP      ┌──────────────────┐
│  EEG Amplifier  │ ──────────▶  │ eeg_tcp_listener │ ──▶ ROS2 Topics
│ (Neuracle/Win)  │   port 8712  └──────────────────┘
└─────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ROS2 Network                            │
│  ┌───────────────────────┐    ┌─────────────────────────────┐  │
│  │ Central Controller    │◀──▶│  SSVEP/P300 Processing      │  │
│  │ (Node3/Node4)         │    │  (eTRCA/FBCCA pipelines)    │  │
│  └───────────┬───────────┘    └─────────────────────────────┘  │
└──────────────┼──────────────────────────────────────────────────┘
               │ ROS-TCP-Endpoint (Port 10000)
               ▼
┌─────────────────────┐     UDP (Triggers)
│   Unity Frontend    │ ◀────────────────▶ ROS Nodes
│  (Visual Stimulus)  │    Ports 9999/10000/10001/12001
└─────────────────────┘
```

### Packages

**ros_tcp_endpoint** (src/ROS-TCP-Endpoint/)
- Unity Robotics package for TCP bridge between Unity and ROS2
- Entry point: `default_server_endpoint` on port 10000

**eeg_processing** (src/eeg_processing/)
- Core BCI processing nodes
- **CentralControllerSSVEPNode3**: Pretrain-focused node with EEG TCP + trigger-based epoch extraction
- **CentralControllerSSVEPNode4**: Unified node supporting both decode and pretrain, with unified communication config
- **SSVEP_Communication_Node2**: Reasoner/decode/pretrain communication node with static config defaults in `ssvep_communication_node2_config.py`
- **ssvep_pipeline.py**: eTRCA algorithm (requires training data)
- **ssvep_processing_fbcca.py**: FBCCA algorithm (zero-training, reference signals only)

**publisher_test** (src/publisher_test/)
- Test utilities and hardware interfaces
- **eeg_tcp_listener_node**: Core hardware interface, receives EEG data from Neuracle via TCP
- **UdpSenderNode**: Sends trigger values via UDP for EEG synchronization
- **reasoner_publish_test**: Test node for reasoner image batch mode (24 images from ~/Pictures/截图 in 4 groups)

### Key Data Flows

1. **SSVEP Decode Mode (Node4)**:
   - ROS publishes images to `/image_seg` (6 images per trial)
   - ROS sends `prepare` → `stim` via `/ssvep_decode_cmd`
   - Unity displays stimuli, sends `trial_started={trial_id}` via UDP to port 10000
   - ROS sends trigger 1 (stim start) → trigger 2 (stim end) to Windows COM forwarder
   - EEG data captured via TCP from Windows port 8712
   - `stop`/`done` commands stop flashing but keep images displayed until next batch

2. **SSVEP Pretrain Mode (Node3/Node4)**:
   - ROS publishes `cue` → `stim` → `rest` via `/ssvep_train_cmd`
   - Unity displays cues and stimuli
   - Triggers recorded via UDP: trigger 1 (stim start), trigger 2 (stim end)
   - Epoch extraction based on trigger channel in TCP stream

3. **Reasoner Mode (SSVEP_Communication_Node2)**:
   - External node provides image batches via `/reasoner/images`
   - Handshake: `ssvep_ready` ↔ `reasoner_ready` before trial flow
   - Slot layout: row 0 = `0,1,2,3`, row 1 = `4,5,6,7` (3=checkmark, 7=X)
   - `mock_selected_index` parameter simulates user selection
   - Only a small runtime override set remains as ROS params: `run_mode`, `reasoner_mode_enabled`, `mock_selected_index`, `save_dir`, `image_dir`, `decode_max_trials`

### ROS Topics

- `/image_seg`: Image batch for SSVEP (6 images per trial, frame_id contains trial/target metadata)
- `/ssvep_train_cmd`: Training commands for pretrain mode (cue/stim/rest/done)
- `/ssvep_decode_cmd`: Control commands for decode mode (prepare/stim/stop/done)
- `/history_image`: History images for Unity display (100x100 thumbnails)
- `/reasoner/images`: External image batch input for reasoner mode
- `/reasoner/feedback`: Feedback to reasoner node

### Network Configuration (Node4 defaults)

- Unity decode UDP: `0.0.0.0:10000`
- Unity pretrain UDP: `0.0.0.0:10001`
- Ubuntu trigger sender: `192.168.56.103:5006`
- Windows COM forwarder: `192.168.56.3:8888`
- Windows EEG TCP: `192.168.56.3:8712`

### Node2 Static Config

`SSVEP_Communication_Node2.py` now reads most defaults from:

- `src/eeg_processing/eeg_processing/ssvep_communication_node2_config.py`

Runtime ROS parameter overrides are intentionally limited to a small set for联调 convenience.

### EEG Data Format

Windows TCP stream format per sample:
```
Ch1(4B) -> Ch2(4B) -> ... -> ChN(4B) -> Trigger(4B) -> Next Ch1...
```

Last float32 in each frame is the trigger channel (1=stim start, 2=stim end).

## Validation Tools

```bash
# Validate pretrain data (Node3)
python3 src/eeg_processing/eeg_processing/validate_ssvep3_npy.py

# Validate decode data (Node4)
python3 src/eeg_processing/eeg_processing/validate_ssvep4_npy.py
```

## Dependencies

- ROS2 (ament_python build system)
- numpy, scipy, PIL
- brainda (brainda module for SSVEP eTRCA algorithm)
- MNE (optional, for P300 data loading)

## SSVEP Algorithm Notes

The codebase implements two SSVEP decoding approaches:

1. **eTRCA** in `ssvep_pipeline.py`: Ensemble Task-Related Component Analysis - requires training data to learn spatial filters. Higher accuracy with sufficient training.

2. **FBCCA** in `ssvep_processing_fbcca.py`: Filter-Bank Standard CCA - uses sinusoidal reference signals, no training required.

Both use filterbank decomposition with configurable passbands.

## Image Coordinate Note

ROS/OpenCV image origin is top-left (Y down), Unity origin is bottom-left (Y up). Images published to Unity must be vertically flipped.
