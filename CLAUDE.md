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

# Build with symlink for development
colcon build --symlink-install

# Source the workspace after building
source install/setup.bash
```

## Running Nodes

```bash
# Start ROS-TCP-Endpoint (Unity bridge)
ros2 launch ros_tcp_endpoint endpoint.py

# P300 calibration controller
ros2 run eeg_processing central_controller_node

# SSVEP controllers (different versions for different modes)
ros2 run eeg_processing central_controller_ssvep_node
ros2 run eeg_processing central_controller_ssvep_node2 --ros-args -p run_mode:=decode
ros2 run eeg_processing central_controller_ssvep_node2 --ros-args -p run_mode:=pretrain

# UDP trigger sender
ros2 run publisher_test udp_sender_node --ros-args -p trigger_value:=1 -p remote_ip:=192.168.56.3

# Image publishers (for testing)
ros2 run publisher_test image_publisher
ros2 run publisher_test seg_image_publisher
```

## Architecture

### System Topology

```
┌─────────────┐     TCP      ┌─────────────┐     UDP      ┌─────────────┐
│   Unity     │◄────────────►│  ROS2 Nodes │◄────────────►│  EEG/Trigger│
│ (Visual UI) │   port 10000 │ (Controller)│   port 9999  │   Devices   │
└─────────────┘              └─────────────┘              └─────────────┘
```

### Packages

**ros_tcp_endpoint** (src/ROS-TCP-Endpoint/)
- Unity Robotics package for TCP bridge between Unity and ROS2
- Entry point: `default_server_endpoint` on port 10000
- Receives images from ROS, forwards to Unity for display

**eeg_processing** (src/eeg_processing/)
- Core BCI processing nodes
- CentralControllerNode: P300 calibration (generates red/white target images)
- CentralControllerSSVEPNode*: SSVEP paradigms with frequency mapping
- ssvep_pipeline.py: FBTRCA/eTRCA algorithm (requires training data)
- ssvep_processing_fbcca.py: FBSCCA algorithm (zero-training, reference signals only)

**publisher_test** (src/publisher_test/)
- Test utilities for image publishing and UDP trigger handling
- UdpSenderNode: Sends trigger values via UDP for EEG synchronization
- ImagePublisher: Test image publisher for Unity integration

### Key Data Flows

1. **SSVEP Decode Mode**: ROS publishes images to `/image_seg` → Unity displays with frequency-specific flashing → Unity sends `trial_started` via UDP → ROS times the trial duration

2. **SSVEP Pretrain Mode**: ROS publishes commands to `/ssvep_train_cmd` (cue/stim/rest) → Unity displays stimuli → Triggers recorded via UDP on port 9999

3. **P300 Calibration**: ROS generates target (red) / non-target (white) images → Unity performs row/column flashing → Trigger signals recorded for EEG alignment

### ROS Topics

- `/image_seg`: Image batch for SSVEP (6 images per trial, header.frame_id contains trial/target metadata)
- `/ssvep_train_cmd`: Training commands for pretrain mode (std_msgs/String)
- `/fetch_head/rgb/image_raw`: Test image topic

### UDP Ports

- 10000: TCP endpoint (Unity-ROS bridge)
- 9999: Trigger signal receiver (CentralControllerNode)
- 10000: Trial start signal receiver (SSVEP nodes)
- 8888: Trigger forwarder to serial device (Windows side)

## Dependencies

- ROS2 (ament_python build system)
- numpy, scipy, PIL
- metabci (brainda module for SSVEP algorithms)
- MNE (optional, for data loading)

## SSVEP Algorithm Notes

The codebase implements two SSVEP decoding approaches:

1. **FBTRCA (eTRCA)** in `ssvep_pipeline.py`: Ensemble Task-Related Component Analysis - requires training data to learn spatial filters. Higher accuracy with sufficient training.

2. **FBSCCA** in `ssvep_processing_fbcca.py`: Filter-Bank Standard CCA - uses sinusoidal reference signals, no training required. Good for zero-training scenarios.

Both use filterbank decomposition with configurable passbands.