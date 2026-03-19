# QWEN.md - ROS2 Unity BCI Project Context

## Project Overview

This is a **ROS2 (Robot Operating System 2) workspace** for a **Brain-Computer Interface (BCI) system** integrated with Unity. The system handles EEG (electroencephalography) signal acquisition, real-time processing (SSVEP and P300 paradigms), and low-latency communication with a Unity frontend for visual stimulus presentation.

### Core Technologies
- **ROS2 (Humble/Iron)**: Distributed middleware for inter-process communication
- **Python 3**: Primary language for ROS nodes and signal processing
- **Unity (C#)**: Visual stimulus rendering and VR interface
- **NumPy/SciPy**: Signal processing and machine learning
- **TCP/UDP**: Hybrid communication for data streaming and time-critical triggers

### Current Mainline
当前维护中的联调通信主线为 **`SSVEP_Communication_Node2.py`**：
- **decode 模式**：标准解码采集 + Reasoner 外部图片分组模式
- **pretrain 模式**：预训练数据采集
- **Reasoner 交互模式**：24图分组测试、双节点握手、history 回传/撤销
- **配置策略**：大部分默认值来自 `ssvep_communication_node2_config.py`，只保留少量 ROS 参数用于运行时覆盖

### Architecture Summary
```
┌─────────────────┐     TCP      ┌──────────────────┐
│  EEG Amplifier │ ──────────▶  │ eeg_tcp_listener │ ──▶ ROS2 Topics
└─────────────────┘              └──────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────┐
│                    ROS2 Network                         │
│  ┌───────────────────┐    ┌─────────────────────────┐  │
│  │ Central Controller│◀──▶│  SSVEP/P300 Processing  │  │
│  │ (State Machine)  │    │  (Decoding Pipeline)    │  │
│  └─────────┬─────────┘    └─────────────────────────┘  │
│            │                                           │
└────────────┼───────────────────────────────────────────┘
             │ ROS-TCP-Endpoint
             ▼
┌─────────────────────┐     UDP (Triggers)
│   Unity Frontend    │ ◀────────────────▶ ROS Nodes
│  (Visual Stimulus)  │    Port 9999/10000/10001/12001
└─────────────────────┘
```

---

## Project Structure

```
ROS_Unity_test/
├── src/                          # ROS2 packages (source)
│   ├── eeg_processing/           # Core BCI logic package
│   │   ├── eeg_processing/       # Python module
│   │   │   ├── CentralController*.py   # Experiment control nodes
│   │   │   ├── ssvep_*.py              # SSVEP processing algorithms
│   │   │   ├── p300_*.py               # P300 processing algorithms
│   │   │   ├── history_sender.py       # UDP state sync to Unity
│   │   │   └── *.cs                   # Unity C# scripts (reference copies)
│   │   ├── package.xml
│   │   └── setup.py
│   ├── publisher_test/           # Hardware abstraction & test utilities
│   │   ├── publisher_test/
│   │   │   ├── eeg_tcp_listener_node.py  # TCP client for EEG amplifier
│   │   │   ├── image_publisher.py        # Test image publisher
│   │   │   ├── reasoner_publish_test.py  # Reasoner group test node
│   │   │   └── udp_sender_node.py        # Test UDP trigger sender
│   │   └── setup.py
│   └── ROS-TCP-Endpoint/         # Unity-ROS bridge (external package)
├── data/                         # Recorded trials, datasets, plots
│   ├── central_controller/
│   ├── central_controller_ssvep2/
│   ├── central_controller_ssvep3/
│   ├── central_controller_ssvep_train/
│   └── analysis/
├── dev_logs/                     # Development notes (Chinese)
├── build/                        # Colcon build artifacts
├── install/                      # Installed packages
└── log/                          # Build logs
```

---

## Build, Run, and Test Commands

### Building
```bash
# From workspace root
colcon build --symlink-install              # Build all packages
colcon build --packages-select eeg_processing  # Build single package
source install/setup.bash                   # Load into environment
```

### Running Nodes
```bash
# Start Unity TCP endpoint
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# SSVEP decode mode (Node4 - latest)
ros2 run eeg_processing central_controller_ssvep_node4 --ros-args -p run_mode:=decode

# SSVEP pretrain mode
ros2 run eeg_processing central_controller_ssvep_node4 --ros-args -p run_mode:=pretrain

# EEG TCP listener (connects to amplifier)
ros2 run publisher_test eeg_tcp_listener_node

# History sender (UDP sync to Unity)
ros2 run eeg_processing history_sender_node

# Reasoner group test (24 images / 4 groups)
ros2 run publisher_test reasoner_publish_test

# SSVEP Communication Node2 (reasoner mode)
ros2 run eeg_processing ssvep_communication_node2 --ros-args \
  -p run_mode:=decode \
  -p reasoner_mode_enabled:=true
```

### Testing
```bash
colcon test --packages-select eeg_processing publisher_test
colcon test-result --verbose
```

### Linting
```bash
ament_flake8 src/eeg_processing/eeg_processing/
ament_pep257 src/eeg_processing/eeg_processing/
```

### Data Validation
```bash
# Validate pretrain dataset
python3 src/eeg_processing/eeg_processing/validate_ssvep3_npy.py

# Validate decode dataset
python3 src/eeg_processing/eeg_processing/validate_ssvep4_npy.py
```

---

## Key ROS2 Nodes and Entry Points

### eeg_processing Package
| Entry Point | Module | Description |
|-------------|--------|-------------|
| `central_controller_node` | CentralControllerNode.py | Basic experiment controller |
| `central_controller_ssvep_node` | CentralControllerSSVEPNode.py | SSVEP controller (v1) |
| `central_controller_ssvep_node2` | CentralControllerSSVEPNode2.py | SSVEP controller (v2, dual-mode) |
| `central_controller_ssvep_node3` | CentralControllerSSVEPNode3.py | SSVEP controller (v3, online fusion) |
| `central_controller_ssvep_node4` | CentralControllerSSVEPNode4.py | SSVEP controller (v4, unified config) |
| `central_controller_ssvep_train_node` | CentralControllerSSVEPTrainNode.py | SSVEP training data collector |
| `history_sender_node` | history_sender.py | UDP state synchronization |
| `ssvep_communication_node` | SSVEP_Communication_Node.py | Reasoner mode communication |
| `ssvep_communication_node2` | SSVEP_Communication_Node2.py | Reasoner/decode/pretrain communication node with static config module |

### publisher_test Package
| Entry Point | Module | Description |
|-------------|--------|-------------|
| `eeg_tcp_listener_node` | eeg_tcp_listener_node.py | TCP client for EEG amplifier data |
| `image_publisher` | image_publisher.py | Test image publisher |
| `seg_image_publisher` | seg_image_publisher.py | Segmentation image publisher |
| `udp_sender_node` | udp_sender_node.py | Test UDP trigger sender |
| `reasoner_publish_test` | reasoner_publish_test.py | Reasoner group test (24 images) |

---

## Signal Processing Pipelines

### SSVEP (Steady-State Visual Evoked Potential)
- **Algorithms**: eTRCA (Ensemble Task-Related Component Analysis), FBCCA (Filter Bank Canonical Correlation Analysis)
- **Pipeline**: `ssvep_pipeline.py` provides modular components:
  - `SSVEPDataLoader`: Dataset loading (MNE format, Nakanishi2015)
  - `SSVEPPretrainer`: Spatial filter training with `.fit(X, y)` and `.save()`
  - `SSVEPDecoder`: Real-time inference with `.from_file()` and `.decode(X)`
  - `SSVEPEvaluator`: Cross-validation evaluation

### P300 (Event-Related Potential)
- **Processing**: Bandpass filtering, baseline correction, EOG artifact removal
- **Files**: `p300_processing_test.py` (offline), `p300_processing_test_online.py` (online system)
- **Components**: `OnlineEpochPreprocessor`, `CircularEEGBuffer`, `TriggeredEpochExtractor`

---

## Communication Protocols

### ROS2 Topics
| Topic | Message Type | Purpose |
|-------|--------------|---------|
| `/eeg_data` | `Float32MultiArray` | Raw EEG samples from amplifier |
| `/image_seg` | `sensor_msgs/Image` | Image segmentation data for Unity |
| `/ssvep_train_cmd` | `std_msgs/String` | Training commands to Unity |
| `/history_image` | `sensor_msgs/Image` | History images for Unity display |
| `/history_control` | `std_msgs/String` | History control commands |
| `/reasoner/images` | Custom | Reasoner image batches |
| `/reasoner/feedback` | Custom | Reasoner feedback commands |

### Node2 Runtime Overrides

`SSVEP_Communication_Node2.py` keeps only these ROS parameters at runtime:
- `run_mode`
- `reasoner_mode_enabled`
- `mock_selected_index`
- `save_dir`
- `image_dir`
- `decode_max_trials`

All other defaults come from `src/eeg_processing/eeg_processing/ssvep_communication_node2_config.py`.

### UDP Ports (Time-Critical Triggers)
| Port | Direction | Purpose |
|------|-----------|---------|
| 9999 | Unity → ROS | Decode markers |
| 10000 | ROS ← Unity | Decode ACK / trial started |
| 10001 | ROS ← Unity | Training markers |
| 12001 | ROS → Unity | History control (delete_last, etc.) |
| 5006 | ROS → Windows | Trigger forwarding |
| 8888 | Windows ← ROS | Windows COM trigger receiver |

### TCP Configuration
- EEG amplifier connects to TCP listener (configurable host/port)
- Default: `192.168.56.3:8712` (Windows Neuracle TCP forward)
- `TCP_NODELAY` enabled for low-latency streaming
- Buffer handling for TCP packet fragmentation
- Data format: `Ch1(4B) → Ch2(4B) → ... → ChN(4B) → Trigger(4B) → Next Ch1...`

---

## Coding Conventions

### Python (ROS Nodes)
- **Style**: PEP 8 with 4-space indentation
- **Docstrings**: PEP 257 format
- **Naming**: `snake_case` for files/modules, `CamelCase` for classes
- **Node naming**: Descriptive names like `CentralControllerSSVEPNode4`

### C# (Unity Scripts)
- Located in `eeg_processing/eeg_processing/` as reference copies
- Files: `SSVEP_Stimulus.cs`, `SSVEP_Stimulus2.cs`, `SSVEP_Train.cs`, `P300_Stimulus.cs`, `HistoryManager.cs`

### Commit Style
- Short, focused commits with imperative summaries
- Format: `<scope>: <description>` (e.g., `eeg_processing: refine UDP listener timeout`)
- Note parameter/port changes in commit messages

---

## Dependencies

### Python (eeg_processing)
- `numpy`, `scipy` - Signal processing
- `brainda` - BCI algorithms (optional, lazy-imported)
- `rclpy` - ROS2 Python client
- `std_msgs`, `sensor_msgs`, `cv_bridge` - ROS2 messages
- `PIL` (Pillow) - Image processing
- `opencv-python` - Computer vision

### System
- ROS2 Humble or later
- Python 3.8+
- OpenCV (`python3-opencv`)
- Pillow (`python3-pil`)

---

## Data Directory Structure

```
data/
├── central_controller/           # Basic controller recordings
├── central_controller_ssvep2/    # SSVEP v2 experiment data
├── central_controller_ssvep3/    # SSVEP v3 experiment data
├── central_controller_ssvep_train/  # Training data
└── analysis/                     # Analysis results and plots
```

---

## Development Notes

### Version Evolution
- **SSVEP Stimulus**: `SSVEP_Stimulus.cs` → `SSVEP_Stimulus2.cs` (unified decode/train modes)
- **Central Controller**: v1 → v2 (dual-mode) → v3 (online fusion with `CircularEEGBuffer`) → v4 (unified config)
- **Communication**: UDP → TCP for EEG data (reliability improvement)

### Key Design Principles
1. **Separation of Concerns**: Controllers handle state machines and timing; separate nodes handle data acquisition
2. **Hybrid Communication**: ROS2 for structured data, UDP for time-critical triggers
3. **Modular Pipelines**: Signal processing algorithms encapsulated in reusable classes

### Recent Features (Node4 & Reasoner Mode)
- **Unified Communication Config**: Shared network parameters for decode/pretrain modes
- **EEG TCP + Trigger Integration**: Decode mode now captures EEG data with trigger markers
- **Reasoner Test Mode**: 24-image/4-group testing with handshake protocol
- **History Management**: Fixed-size thumbnails (100x100) with add/delete/clear operations

---

## Related Documentation
- `Architecture.md` - Detailed system architecture (Chinese)
- `AGENTS.md` - Repository guidelines and conventions
- `FunctionLog.md` - Function and class reference (Chinese)
- `dev_logs/` - Daily development notes (Chinese)
