# QWEN.md - ROS2 Unity BCI Project Context

## Project Overview

This is a **ROS2 (Robot Operating System 2) workspace** for a **Brain-Computer Interface (BCI) system** integrated with Unity. The system handles EEG (electroencephalography) signal acquisition, real-time processing (SSVEP and P300 paradigms), and low-latency communication with a Unity frontend for visual stimulus presentation.

### Core Technologies
- **ROS2 (Humble/Iron)**: Distributed middleware for inter-process communication
- **Python 3**: Primary language for ROS nodes and signal processing
- **Unity (C#)**: Visual stimulus rendering and VR interface
- **NumPy/SciPy**: Signal processing and machine learning
- **TCP/UDP**: Hybrid communication for data streaming and time-critical triggers

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
│  (Visual Stimulus)  │    Port 9999/10000/10001
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
│   │   │   └── udp_sender_node.py        # Test UDP trigger sender
│   │   └── setup.py
│   └── ROS-TCP-Endpoint/         # Unity-ROS bridge (external package)
├── data/                         # Recorded trials, datasets, plots
│   ├── central_controller/
│   ├── central_controller_ssvep2/
│   ├── central_controller_ssvep3/
│   └── central_controller_ssvep_train/
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
ros2 launch ros_tcp_endpoint endpoint.py

# SSVEP decode mode
ros2 run eeg_processing central_controller_ssvep_node2 --ros-args -p run_mode:=decode

# SSVEP pretrain mode
ros2 run eeg_processing central_controller_ssvep_node2 --ros-args -p run_mode:=pretrain

# EEG TCP listener (connects to amplifier)
ros2 run publisher_test eeg_tcp_listener_node

# History sender (UDP sync to Unity)
ros2 run eeg_processing history_sender_node
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

---

## Key ROS2 Nodes and Entry Points

### eeg_processing Package
| Entry Point | Module | Description |
|-------------|--------|-------------|
| `central_controller_node` | CentralControllerNode.py | Basic experiment controller |
| `central_controller_ssvep_node` | CentralControllerSSVEPNode.py | SSVEP controller (v1) |
| `central_controller_ssvep_node2` | CentralControllerSSVEPNode2.py | SSVEP controller (v2, dual-mode) |
| `central_controller_ssvep_node3` | CentralControllerSSVEPNode3.py | SSVEP controller (v3, online fusion) |
| `central_controller_ssvep_train_node` | CentralControllerSSVEPTrainNode.py | SSVEP training data collector |
| `history_sender_node` | history_sender.py | UDP state synchronization |

### publisher_test Package
| Entry Point | Module | Description |
|-------------|--------|-------------|
| `eeg_tcp_listener_node` | eeg_tcp_listener_node.py | TCP client for EEG amplifier data |
| `image_publisher` | image_publisher.py | Test image publisher |
| `seg_image_publisher` | seg_image_publisher.py | Segmentation image publisher |
| `udp_sender_node` | udp_sender_node.py | Test UDP trigger sender |

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
| `/image_seg` | Custom | Image segmentation data for Unity |
| `/ssvep_train_cmd` | Custom | Training commands to Unity |

### UDP Ports (Time-Critical Triggers)
| Port | Direction | Purpose |
|------|-----------|---------|
| 9999 | Unity → ROS | Decode markers |
| 10000 | ROS → Unity | Decode ACK |
| 10001 | Unity → ROS | Training markers |

### TCP Configuration
- EEG amplifier connects to TCP listener (configurable host/port)
- `TCP_NODELAY` enabled for low-latency streaming
- Buffer handling for TCP packet fragmentation

---

## Coding Conventions

### Python (ROS Nodes)
- **Style**: PEP 8 with 4-space indentation
- **Docstrings**: PEP 257 format
- **Naming**: `snake_case` for files/modules, `CamelCase` for classes
- **Node naming**: Descriptive names like `CentralControllerSSVEPNode2`

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
- **Central Controller**: v1 → v2 (dual-mode) → v3 (online fusion with `CircularEEGBuffer`)
- **Communication**: UDP → TCP for EEG data (reliability improvement)

### Key Design Principles
1. **Separation of Concerns**: Controllers handle state machines and timing; separate nodes handle data acquisition
2. **Hybrid Communication**: ROS2 for structured data, UDP for time-critical triggers
3. **Modular Pipelines**: Signal processing algorithms encapsulated in reusable classes

---

## Related Documentation
- `Architecture.md` - Detailed system architecture (Chinese)
- `AGENTS.md` - Repository guidelines and conventions
- `FunctionLog.md` - Function and class reference (Chinese)
- `dev_logs/` - Daily development notes (Chinese)