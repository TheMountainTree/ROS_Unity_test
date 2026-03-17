# ROS2 Unity BCI System (脑机接口系统)

## 项目简介 (Project Overview)
本项目是一个基于 **ROS2 (Robot Operating System 2)** 和 **Unity** 构建的分布式脑机接口 (BCI) 系统。系统的主要目标是进行脑电 (EEG) 信号的采集、实时与离线处理（支持 SSVEP 和 P300 范式），并实现与 Unity 前端（用于视觉刺激呈现和交互）的低延迟跨平台通信。

## 核心架构设计 (Architecture)

本系统采用模块化、分布式的混合通信架构，保障复杂数据吞吐与控制指令的极低延迟：

```text
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
             │ ROS-TCP-Endpoint (Port 10000)
             ▼
┌─────────────────────┐     UDP (Triggers)
│   Unity Frontend    │ ◀────────────────▶ ROS Nodes
│  (Visual Stimulus)  │    Port 9999/10000/10001/12001
└─────────────────────┘
```

### 主要子系统：
1. **实验控制与状态机 (Central Controllers)** (`src/eeg_processing`):
   - 核心节点主线为 `CentralControllerSSVEPNodeX.py`（最新迭代至 V3 版本）。
   - 负责管理实验生命周期（Trial 控制）、范式状态同步、数据记录，主要支持 `decode` (在线解码验证) 和 `pretrain` (离线模型训练数据采集) 两种模式。
2. **数据采集与硬件抽象 (Data Acquisition)** (`src/publisher_test`):
   - `eeg_tcp_listener_node.py` 作为核心硬件接口，通过 TCP 监听脑电放大器（如 Neuracle 软件转发）的原始数据流。节点内部处理了 TCP 拆包/粘包逻辑，并将数据整合为 ROS2 的 `Float32MultiArray` 消息发布。
3. **信号处理与解码流水线 (Processing Pipeline)** (`src/eeg_processing`):
   - **SSVEP**: 提供基于 eTRCA (Ensemble Task-Related Component Analysis) 算法的核心流水线 (`ssvep_pipeline.py`)，并包含 FBCCA 作为零需训练比对基准。
   - **P300**: 包含基于 EOG 回归伪迹去除机制的离线处理与在线缓冲流提取系统 (`p300_processing_test_online.py`)。
4. **跨平台通信模型 (IPC)**:
   - **ROS2 Topics**: 用于传输常规控制指令及较大数据体（如通过 `/image_seg` 发送给 Unity 的结构化图像组）。依赖官方包 `ROS-TCP-Endpoint`。
   - **UDP Socket**: 为确保脑电打标与视觉刺激极高的时间对齐要求，Trigger 与精准 Timing 确认使用旁路 UDP 协议。

## 环境与依赖 (Dependencies)
- **系统要求**: Ubuntu (运行 ROS2 核心) / Windows (运行 Unity 与脑电设备接口软件)
- **核心框架**: ROS2 (Humble / Iron), Unity
- **开发语言**: Python 3.8+, C# (Unity 脚本存放于 `eeg_processing/` 备查)
- **主要 Python 依赖**: `numpy`, `scipy`, `mne` (可选), `opencv-python`, `Pillow`, `brainda` (部分 SSVEP 算法库)。

## 编译与运行 (Build & Run)

### 1. 编译工作空间
在工作空间根目录下执行：
```bash
# 推荐使用 symlink 便于开发时热更新 Python 代码
colcon build --symlink-install

# 若只需编译特定包：
colcon build --packages-select eeg_processing publisher_test ros_tcp_endpoint

# 刷新环境变量
source install/setup.bash
```

### 2. 运行核心模块
```bash
# 1. 启动 Unity 通信桥接
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# 2. 启动脑电数据 TCP 接收接口
ros2 run publisher_test eeg_tcp_listener_node

# 3. 启动 SSVEP 中心控制节点 (可配置 run_mode 为 pretrain 或 decode)
ros2 run eeg_processing central_controller_ssvep_node3 --ros-args -p run_mode:=decode
```

### 3. 测试与工具工具
```bash
# UDP 触发器发送模拟 (用于打标对齐测试)
ros2 run publisher_test udp_sender_node --ros-args -p trigger_value:=1 -p remote_ip:=192.168.56.3

# 历史图像推流工具（向 Unity 异步发送操作记录）
ros2 run eeg_processing history_sender_node
```

## 目录结构 (Directory Structure)
```text
ROS_Unity_test/
├── src/
│   ├── eeg_processing/         # 核心算法：实验状态机、机器学习流水线及 Unity C# 脚本参考
│   ├── publisher_test/         # 数据接口：TCP 脑电监听器，模拟图像/UDP发布器
│   └── ROS-TCP-Endpoint/       # Unity-ROS 通信桥 (第三方依赖)
├── data/                       # 生成数据：实验记录 (.csv)，EEG 数据集 (.npy) 和 验证图表
├── dev_logs/                   # 每日开发与变更日志
├── build/ / install/ / log/    # ROS2 colcon 编译产物
├── Architecture.md             # 架构设计详情 (AI 与开发者参阅)
├── FunctionLog.md              # 核心类、函数接口对照字典 (AI 与开发者参阅)
├── AGENTS.md / QWEN.md / CLAUDE.md # 辅助 AI (如 Cursor/Claude) 阅读的上下文设定文档
└── README.md                   # 本指南文件
```

## 系统演进与近期亮点 (Recent Evolutions)
系统经过高强度迭代（详见 `dev_logs/`）：
1. **TCP 替代 UDP 接收**: 由于脑电采样率要求高，已废弃 UDP 脑电数据监听，全面升级为具有可靠粘包和缓存处理能力的 `eeg_tcp_listener_node`，保障了底层数据稳定。
2. **Epoch 精准提取**: SSVEP V3 预训练节点废弃了“基于预设时间硬截断”的机制，重构为直接侦听 TCP 数据流内的 trigger 通道 (`trigger=1` 开始, `trigger=2` 结束)，完美解决了变长 Epoch 的网络时延导致的抖动问题。
3. **多协议深度融合**: 完全确立了以 ROS Topic 为控制骨架，以 UDP 为低时延神经肌肉“韧带”的混合架构设计，并通过 `validate_ssvep3_npy.py` 等工具链完善了从采集到算法验证的全套闭环。

## 协作与开发规范 (Contributing)
- **代码规范**: Python 代码请严格遵循 PEP 8（推荐 4 空格缩进）并提供完整的 PEP 257 Docstring，支持使用 `ament_flake8` 和 `ament_pep257` 进行 Lint 检测。
- **提交规范 (Commit)**: 建议一功能一提交，Summary 需使用明确的动词/祈使句（中英皆可，例如 `eeg_processing: 重构 epoch 边界检测逻辑`）。
- **测试框架**: 使用 `pytest`，测试代码存放在各 Package 下的 `test/` 文件夹中。提 PR 前请确保 `colcon test` 无报错。