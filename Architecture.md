# 项目架构文档 (Project Architecture)

## 1. 系统概述
本项目是一个基于 ROS2 (Robot Operating System 2) 和 Unity 构建的分布式脑机接口 (BCI) 系统。系统的主要目标是进行脑电(EEG)信号的采集、处理（如 SSVEP 和 P300 范式），并实现与 Unity 前端（可能用于虚拟现实或视觉刺激呈现）的低延迟交互。

## 2. 核心架构设计
系统采用模块化、分布式的架构，主要分为以下几个子系统：

### 2.1 实验控制与状态机 (Central Controllers)
- **位置**: `src/eeg_processing/`
- **功能**: 管理实验范式的生命周期（如 Trial 的开始、刺激呈现、结束）、数据日志记录，以及与 Unity 端的状态同步。
- **演进**: 系统中存在多个版本的中心节点 (`CentralControllerNode`, `CentralControllerSSVEPNode`, `CentralControllerSSVEPNode2`, `CentralControllerSSVEPNode3`, `CentralControllerSSVEPTrainNode`)，表明该系统经历了多轮迭代，支持不同复杂度或类型的 SSVEP/P300 实验范式（如在线解码、离线训练等）。

### 2.2 数据采集与硬件抽象 (Data Acquisition)
- **位置**: `src/publisher_test/`
- **功能**: 作为脑电设备的硬件抽象层。核心节点 `eeg_tcp_listener_node.py` 通过 TCP 协议监听并接收外部脑电放大器发送的原始数据流，将其转换为 ROS2 标准的 `Float32MultiArray` 消息，发布到 ROS2 网络中供处理节点使用。

### 2.3 信号处理与解码流水线 (Processing Pipeline)
- **位置**: `src/eeg_processing/`
- **功能**: 负责对接收到的脑电信号进行实时或离线的去噪、特征提取和模式识别。
- **算法支持**: 
  - SSVEP (稳态视觉诱发电位): 支持基于 eTRCA (Ensemble Task-Related Component Analysis) 和 FBCCA (Filter Bank Canonical Correlation Analysis) 的流水线 (`ssvep_pipeline.py`, `ssvep_processing_fbcca.py`)。
  - P300 (事件相关电位): 包含在线和离线处理测试脚本 (`p300_processing_test.py`, `p300_processing_test_online.py`)，带有伪迹去除（如眼电 EOG 回归）和动态缓冲机制。

### 2.4 跨平台通信 (Inter-Process Communication)
系统采用了**混合通信模型**以平衡复杂数据传输与低延迟控制需求：
- **ROS2 Topics**: 用于传输结构化、体积较大或非时间关键的数据（如脑电信号流、图像分割数据 `/image_seg`）。Unity 端通过 `ROS-TCP-Endpoint` 接入 ROS2 网络。
- **UDP 通信**: 针对具有极高时间敏感性的同步信号（Triggers）和控制指令，系统额外使用了 UDP 协议（常用端口如 9999, 10000, 12001），绕过 ROS2 的中间件开销，确保脑电打标(Marking)与视觉刺激的精确对齐（详见 `history_sender.py` 及 C# 侧的 `HistoryManager.cs`）。

## 3. 主要包结构说明
- `eeg_processing`: 包含核心的业务逻辑，涵盖了实验流控制(Central Controller)、机器学习流水线(Pipelines)和部分与 Unity 交互的辅助工具。
- `publisher_test`: 包含用于数据注入和测试的节点。除了 EEG 数据的 TCP 接收客户端外，还包含图像发布器 (`image_publisher.py`, `seg_image_publisher.py`) 和 UDP 发送器 (`udp_sender_node.py`)，通常用于模拟外部设备的输入。
- `ROS-TCP-Endpoint`: 第三方/官方维护的桥接包，使 Unity 能够作为 ROS2 节点发送和接收消息。