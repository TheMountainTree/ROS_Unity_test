# 项目架构文档 (Project Architecture)

## 1. 系统概述
本项目是一个基于 ROS2 (Robot Operating System 2) 和 Unity 构建的分布式脑机接口 (BCI) 系统。系统的主要目标是进行脑电(EEG)信号的采集、处理SSVEP范式，并实现与 Unity 前端（用于视觉刺激呈现）的交互。

## 2. 核心架构设计
系统采用模块化、分布式的架构，主要分为以下几个子系统：

### 2.1 实验控制与状态机 (Central Controllers)
- **位置**: `src/eeg_processing/`
- **功能**: 管理实验范式的生命周期（如 Trial 的开始、刺激呈现、结束）、数据日志记录，以及与 Unity 端的状态同步。主要以`CentralControllerSSVEPNodeX.py`为主线，其他节点为辅助和测试节点。默认启动为decode模式，也可以使用参数声明为pretrain模式。该节点为核心节点，控制unity端闪烁和停止。
- **演进**: 系统中存在多个版本的中心节点 (`CentralControllerNode`, `CentralControllerSSVEPNode`, `CentralControllerSSVEPNode2`, `CentralControllerSSVEPNode3`, `CentralControllerSSVEPNode4`, `CentralControllerSSVEPTrainNode`)，表明该系统经历了多轮迭代，支持不同复杂度或类型的 SSVEP/P300 实验范式（如在线解码、离线训练等）。主线是`CentralControllerSSVEPNodeX.py`
- **decode**:通过发布话题 `/image_seg`（注意 ROS/OpenCV 的二维图像坐标系原点在左上角、Y 轴向下，而 Unity 原点在左下角、Y 轴向上，需要在发布前做垂直翻转），发布 6 张实际显示的图片。当 6 张图片接收完成后，Unity 类似`SSVEP_Stimulus2.cs`的脚本设置`isBatchCompleted = true`。
接着 ROS 发布命令 `cmd=decode_prepare` 到 `/image_seg`，Unity 接收后准备显示界面。随后 ROS 发布 `cmd=decode_stim` 到 `/image_seg`，Unity 检查 `isBatchCompleted == true` 后开始 SSVEP 闪烁，同时通过 UDP 发送 t`rial_started={trial_id}` 到 ROS 端口 10000 确认 trial 开始。
在 `CentralControllerSSVEPNode4.py` 中，decode 模式还会复用 pretrain 的 EEG 采集链路：进入 `decode_stimulating` 时 ROS 通过 UDP 向 Windows COM 转发器发送 `trigger 1`，停止刺激时发送 `trigger 2`。Windows 侧 Neuracle TCP 转发流中的 trigger 通道用于闭合当前 decode epoch，最终除了图片槽位映射关系（mapping CSV）和 trial 元数据（trials CSV）外，还会额外保存 decode EEG 的 trial CSV、metadata CSV 和 `ssvep4_decode_dataset_*.npy`。
- **pretrain**:Pretrain 模式用于采集 SSVEP 训练数据。首先根据 `num_targets`（目标数量，默认 8）和 `pretrain_repetitions_per_target`（每个目标重复次数，默认 3）生成试次计划并打乱顺序。
每个 `trial` 开始时，ROS 通过话题 `/ssvep_train_cmd` 发布命令 `cmd=cue`，Unity 接收后显示提示界面（高亮当前目标，告诉受试者注视哪个位置），持续 `pretrain_cue_duration_s` 秒（默认 2.0s）。
随后 ROS 发布 `cmd=stim` 到 `/ssvep_train_cmd`，Unity 类似`SSVEP_Stimulus2.cs`的脚本开始 SSVEP 闪烁刺激。同时 ROS 通过 UDP 发送 `trigger 1` 到 `192.168.56.3:8888`（Windows COM 转发器），用于 EEG 打标标记刺激开始。Unity 通过 UDP 发送 `trial_start={trial_id};target={target_id}` 到 ROS 端口 `10001` 确认。刺激持续 `pretrain_stim_duration_s` 秒（默认 1.5s）。
刺激结束后，ROS 发布 `cmd=rest` 到 `/ssvep_train_cmd`，Unity 显示休息界面。同时 ROS 通过 UDP 发送 `trigger 2` 到 `192.168.56.3:8888`，标记刺激结束。ROS 从 EEG 环形缓冲区（CircularEEGBuffer）提取 `trigger=1` 到 `trigger=2` 之间的 EEG 数据作为 epoch，保存到 `dataset_x`，标签保存到 `dataset_y`。休息持续 `pretrain_rest_duration_s` 秒（默认 1.0s）。
所有 trial 结束后，ROS 将 EEG 数据集保存为 `.npy` 文件（包含 x 和 y 数组），并发布 `cmd=done` 到 `/ssvep_train_cmd` 通知 Unity 实验结束。

### 2.1.1 Node4 统一通信配置区
`CentralControllerSSVEPNode4.py` 在参数层面把网络链路统一整理为一组共享配置，decode 与 pretrain 共用：
- Unity decode 回执 UDP：`decode_start_bind_ip` / `decode_start_port`
- Unity pretrain 回执 UDP：`train_trigger_bind_ip` / `train_trigger_bind_port`
- Windows trigger 转发：`trigger_local_ip` / `trigger_local_port` -> `trigger_remote_ip` / `trigger_remote_port`
- Windows EEG TCP：`eeg_server_ip` / `eeg_server_port`
- EEG 数据帧：`eeg_recv_buffer_size` / `eeg_n_channels` / `eeg_frame_floats` / `eeg_fs`

在代码内部，这些配置由统一加载函数管理，decode / pretrain 只读取各自的时序参数，不再重复解析同一套 IP 和端口。

### 2.2 数据采集与硬件抽象 (Data Acquisition)
- **位置**: `src/publisher_test/`
- **功能**: 作为脑电设备的硬件抽象层。测试节点 `eeg_tcp_listener_node.py` 通过 TCP 协议监听并接收外部脑电放大器发送的原始数据流，来自windows端的软件`Neuracle`。为了能够和ubuntu端的ros结合（为了在unity上显示），需要通过脚本实现数据传输和信号转发。因此需要ros节点先通过类似`udp_sender_node.py`的脚本，通过udp协议把trigger信号从本地的192.168.56.103：5006发送到远端的192.168.56.3:8888。而远端则会运行`windows_com_publisher.py`的程序，持续监听192.168.53.3:8888的信号，然后按照`Neuracle`的需要，把trigger value和command的帧头拼装，以115200的波特率发送给`COM7`，注意`COM7`需要基于TriggerBox的USB具体注册了哪个虚拟口来决定。在`Neuracle`中打开数据转发后，会按照`Ch1(4byte)->Ch2(4byte)->...->ChN(4byte)->TriggerValue(4byte)->Next Ch1(4byte)`的形式发送在远端的`8712`端口上，通过tcp监听连接192.168.56.3:8712实现在ros端接受eeg数据。

### 2.3 信号处理与解码流水线 (Processing Pipeline)
- **位置**: `src/eeg_processing/`
- **功能**: 负责对接收到的脑电信号进行实时或离线的去噪、特征提取和模式识别。
- **算法支持**: 
  - SSVEP (稳态视觉诱发电位): 支持基于 eTRCA (Ensemble Task-Related Component Analysis) 和 FBCCA (Filter Bank Canonical Correlation Analysis) 的流水线 (`ssvep_pipeline.py`, `ssvep_processing_fbcca.py`)。主要使用ssvep的eTRCA，而FBCCA仅仅作为对比。
  - P300 (事件相关电位): 包含在线和离线处理测试脚本 (`p300_processing_test.py`, `p300_processing_test_online.py`)，带有伪迹去除（如眼电 EOG 回归）和动态缓冲机制。仅仅作为测试程序。

### 2.4 跨平台通信 (Inter-Process Communication)
系统采用了**混合通信模型**以平衡复杂数据传输与低延迟控制需求：
- **ROS2 Topics**: 用于传输结构化、体积较大或非时间关键的数据（如脑电信号流、图像分割数据 `/image_seg`）。Unity 端通过 `ROS-TCP-Endpoint` 接入 ROS2 网络。
- **UDP 通信**: 针对具有极高时间敏感性的同步信号（Triggers）和控制指令，系统额外使用了 UDP 协议（常用端口如 9999, 10000, 12001），绕过 ROS2 的中间件开销，确保脑电打标(Marking)与视觉刺激的精确对齐（详见 `history_sender.py` 及 C# 侧的 `HistoryManager.cs`）。
- **EEG TCP + Trigger 内嵌通道**: 当前 SSVEP 主线节点 (`Node3` pretrain, `Node4` decode/pretrain) 直接连接 Windows `8712` EEG TCP 服务，并以每采样点帧的最后一个 float32 作为 trigger 通道，从而按 `trigger=1 -> trigger=2` 精确切 epoch。Windows 端发送格式为 `Ch1(4B) -> Ch2(4B) -> ... -> ChN(4B) -> Trigger(4B) -> Next Ch1...`。
- **待测试和合并的脚本**:`history_sender_node` 是一个用于向 Unity 发送历史图片队列的辅助节点，主要用于在 Unity 界面上显示已选择/已操作的图片历史记录，并支持通过命令删除历史记录中的图片。
节点启动后，通过定时器以 1 秒为间隔，从本地图片目录（默认 `~/workspace/eeg_robot/src/robot_ctr/graph/graph/results/segmentation_20260206_223629`）读取图片，经过垂直翻转（适配 Unity 坐标系）和统一尺寸调整（默认 100x100）后，发布到话题 `/history_image`。每张图片的 `frame_id` 携带唯一标识 `hist_id={image_id}`。最多发送 10 张图片后自动停止定时器。
同时，节点订阅话题 `/history_control`，接收删除命令，并通过 UDP 发送 JSON 格式的控制指令到 `127.0.0.1:12001（Unity 端）`。支持的命令包括：

`delete_last`：删除最后一张图片
`clear 或 delete_all`：清空所有历史记录
`delete_id:<int>`：删除指定 ID 的图片

节点在发送完 10 张图片后保持在线，可继续通过 `/history_control` 话题响应删除命令。

## 3. 主要包结构说明
- `eeg_processing`: 包含核心的业务逻辑，涵盖了实验流控制(Central Controller)、机器学习流水线(Pipelines)和部分与 Unity 交互的辅助工具。
- `publisher_test`: 包含用于数据注入和测试的节点。除了 EEG 数据的 TCP 接收客户端外，还包含图像发布器 (`image_publisher.py`, `seg_image_publisher.py`) 和 UDP 发送器 (`udp_sender_node.py`)，通常用于模拟外部设备的输入。
- `ROS-TCP-Endpoint`: 第三方/官方维护的桥接包，使 Unity 能够作为 ROS2 节点发送和接收消息。
