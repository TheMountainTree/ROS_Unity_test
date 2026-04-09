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
   - 核心节点主线为 `CentralControllerSSVEPNodeX.py`（当前维护到 V4 版本）。
   - 核心节点主线为 `CentralControllerSSVEPNodeX.py`（当前维护到 V4 版本）。
   - 负责管理实验生命周期（Trial 控制）、范式状态同步、数据记录，主要支持 `decode` (在线解码验证) 和 `pretrain` (离线模型训练数据采集) 两种模式。
2. **数据采集与硬件抽象 (Data Acquisition)** (`src/publisher_test`):
   - `eeg_tcp_listener_node.py` 作为核心硬件接口，通过 TCP 监听脑电放大器（如 Neuracle 软件转发）的原始数据流。节点内部处理了 TCP 拆包/粘包逻辑，并将数据整合为 ROS2 的 `Float32MultiArray` 消息发布。
3. **信号处理与解码流水线 (Processing Pipeline)** (`src/eeg_processing`):
   - **SSVEP**: 提供基于 eTRCA (Ensemble Task-Related Component Analysis) 算法的核心流水线 (`ssvep_pipeline.py`)，并包含 FBCCA 作为零需训练比对基准。
   - **P300**: 包含基于 EOG 回归伪迹去除机制的离线处理与在线缓冲流提取系统 (`p300_processing_test_online.py`)。
4. **跨平台通信模型 (IPC)**:
- **ROS2 Topics**: 用于传输常规控制指令及较大数据体（如通过 `/image_seg` 发送给 Unity 的结构化图像组）。依赖官方包 `ROS-TCP-Endpoint`。
- **SSVEP decode 双话题模型**: 当前主链路中，decode 图片走 `/image_seg`，decode 控制命令单独走 `/ssvep_decode_cmd`，与 pretrain 控制话题 `/ssvep_train_cmd` 分离。
   - **UDP Socket**: 为确保脑电打标与视觉刺激极高的时间对齐要求，Trigger 与精准 Timing 确认使用旁路 UDP 协议。
5. **Reasoner 分组交互测试 (Reasoner Decode Loop)**:
   - 新增 `publisher_test/reasoner_publish_test.py`，可从 `~/Pictures/截图` 中按自然排序取前 24 张图片，切成 4 组，每组 6 张。
   - `SSVEP_Communication_Node2.py` 可在 decode 模式下启用 reasoner 外部图片组驱动 `/image_seg`，并支持：
     - 双节点握手：`cmd=ssvep_ready` / `cmd=reasoner_ready`
     - 参数模拟选择：`mock_selected_index`
     - 普通选择写入 `/history_image`
     - 勾/叉特殊动作：历史回传与回退删除

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

# 4. 启动 SSVEP Node4（decode/pretrain 均支持 EEG TCP + trigger + npy 保存）
ros2 run eeg_processing central_controller_ssvep_node4 --ros-args -p run_mode:=decode

# 5. 启动 reasoner 分组测试节点（按 24 张图分 4 组）
ros2 run publisher_test reasoner_publish_test

# 6. 启动 reasoner 版 decode 通信节点（Node2）
ros2 run eeg_processing ssvep_communication_node2 --ros-args \
  -p run_mode:=decode \
  -p reasoner_mode_enabled:=true

# 7. 启动模块化通信节点（Node3）
ros2 run eeg_processing ssvep_communication_node3 --ros-args \
  -p run_mode:=decode \
  -p reasoner_mode_enabled:=true

# 8. 启动EEG Bypass 版本节点 （Node3_1）
ros2 run eeg_processing ssvep_communication_node3_1 --ros-args \
-p eeg_bypass_debug:=true \
-p reasoner_mode_enabled:=true

# 8.1 启动reasoner 推理器节点
ros2 run publisher_test reasoner_publish_test_2

# 9. Node4_test: 集成真实 eTRCA 解码的完整流程
#    包含: pretrain 数据采集 + 自动训练模型, decode 真实 EEG 解码, reasoner 多阶段交互

# 9.1 Pretrain 模式 - 采集 EEG 数据并自动训练 eTRCA 模型
#     注意：需要连接真实 EEG 设备 (TCP: 192.168.56.3:8712)
ros2 run eeg_processing ssvep_communication_node4_test --ros-args \
  -p run_mode:=pretrain

# 9.2 Decode 模式 - 使用预训练模型进行真实 EEG 解码
ros2 run eeg_processing ssvep_communication_node4_test --ros-args \
  -p run_mode:=decode \
  -p reasoner_mode_enabled:=true

# 9.3 Reasoner 多阶段推理节点 (object -> category -> activity)
#     默认读取 ./picture 目录下的图片
ros2 run publisher_test reasoner_publish_test_3_test

# 调试模式：跳过 EEG TCP 连接和 trigger 发送 (无真实设备时使用)
# ros2 run eeg_processing ssvep_communication_node4_test --ros-args \
#   -p run_mode:=pretrain \
#   -p eeg_bypass_debug:=true
```

### 3. 测试与工具工具
```bash
# UDP 触发器发送模拟 (用于打标对齐测试)
ros2 run publisher_test udp_sender_node --ros-args -p trigger_value:=1 -p remote_ip:=192.168.56.3

# 历史图像推流工具（向 Unity 异步发送操作记录）
ros2 run eeg_processing history_sender_node
```

### 4. Reasoner 测试模式说明

`reasoner_publish_test.py` 与 `SSVEP_Communication_Node2.py` 配合时，运行语义如下：

- 输入图片来源：`~/Pictures/截图` 前 24 张，按自然排序切为 4 组
- 屏幕槽位布局：两行四列，第一行 `0 1 2 3`，第二行 `4 5 6 7`
- 特殊槽位：`3` 为勾，`7` 为叉
- 动态图片槽位：`0 1 2 4 5 6`
- 握手协议：
  - communication -> reasoner: `cmd=ssvep_ready`
  - reasoner -> communication: `cmd=reasoner_ready`
- decode 控制话题：
  - 图片：`/image_seg`
  - 命令：`/ssvep_decode_cmd`
  - 命令集合：`prepare / stim / stop / done`
  - `stop / done` 语义：停止闪烁，但保留当前 6 张图片显示，直到下一批图片覆盖
- 选择参数：
  - `mock_selected_index=0/1/2/4/5/6`：将当前屏幕对应槽位图片推入 history，并请求下一组
  - `mock_selected_index=3`：把所有 history 图片回传给 reasoner，当前 6 张保持不变
  - `mock_selected_index=7`：删除上一步操作（若上一步是 `3(confirm)` 则执行 `rollback`；若上一步是选项选择则取消该选择并回到对应页面）

示例：
```bash
ros2 param set /central_controller_ssvep_node3_1 mock_selected_index 0
```

### 5. Node2 配置说明

`SSVEP_Communication_Node2.py` 当前采用“静态配置 + 少量 ROS 覆盖”的方式：

- 静态默认值：编辑 `src/eeg_processing/eeg_processing/ssvep_communication_node2_config.py`
- 运行时可覆盖参数：
  - `run_mode`
  - `reasoner_mode_enabled`
  - `mock_selected_index`
  - `save_dir`
  - `image_dir`
  - `decode_max_trials`

例如：

```bash
ros2 run eeg_processing ssvep_communication_node2 --ros-args \
  -p run_mode:=pretrain \
  -p save_dir:=data/central_controller_ssvep3 \
  -p decode_max_trials:=1
```

### 6. Node3 模块化配置说明

`SSVEP_Communication_Node3.py` 在保持 Node2 行为一致的前提下，按模块拆分为：

- `decode.py`：decode 状态机与图像发布逻辑
- `pretrain.py`：pretrain 状态机与共享 EEG/epoch 采集逻辑
- `reasoner.py`：reasoner 握手、分组、selection/rollback 逻辑
- `SSVEP_Communication_Node3.py`：主节点初始化、调度、清理

Node3 同样采用”静态配置 + 少量 ROS 覆盖”：

- 静态默认值：编辑 `src/eeg_processing/eeg_processing/ssvep_communication_node3_config.py`
- 运行时可覆盖参数：
  - `run_mode`
  - `reasoner_mode_enabled`
  - `mock_selected_index`
  - `save_dir`
  - `image_dir`
  - `decode_max_trials`

例如：

```bash
ros2 run eeg_processing ssvep_communication_node3 --ros-args \
  -p run_mode:=decode \
  -p reasoner_mode_enabled:=true
```

### 7. Node4_test: 真实 eTRCA 解码完整流程

`SSVEP_Communication_Node4_test.py` 是集成真实 EEG 解码的完整系统，支持：

- **Pretrain 模式**：采集训练数据 → 自动训练 eTRCA 模型 → 保存权重
- **Decode 模式**：加载预训练模型 → 真实 EEG 解码 → 返回选择结果
- **Reasoner 集成**：与推理器节点配合实现多阶段交互

#### 7.1 Pretrain 模式（数据采集 + 自动训练）

**流程：**
1. 发送 `cue` 命令 → Unity 显示目标提示
2. 发送 `stim` 命令 + trigger 1 → Unity 开始闪烁，EEG 开始记录
3. 发送 `rest` 命令 + trigger 2 → 停止闪烁，提取 EEG epoch
4. 重复 24 trials (8 targets × 3 reps)
5. 自动重采样 (1000Hz → 256Hz) 并训练 eTRCA 模型
6. 保存模型到 `data/ssvep_etrca_model.pkl`

**启动命令：**
```bash
# 终端1: 启动 Unity 通信桥接
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# 终端2: 启动主节点 (pretrain 模式，需要连接真实 EEG 设备)
ros2 run eeg_processing ssvep_communication_node4_test --ros-args \
  -p run_mode:=pretrain

# 调试模式 (无真实 EEG 设备时使用)
# ros2 run eeg_processing ssvep_communication_node4_test --ros-args \
#   -p run_mode:=pretrain \
#   -p eeg_bypass_debug:=true
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `run_mode` | `decode` | 运行模式：`pretrain` 或 `decode` |
| `eeg_bypass_debug` | `false` | 调试模式，跳过 EEG TCP 和 trigger (无真实设备时使用) |
| `save_dir` | `data/central_controller_ssvep_node4_test` | 数据保存目录 |
| `etrca_model_path` | `data/ssvep_etrca_model.pkl` | 模型保存路径 |

**输出文件：**
- `ssvep4_pretrain_dataset_*.npy` - EEG epochs 数据集
- `ssvep4_pretrain_trials_*.csv` - Trial 记录
- `ssvep4_pretrain_metadata_*.csv` - Epoch 元数据
- `ssvep_etrca_model.pkl` - 训练好的 eTRCA 模型

#### 7.2 Decode + Reasoner 模式（真实 EEG 解码交互）

**完整流程：**
```
┌─────────────────┐                    ┌─────────────────┐              ┌────────┐
│ reasoner_       │  1. reasoner_ready │ Node4_test      │              │ Unity  │
│ publish_test_   │ ─────────────────> │ (主节点)         │              │        │
│ 3_test.py       │                    │                 │              │        │
│ (推理器节点)     │  2. 图片批次        │                 │  3. 显示图片  │        │
│                 │ ─────────────────> │                 │ ───────────> │        │
│                 │                    │                 │              │        │
│                 │                    │  4. EEG 采集    │  5. 刺激闪烁  │        │
│                 │                    │ <───────────────────────────────│        │
│                 │                    │                 │              │        │
│                 │                    │  6. eTRCA 解码   │              │        │
│                 │                    │ (加载预训练权重)  │              │        │
│                 │                    │                 │              │        │
│                 │  7. selection 结果  │                 │              │        │
│                 │ <─────────────────│                 │              │        │
│                 │                    │                 │              │        │
│                 │  8. 下一批图片      │                 │              │        │
│                 │ ─────────────────>│                 │              │        │
└─────────────────┘                    └─────────────────┘              └────────┘
```

**启动命令：**
```bash
# 终端1: Unity 通信桥接
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# 终端2: 主节点 (decode + reasoner 模式，需要真实 EEG 设备和预训练模型)
# 注意：必须先运行 pretrain 模式生成模型文件 data/ssvep_etrca_model.pkl
ros2 run eeg_processing ssvep_communication_node4_test --ros-args \
  -p run_mode:=decode \
  -p reasoner_mode_enabled:=true

# 终端3: 推理器节点 (多阶段: object -> category -> activity)
# 默认读取 ./picture 目录下的图片
ros2 run publisher_test reasoner_publish_test_3_test
```

#### 7.3 Reasoner 多阶段交互说明

`reasoner_publish_test_3_test.py` 实现三阶段推理流程：

| 阶段 | Stage | 说明 |
|------|-------|------|
| StateA | `object` | 物体选择，可多选，支持翻页 |
| StateB | `category` | 类别选择，单选 |
| StateC | `activity` | 活动选择，LLM 生成候选 |

**槽位布局：**
```
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │   Row 0: 图片槽位 0,1,2 + 确认(✓)
├─────┼─────┼─────┼─────┤
│  4  │  5  │  6  │  7  │   Row 1: 图片槽位 4,5,6 + 撤销(✗)
└─────┴─────┴─────┴─────┘
```

**选择动作：**
- **Slot 0,1,2,4,5,6**: 选择图片，记录到 history
- **Slot 3 (确认)**: 翻页或进入下一阶段
- **Slot 7 (撤销)**: 撤销上一步操作

#### 7.4 eTRCA 解码流程

```python
# 1. EEG epoch 捕获 (trigger 1 -> trigger 2)
epoch = dataset_x[-1]  # shape: (n_channels, n_samples)

# 2. 重采样 1000Hz -> 256Hz
epoch = signal.resample(epoch, n_samples, axis=1)

# 3. eTRCA 解码
predicted_label = decoder.decode(epoch)  # 返回 1-8

# 4. 映射到槽位
predicted_freq = ssvep_frequencies[predicted_label - 1]  # 8.0, 10.0, ..., 45.0 Hz
slot_index = find_slot_with_frequency(predicted_freq)
```

**频率映射表：**
| Label | 频率 (Hz) |
|-------|-----------|
| 1 | 8.0 |
| 2 | 10.0 |
| 3 | 12.0 |
| 4 | 15.0 |
| 5 | 20.0 |
| 6 | 30.0 |
| 7 | 40.0 |
| 8 | 45.0 |

#### 7.5 模块文件

| 文件 | 功能 |
|------|------|
| `SSVEP_Communication_Node4_test.py` | 主节点，组合各模块 |
| `decode_2_test.py` | Decode 模块，EEG 解码逻辑 |
| `pretrain_2_test.py` | Pretrain 模块，自动训练逻辑 |
| `reasoner_2_test.py` | Reasoner 模块，选择处理逻辑 |
| `ssvep_communication_node4_test_config.py` | 静态配置 |
| `ssvep_pipeline.py` | eTRCA 算法实现 (SSVEPPretrainer, SSVEPDecoder) |

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
2. **Epoch 精准提取**: SSVEP V3 预训练节点废弃了”基于预设时间硬截断”的机制，重构为直接侦听 TCP 数据流内的 trigger 通道 (`trigger=1` 开始, `trigger=2` 结束)，完美解决了变长 Epoch 的网络时延导致的抖动问题。
3. **Decode EEG 采集闭环**: `CentralControllerSSVEPNode4.py` 将 decode 模式升级为可发送 trigger、接收 EEG TCP 数据、保存 decode `.npy` 数据集，并写出 decode EEG trial/meta CSV。
4. **Node2 静态配置抽离**: `SSVEP_Communication_Node2.py` 现已把大部分 decode / pretrain / reasoner / 网络默认配置移到 `ssvep_communication_node2_config.py`，只保留少数联调参数继续通过 ROS 覆盖。
5. **验证工具补全**: 除 `validate_ssvep3_npy.py` 外，新增 `validate_ssvep4_npy.py`，用于绘制 decode 阶段的 EEG epoch 图（默认单 epoch）。
6. **Reasoner 闭环测试**: 新增 24 图/4 组 reasoner 测试模式，支持握手、history 回传、撤销回退，以及 history 缩略图固定尺寸（默认 `100x100`）。
7. **Decode 控制拆分与停闪保留画面**: 当前主链路把 decode 控制命令拆到 `/ssvep_decode_cmd`，并将 `stop/done` 改为”停止闪烁但保留当前图片，直到下一批覆盖”。
8. **Node3 模块化重构**: 新增 `SSVEP_Communication_Node3.py` 与 `decode.py / pretrain.py / reasoner.py`，主节点仅保留初始化与调度，配置默认值集中到 `ssvep_communication_node3_config.py`。
9. **Node4_test 真实 EEG 解码集成**: 新增 `SSVEP_Communication_Node4_test.py`，实现完整闭环：
   - Pretrain 模式：采集数据 → 自动训练 eTRCA 模型 → 保存权重文件
   - Decode 模式：加载预训练模型 → 真实 EEG 解码 → 返回选择结果
   - Reasoner 集成：与 `reasoner_publish_test_3_test.py` 配合实现多阶段交互 (object → category → activity)
   - 采样率自动转换：EEG 数据 1000Hz → 256Hz 重采样后送入 eTRCA 算法
10. **LLM 流式输出转发**: 新增 `/reasoner/llm_stream` → `/llm_output_stream` 话题转发，支持 Unity 前端实时显示 LLM 生成的活动候选。

## Node2 静态配置速览
`SSVEP_Communication_Node2.py` 当前按设备而不是按模式组织默认网络配置：

- Unity 通信：`host_ip` + `decode_start_port` / `pretrain_start_port`
- Windows trigger 转发：`local_ip` / `local_port` -> `remote_ip` / `remote_port`
- Windows EEG TCP：`server_ip` / `server_port`
- EEG 数据帧：`recv_buffer_size` / `n_channels` / `frame_floats` / `fs`

默认主链路在 `ssvep_communication_node2_config.py` 中定义：
- Ubuntu trigger 发送：`192.168.56.103:5006`
- Windows COM 转发：`192.168.56.3:8888`
- Windows EEG TCP：`192.168.56.3:8712`
- Unity decode 回执：`0.0.0.0:10000`
- Unity pretrain 回执：`0.0.0.0:10001`

## 数据验证与绘图
```bash
# 验证 pretrain 数据集
python3 src/eeg_processing/eeg_processing/validate_ssvep3_npy.py

# 验证并绘制 decode 数据集（默认只画 1 个 epoch）
python3 src/eeg_processing/eeg_processing/validate_ssvep4_npy.py
```

## 协作与开发规范 (Contributing)
- **代码规范**: Python 代码请严格遵循 PEP 8（推荐 4 空格缩进）并提供完整的 PEP 257 Docstring，支持使用 `ament_flake8` 和 `ament_pep257` 进行 Lint 检测。
- **提交规范 (Commit)**: 建议一功能一提交，Summary 需使用明确的动词/祈使句（中英皆可，例如 `eeg_processing: 重构 epoch 边界检测逻辑`）。
- **测试框架**: 使用 `pytest`，测试代码存放在各 Package 下的 `test/` 文件夹中。提 PR 前请确保 `colcon test` 无报错。
