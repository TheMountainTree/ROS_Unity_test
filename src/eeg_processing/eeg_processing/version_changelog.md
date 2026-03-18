# SSVEP 架构版本演进日志 (基于实际文件)

---

## 📅 文件更替关系

| 类别 | 旧版架构文件 | 修改/废弃/保留 | 新版架构文件 |
|---|---|---|---|
| **Unity视觉渲染** | `SSVEP_Stimulus.cs` (纯解码) | ➡ 吸收合并 | `SSVEP_Stimulus2.cs` (双模式统一) |
| **Unity视觉渲染** | `SSVEP_Train.cs` (纯训练) | ➡ 被吸收合并 | `SSVEP_Stimulus2.cs` |
| **ROS主控节点** | `CentralControllerSSVEPNode.py` (仅解码控制) | ➡ 升级统一 | `CentrlControllerSSVEPNode2.py` (双模式控制) |
| **ROS训练+采集节点** | - | ➡ 曾尝试的完整流 | `CentralControllerSSVEPTrainNode.py` (带EEG采集) |
| **算法核心类库** | （原系统算法零散） | ➡ 全新封装 | `ssvep_pipeline.py` (4个独立组件) |
| **全链路集成测试** | （原先为全流程单体脚本）| ➡ 大幅重构 | `ssvep_processing_test.py` (算法流水线测试) |

---

## 🛠️ 各模块具体改动点

### 1. Unity 端完全统一：SSVEP_Stimulus2.cs
旧版本将解码（发图闪烁）和训练（cue/stim/rest）强行分为两个独立的 MonoBehaviour 脚本 (`SSVEP_Stimulus.cs` 和 `SSVEP_Train.cs`)。
**`SSVEP_Stimulus2.cs` 做的改变：**
- **合二为一**：通过引入内部枚举 `VisualMode (None, Decode, Pretrain)`，单个脚本同时处理两类工作。
- **双重订阅与路由**：同时订阅了 `/image_seg` 和 `/ssvep_train_cmd`，并在解码消息中探测 `cmd=` 自动路由到训练逻辑。
- **优化解码闪烁**：旧版解码时，8个框全部统一闪烁。新版加入了对 `targetIndices` 的过滤，保证**只有实际使用的6个目标框会按对应频率闪烁**。
- **三路 UDP 通信**：独立管理解码 marker (9999端口)、解码 ACK (10000端口) 和 训练 marker (10001端口)，互不干扰，向ROS回传精准时标。

---

### 2. ROS 主控端大统一：CentrlControllerSSVEPNode2.py
旧版的 `CentralControllerSSVEPNode.py` 只有简单的发图片和计时逻辑（纯解码模式），而曾经存在过的 `CentralControllerSSVEPTrainNode.py` 内部耦合了基于环形缓冲区的纯 EEG 数据采集逻辑。
而现存的最终版本 **`CentrlControllerSSVEPNode2.py`** 实现了向控制面的终极统一：
- **双模式参数加载**：通过 ROS 配置 `run_mode = decode / pretrain` 决定启动哪个状态机。
- **Decode 模式**：保留了旧版的功能，随机生成 6 张图映射关系（支持占位色块自动生成以备不时之需），每次发布一套带频率参数的图像帧数据给 Unity 进行交互。
- **Pretrain 模式**：内置了试次随机计划表生成（Cue -> Stim -> Rest 三阶段状态机），向 Unity 发布带有时间配置的字符命令包控制闪烁流。
- **完全解耦底层数据**：该 Node2 版本**主动丢弃了**旧尝试中的 `CircularEEGBuffer` 和 `SSVEPTrialDataCollector` 代码。它专注于作为“中枢时序调度器”（写精确的 CSV 时序切片和 Marker），而将真正的 EEG Raw Data 采集过程独立留给专门的数据流节点去做（符合 ROS 节点高内聚低耦合的设计）。

---

### 3. 系统级算法剥离：ssvep_pipeline.py
将科研级随意穿插的数据处理脚本，转化为生产级可用库。
**四个开箱即用的模块被专门创造出来：**
- **`SSVEPDataLoader`**：处理 MNE 格式及 Nakanishi2015 BCI 开放数据集，包含 lazy-import 避免在没有 PyTorch 环境下的报错。
- **`SSVEPPretrainer`**：把 eTRCA/FBTRCA 及复杂的 FilterBank 构建逻辑黑盒化，对上层节点只暴露出极简的 `.fit(X, y)` 以及自带配置持久化的 `.save('model.pkl')` 接口。
- **`SSVEPDecoder`**：完全面向在线推理环境。通过 `decoder = SSVEPDecoder.from_file('model.pkl')` 即可瞬间加载，随时准备接收实时 epoch 的 `.decode(X)` 调用，零学习成本。
- **`SSVEPEvaluator`**：自带纯 NumPy 实现的 Stratified K-Fold 交叉验证算法（`fallback`），在脱离 torch 的边缘机器上照样能进行评估部署。

---

### 4. 测试逻辑瘦身：ssvep_processing_test.py
庞杂的长篇单体执行脚本完成了历史使命。
**现在的 `ssvep_processing_test.py` 转变为了集成测试断言与核心 Demo：**
- 一气呵成演示从 DataLoader 拿数据 $\rightarrow$ Pretrainer fit 并落盘为 `.pkl` 权重 $\rightarrow$ 重新实例化出 Decoder 进行反向推理的流程。
- 调用 Evaluator 运行纯净的 6-Fold 交叉验证看极限精度。
- **安全性防御测试**：专门增加了一个比对 direct-fit 模型和从物理磁盘重新 load-from-disk 的模型，二者的预测矩阵必须 100% 一致。代码行数大幅降至 130 行左右，可读性极高。

---

## 🎯 总结
这是一次从 **“混乱的单体探索脚本”** 向 **“星状微服务架构体系”** 迈进的规范重构。
1. **控制面板与显示面板各自实现了内聚统一**：Unity 显示端（C#）和 ROS 控制端（Python）现在都同时兼容并且原生内置 Decode + Pretrain 双工作模式。
2. **职责纯粹化**：在 `CentrlControllerSSVEPNode2.py` 定型中，明确了“主控节点不要碰真实数据，只管发号施令与记录时序字典”，排除了繁重数据缓冲造成的调度卡顿。
3. **算法箱封装就绪**：`ssvep_pipeline.py` 让所有复杂的信号数学过程退居幕后，上游 ROS 节点随时可以用两行代码将其调用。

---

## 📅 2026-03-18 增量：Reasoner 分组交互测试链路

在原有 SSVEP 主线之外，本次新增了一套面向 Unity 上层交互验证的 reasoner 模式，核心文件为：

- `publisher_test/reasoner_publish_test.py`
- `eeg_processing/SSVEP_Communication_Node.py`
- `HistoryManager.cs`（沿用现有协议）

### 新增能力

- **24 图 / 4 组批次发布**：`reasoner_publish_test.py` 从 `~/Pictures/截图` 中按自然排序取前 24 张图片，切成 4 组，每组 6 张。
- **显式握手**：通信节点先发 `cmd=ssvep_ready`，reasoner 回 `cmd=reasoner_ready` 后才进入正常流程，解决两个节点启动顺序敏感的问题。
- **Decode 控制话题拆分**：主链路中的 decode 控制命令已从 `/image_seg` 拆分到独立话题 `/ssvep_decode_cmd`，与 pretrain 控制话题 `/ssvep_train_cmd` 分离。
- **参数注入式选择**：通过 `mock_selected_index` 模拟 EEG 已给出的用户选择结果。
- **history 交互**：
  - `0/1/2/4/5/6`：把对应槽位图片写入 `/history_image`
  - `3`：把所有 history 图片打包回传给 reasoner
  - `7`：删除最后一个 history，并回退到上一组 6 图

### 关键修复

- **history 缩略图尺寸修复**：reasoner 模式初版直接把 `640x480` 的主显示图发给 `/history_image`，导致 `HistoryManager.cs` 所在 ScrollView 布局被撑开。现已在 ROS 端统一缩放为 `100x100`，与原 `history_sender.py` 协议保持一致。
- **槽位索引映射修复**：reasoner 模式初版仍然沿用 decode 里的随机 `shuffle`，导致 `mock_selected_index=0` 不一定对应屏幕 0 号槽位显示的那张图。现已在 reasoner 模式中关闭 shuffle，固定屏幕槽位与输入组顺序的一一对应关系。
- **Decode stop/done 语义修复**：Unity 侧 decode 命令初版虽然已迁移到 `/ssvep_decode_cmd`，但 `stop/done` 仍沿用旧收尾逻辑，会隐藏 `stimulusPanel` 并清空当前纹理。现已改为仅停止闪烁、保留当前图片，直到下一批图片覆盖。

### 当前槽位定义

```text
0 1 2 3
4 5 6 7
```

- `3`：勾
- `7`：叉
- 动态图片槽位：`0/1/2/4/5/6`
