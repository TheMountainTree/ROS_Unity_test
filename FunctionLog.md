# 函数日志文档 (Function & Class Log)

本文档记录了 `src/eeg_processing` 和 `src/publisher_test` 两个主要包中的关键类与函数，以便于追踪项目演进和代码逻辑。

## 1. 包: eeg_processing

### 1.1 中心控制节点 (Central Controllers)
实验流程控制的核心 ROS2 节点。
* **`CentralControllerNode.py`**
  * `class CentralControllerNode(Node)`: 基础实验控制节点。
  * `def main(args=None)`
* **`CentralControllerSSVEPNode.py`**
  * `class CentralControllerSSVEPNode(Node)`: 针对 SSVEP 实验的控制节点。
  * `def main(args=None)`
* **`CentralControllerSSVEPNode2.py`**
  * `class CentrlControllerSSVEPNode2(Node)`: SSVEP 控制节点 v2。
  * `def main(args=None)`
* **`CentralControllerSSVEPNode3.py`**
  * `class CircularEEGBuffer`: 用于在线数据的环形缓冲区。
  * `class CentrlControllerSSVEPNode3(Node)`: SSVEP 控制节点 v3，目前最先进的在线状态机。
  * `def main(args=None)`
* **`CentralControllerSSVEPNode4.py`**
  * `class CircularEEGBuffer`: 环形缓冲区，和 Node3 一样用于连续 EEG 样本缓存。
  * `class CentrlControllerSSVEPNode4(Node)`: SSVEP 控制节点 v4，在 Node3 基础上为 decode 模式补齐 EEG TCP、trigger 打标、epoch 提取、`.npy` 保存，并把 decode/pretrain 的通信参数整理为统一配置区。
  * 关键内部方法：
    * `_load_common_config()`
    * `_load_shared_comm_config()`
    * `_init_trigger_sender()`
    * `_init_eeg_streaming()`
    * `_init_decode_ack_receiver()`
    * `_init_pretrain_ack_receiver()`
  * `def main(args=None)`
* **`SSVEP_Communication_Node2.py`**
  * `class CentralControllerSSVEPNode2(Node)`: 面向 decode/pretrain/reasoner 联调的通信节点，使用枚举状态机与试次状态数据类，并把大部分默认配置外移到静态配置模块。
  * 关键内部方法：
    * `_declare_runtime_parameters()`
    * `_load_all_configs()`
    * `_load_decode_config()`
    * `_load_pretrain_config()`
    * `_handle_decode_state()`
    * `_handle_pretrain_state()`
  * `def main(args=None)`
* **`ssvep_communication_node2_config.py`**
  * 配置类：`GeneralConfig`, `UnityCommConfig`, `TriggerForwardConfig`, `EEGServerConfig`, `DecodeConfig`, `PretrainConfig`, `ReasonerConfig`, `SSVEPCommunicationConfig`
  * `DEFAULT_SSVEP_COMMUNICATION_CONFIG`: Node2 静态默认配置
  * `def make_default_config()`: 返回深拷贝后的可变配置对象
* **`CentralControllerSSVEPTrainNode.py`**
  * `class CircularEEGBuffer`: 环形缓冲区。
  * `class PendingCapture`: 记录待处理的数据捕获任务。
  * `class SSVEPTrialDataCollector`: SSVEP 训练试次数据收集器。
  * `class CentralControllerSSVEPTrainNode(Node)`: 专门用于 SSVEP 范式训练数据采集的节点。
  * `def main(args=None)`

### 1.2 信号处理与解码器 (Processing & Decoding)
脑电信号处理核心算法层。
* **`ssvep_pipeline.py`** (可能基于 eTRCA)
  * `class SSVEPDataLoader`: 数据加载模块。
  * `class SSVEPPretrainer`: 预训练模块（如空间滤波器计算）。
  * `class SSVEPDecoder`: 核心解码器。
  * `class SSVEPEvaluator`: 模型评估模块。
* **`ssvep_processing_fbcca.py`**
  * `class SSVEPDataLoaderFBSCCA`
  * `class SSVEPPretrainerFBSCCA`
  * `class SSVEPDecoderFBSCCA`
  * `class SSVEPEvaluatorFBSCCA`
  * `def compare_etrca_fbscca(...)`: 用于对比 eTRCA 和 FBCCA 算法效果的测试函数。
  * `def main()`
* **`ssvep_processing_etrca.py`**
  * `def main()`
* **`p300_processing_test.py`** (P300 离线分析)
  * `def bandpass_filter_trials(...)`
  * `def baseline_correction_trials(...)`
  * `def remove_artifacts_by_amplitude_threshold(...)`
  * `def remove_artifacts_by_eog_regression(...)`
  * `def trial_to_features(...)`
  * `def build_train_set(...)`
  * `def build_binary_train_set(...)`
  * `def run_p300_decoding(...)`
  * `def main()`
* **`p300_processing_test_online.py`** (P300 在线系统原型)
  * 配置类：`OnlinePreprocessConfig`, `EpochWindowConfig`, `UDPTriggerConfig`, `OnlineRunConfig`
  * `class OnlineEpochPreprocessor`: 在线分段预处理。
  * `class CircularEEGBuffer`: 环形数据缓存。
  * `class TriggeredEpochExtractor`: 基于 Trigger 的 Epoch 提取器。
  * `class OnlineP300Decoder`: P300 在线解码器。
  * `class UDPTriggerReceiver`: UDP 触发接收器。
  * `class OnlineP300UDPSystem`: 集成了上述组件的完整在线 P300 系统抽象。
  * `def main()`

### 1.3 辅助工具与测试 (Utilities & Validation)
* **`history_sender.py`**
  * `class HistorySenderNode(Node)`: 用于将历史或状态信息发送到 Unity。
  * `def main(args=None)`
* **`validate_ssvep3_npy.py`**
  * 包含一系列数据验证函数：`_convert_to_3d`, `_load_dataset`, `_check_dataset`, `_print_summary`, `_dominant_freq`, `_run_diagnostic`, `_plot_epochs`。
  * `def main()`
* **`validate_ssvep4_npy.py`**
  * 用于验证和可视化 `ssvep4_decode_dataset_*.npy`。
  * 默认采样率 1000Hz，默认仅绘制 1 个 decode epoch。
  * 主要函数：`_convert_to_3d`, `_load_dataset`, `_check_dataset`, `_print_summary`, `_dominant_freq`, `_run_diagnostic`, `_plot_epochs`。
  * `def main()`

---

## 2. 包: publisher_test

主要用于数据接入、测试流发布以及硬件接口模拟。

### 2.1 数据接收客户端
* **`eeg_tcp_listener_node.py`**
  * `class EegTcpClientNode(Node)`: 核心硬件接口，通过 TCP 接收脑电放大器的数据并发布为 ROS2 `Float32MultiArray` 消息。
  * `def main(args=None)`

### 2.2 测试与模拟发布器
* **`udp_sender_node.py`**
  * `class UdpSenderNode(Node)`: 模拟发送 UDP 控制/触发指令。
  * `def main(args=None)`
* **`image_publisher.py`**
  * `class ImagePublisher(Node)`: 图像数据发布器，用于视觉刺激或监控测试。
  * `def main(args=None)`
* **`seg_image_publisher.py`**
  * `class SegImagePublisher(Node)`: 分割图像数据发布器，可能用于特定视觉任务测试。
  * `def main(args=None)`
