#!/usr/bin/env python3
"""Static configuration for SSVEP_Communication_Node4_test.

Node4_test runtime decoding now uses FBCCA with standalone preprocessing/decoder
components.

本模块是 SSVEP_Communication_Node4_test 的**静态配置**集中管理文件。所有子系统的
默认参数均在此处以 dataclass 形式声明，节点代码只暴露少量高频运行时覆盖参数
（run_mode、reasoner_mode_enabled、mock_selected_index、save_dir、image_dir、
decode_max_trials、eeg_bypass_debug），其余默认值应在此处编辑而非扩展 ROS
参数声明。

配置层次:
  SSVEPCommunicationConfig
    ├── GeneralConfig          — 通用参数（QoS、循环频率、SSVEP 频率列表、存储路径）
    ├── UnityCommConfig       — Unity 通信（IP、端口、ROS 话题名）
    ├── TriggerForwardConfig  — 触发信号 UDP 转发（本地/远端 IP:Port）
    ├── EEGServerConfig       — EEG TCP 数据源（IP、端口、通道数、采样率）
    ├── DecodeConfig          — Decode 模式参数（试次时长、图片发布、超时）
    ├── PretrainConfig        — Pretrain 模式参数（重复次数、刺激/休息时长）
    ├── ReasonerConfig        — Reasoner 分组交互（话题、LLM 流转发、历史图片 UDP）
    └── FBCCARuntimeConfig    — 运行时 FBCCA 解码与预处理（滤波器组、降采样、坏导处理）
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GeneralConfig:
    """通用运行参数。

    Attributes:
        use_reliable_qos: 是否使用 RELIABLE QoS 发布消息；False 时降级为 BEST_EFFORT。
        loop_period_s: 主定时器回调周期（秒），决定状态机轮询频率。
        startup_delay: 节点启动后进入 INIT_WAIT 状态的等待时间（秒），用于等待
            外部订阅者（如 Unity）就绪。
        num_targets: SSVEP 刺激目标总数（与 UI 插槽数量一致，默认 8）。
        ssvep_frequencies_hz: 各目标对应的 SSVEP 刺激频率列表（Hz）。列表长度须
            >= num_targets。这 8 个频率值与 Unity 端闪烁频率一一对应，选择时
            需保证频率间最小间距满足 FBCCA 可分辨性要求。
        save_dir: 数据输出根目录（相对路径会基于当前工作目录解析为绝对路径）。
    """

    use_reliable_qos: bool = True
    loop_period_s: float = 0.02
    startup_delay: float = 1.0
    num_targets: int = 8
    ssvep_frequencies_hz: List[float] = field(
        default_factory=lambda: [8.684, 9.706, 11.0, 11.786, 12.692, 13.75, 15.0, 18.333]
    )
    save_dir: str = "data/central_controller_ssvep_node4_test"


@dataclass
class UnityCommConfig:
    """与 Unity 端通信的网络和话题配置。

    Attributes:
        host_ip: 本端绑定 IP（"0.0.0.0" 表示绑定所有网卡）。
        decode_start_port: 旧版 decode 握手 UDP 端口。Node4_test 当前解码流程
            不再等待 Unity trial_started 回执，保留字段仅为兼容旧配置。
        pretrain_start_port: Pretrain 模式下接收 Unity "开始/触发" 信号的 UDP 端口。
        image_topic: 向 Unity 发布分割图像的 ROS 话题名。
        decode_command_topic: 向 Unity 发布解码命令（如 batch_start/batch_end、
            trial/target 信息）的 ROS 话题名。
        command_topic: 向 Unity 发布训练控制命令的 ROS 话题名。
    """

    host_ip: str = "0.0.0.0"
    decode_start_port: int = 10000
    pretrain_start_port: int = 10001
    image_topic: str = "/image_seg"
    decode_command_topic: str = "/ssvep_decode_cmd"
    command_topic: str = "/ssvep_train_cmd"


@dataclass
class TriggerForwardConfig:
    """触发信号 UDP 转发配置。

    用于在试次开始/结束时向 EEG 采集端发送事件触发标记（trigger），以便后续
    EEG 数据与刺激事件对齐。

    Attributes:
        local_ip: 本端（发送方）绑定的 IP 地址。
        local_port: 本端绑定的 UDP 源端口。
        remote_ip: 远端（EEG 采集端）IP 地址。
        remote_port: 远端 UDP 目标端口。
    """

    local_ip: str = "192.168.56.103"
    local_port: int = 5006
    remote_ip: str = "192.168.56.3"
    remote_port: int = 8888


@dataclass
class EEGServerConfig:
    """EEG TCP 数据源配置。

    描述与 EEG 放大器 TCP 服务器的连接参数及数据帧格式。Node4_test 通过 TCP
    连接到 EEG 服务器实时接收脑电数据。

    Attributes:
        server_ip: EEG TCP 服务器 IP 地址。
        server_port: EEG TCP 服务器监听端口。
        recv_buffer_size: 单次 TCP recv 调用的最大缓冲区字节数。
        n_channels: EEG 通道数（不含 trigger 通道）。
        frame_floats: 每帧浮点数数量，须等于 n_channels + 1（EEG 通道 + 1 个
            trigger 通道），节点启动时会校验此约束。
        fs: EEG 采集采样率（Hz），用于环形缓冲区容量计算和 FBCCA 预处理降采样。
    """

    server_ip: str = "192.168.56.3"
    server_port: int = 8712
    recv_buffer_size: int = 4096
    n_channels: int = 8
    frame_floats: int = 9
    fs: float = 1000.0


@dataclass
class DecodeConfig:
    """Decode（在线解码）模式参数。

    控制 decode 模式下试次的时间序列、图片发布行为和超时策略。

    Attributes:
        image_publish_period: 向 Unity 连续发布图片的时间间隔（秒）。
        inter_trial_interval: 相邻试次之间的间隔时间（秒），0 表示无间隔。
        trial_duration_s: 单次试次的刺激呈现时长（秒），即 EEG 数据截取窗口长度。
        pre_stim_hold_s: 刺激呈现前的准备等待时间（秒），用于让受试者注视目标。
        num_images: 单次试次中发布给 Unity 的图片数量（最多 6 张，对应 UI 图片
            插槽 0/1/2/4/5/6；插槽 3 为 confirm，插槽 7 为 undo/rollback）。
            Node4_test 解码元数据约定 `img/index/image_id/slot` 全部使用 0-based。
        max_trials: 单次 decode 会话的最大试次数，0 表示无限循环。
        start_wait_timeout_s: 旧版 decode 握手超时（秒）。Node4_test 当前解码流程
            不再使用该等待逻辑，保留字段仅为兼容旧配置。
        capture_wait_timeout_s: 刺激结束后等待 EEG 数据就绪的额外缓冲时间（秒）。
        image_height: 发布图像的高度像素数。
        image_width: 发布图像的宽度像素数。
        image_paths: 显式图片路径列表；为空时从 image_dir 自动扫描加载。
        image_dir: 图片目录路径（当 image_paths 为空时使用），可被 ROS 参数覆盖。
    """

    image_publish_period: float = 0.5
    inter_trial_interval: float = 0.0
    trial_duration_s: float = 4.0
    pre_stim_hold_s: float = 1.5
    num_images: int = 6
    max_trials: int = 1
    start_wait_timeout_s: float = 15.0
    capture_wait_timeout_s: float = 1.0
    image_height: int = 480
    image_width: int = 640
    image_paths: List[str] = field(default_factory=list)
    image_dir: str = (
        "~/workspace/eeg_robot/src/robot_ctr/graph/graph/results/"
        "segmentation_20260206_223629"
    )


@dataclass
class PretrainConfig:
    """Pretrain（数据采集）模式参数。

    控制 pretrain 模式下各阶段的时间安排。Pretrain 仅记录 EEG + 标签数据，
    不进行模型训练。

    Attributes:
        repetitions_per_target: 每个目标频率的重复试次数。
        cue_duration_s: 提示（cue）呈现时长（秒），告知受试者即将闪烁的目标。
        stim_duration_s: SSVEP 刺激闪烁持续时长（秒）。
        rest_duration_s: 试次间休息时长（秒）。
    """

    repetitions_per_target: int = 5
    cue_duration_s: float = 1.0
    stim_duration_s: float = 2.0
    rest_duration_s: float = 1.0


@dataclass
class ReasonerConfig:
    """Reasoner（外部推理器）交互配置。

    控制 Node4_test 与 reasoner_publish_test_2 等外部推理节点之间的分组图片
    接收、LLM 流转发和历史图片管理。

    Attributes:
        enabled: 是否启用 reasoner 外部图片分组模式（可通过 ROS 参数覆盖）。
        input_topic: 订阅 reasoner 发来的图片帧的 ROS 话题名。
        output_topic: 向 reasoner 发布反馈命令的 ROS 话题名。
        llm_stream_input_topic: 订阅 reasoner 发来的 LLM 文本流事件的 ROS 话题名。
        llm_stream_output_topic: 转发 LLM 文本流到 Unity 显示端的话题名。
        mock_selected_index: 测试参数：模拟 EEG 已判定的用户选择槽位索引
            （0..7，-1 表示不使用），消费后自动重置为 -1。
        history_image_topic: 发布选择历史缩略图的 ROS 话题名。
        history_image_width: 历史缩略图宽度像素数。
        history_image_height: 历史缩略图高度像素数。
        history_udp_ip: 历史图片 UDP 通知目标 IP（用于通知 Unity 删除历史等操作）。
        history_udp_port: 历史图片 UDP 通知目标端口。
    """

    enabled: bool = False
    input_topic: str = "/reasoner/images"
    output_topic: str = "/reasoner/feedback"
    llm_stream_input_topic: str = "/reasoner/llm_stream"
    llm_stream_output_topic: str = "/llm_output_stream"
    mock_selected_index: int = -1
    history_image_topic: str = "/history_image"
    history_image_width: int = 140
    history_image_height: int = 140
    history_udp_ip: str = "127.0.0.1"
    history_udp_port: int = 12001


@dataclass
class FBCCARuntimeConfig:
    """运行时 FBCCA 解码与预处理配置。

    本 dataclass 定义了 ssvep_runtime_fbcca.SSVEPFBCCARuntime 所需的全部参数，
    包括预处理链（去均值→去趋势→带通→陷波→降采样）、FBCCA 滤波器组设计参数
    以及手动坏导通道剔除策略。

    预处理流水线（在 ssvep_runtime_fbcca.preprocess_epoch 中执行）:
      1. Demean（逐通道去均值）
      2. Detrend（线性去趋势）
      3. Band-pass（Butterworth 带通滤波，频带由 bandpass_low_hz/bandpass_high_hz 决定）
      4. Notch（IIR 陷波滤波，抑制工频及谐波）
      5. Resample（降采样至 target_srate）

    Attributes:
        target_srate: FBCCA 解码目标采样率（Hz）。原始 EEG 数据将在预处理后
            降采样到此频率。
        bandpass_low_hz: 预处理带通滤波器低截止频率（Hz）。
        bandpass_high_hz: 预处理带通滤波器高截止频率（Hz）。
        bandpass_order: 预处理带通滤波器阶数。
        notch_freqs_hz: 陷波滤波器目标频率列表（Hz），默认抑制 50Hz 工频及其
            100Hz 二次谐波。
        notch_q: 陷波滤波器品质因数 Q 值，越大陷波带宽越窄。
        wp: 滤波器组各子带通带边界列表 [低, 高]（Hz）。与 ws 配对用于
            generate_filterbank 构造 FBCCA 所需的多子带滤波器组。
            默认 5 个子带，通带分别为 6-90、14-90、22-90、30-90、38-90 Hz。
        ws: 滤波器组各子带阻带边界列表 [低, 高]（Hz）。每个子带的阻带略宽于
            通带以保证过渡带性能。
        filter_order: 滤波器组各子带 Chebyshev I 型滤波器阶数。
        rp: Chebyshev I 型滤波器通带波纹（dB）。
        n_harmonics: CCA 参考信号中使用的谐波数量（含基频）。
        n_components: FBCCA 每个子带保留的 CCA 成分数。
        n_jobs: FBCCA fit/predict 并行度（1 为单线程）。
        channel_name_order: 原始 EEG 通道顺序（固定 8 通道），用于将
            manual_bad_channels 中的名称映射到通道索引。
        manual_bad_channels: 手动坏导名称列表（例如 ["O2", "Oz"]）。
            runtime 将按名称匹配并在解码前剔除这些通道。
    """

    target_srate: int = 256
    bandpass_low_hz: float = 6.0
    bandpass_high_hz: float = 100.0
    bandpass_order: int = 4
    notch_freqs_hz: List[float] = field(default_factory=lambda: [50.0, 100.0])
    notch_q: float = 35.0
    wp: List[Tuple[float, float]] = field(
        default_factory=lambda: [(6.0, 50.0), (14.0, 50.0), (22.0, 50.0)]
    )
    ws: List[Tuple[float, float]] = field(
        default_factory=lambda: [(4.0, 52.0), (12.0, 52.0), (20.0, 52.0)]
    )
    filter_order: int = 4
    rp: float = 0.5
    n_harmonics: int = 4
    n_components: int = 1
    n_jobs: int = 1
    channel_name_order: List[str] = field(
        default_factory=lambda: ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
    )
    manual_bad_channels: List[str] = field(default_factory=lambda: [])


@dataclass
class SSVEPCommunicationConfig:
    """Node4_test 顶层配置容器。

    聚合所有子模块配置为一个可整体传递的数据对象。通过 make_default_config()
    获取深拷贝实例，避免多处引用共享同一可变对象。

    Attributes:
        general: 通用运行参数。
        unity: Unity 通信配置。
        trigger_forward: 触发信号 UDP 转发配置。
        eeg_server: EEG TCP 数据源配置。
        decode: Decode 模式参数。
        pretrain: Pretrain 模式参数。
        reasoner: Reasoner 分组交互配置。
        fbcca_runtime: FBCCA 运行时解码与预处理配置。
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    unity: UnityCommConfig = field(default_factory=UnityCommConfig)
    trigger_forward: TriggerForwardConfig = field(default_factory=TriggerForwardConfig)
    eeg_server: EEGServerConfig = field(default_factory=EEGServerConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    fbcca_runtime: FBCCARuntimeConfig = field(default_factory=FBCCARuntimeConfig)


DEFAULT_SSVEP_COMMUNICATION_CONFIG = SSVEPCommunicationConfig()
"""模块级默认配置单例（只应用于读取模板，不应直接修改）。

如需获取可安全修改的配置副本，请使用 make_default_config()。"""


def make_default_config() -> SSVEPCommunicationConfig:
    """Return a deep-copied mutable config object.

    返回 DEFAULT_SSVEP_COMMUNICATION_CONFIG 的深拷贝，确保各节点实例持有
    独立的配置对象，互不影响。
    """
    return deepcopy(DEFAULT_SSVEP_COMMUNICATION_CONFIG)
