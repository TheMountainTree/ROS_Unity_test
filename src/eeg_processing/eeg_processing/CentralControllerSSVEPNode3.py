#!/usr/bin/env python3
"""
CentrlControllerSSVEPNode3
==========================
SSVEP 合并中控节点（ROS 侧状态机），支持两种模式：
1) decode（默认）：解码采集。发布图像批次给 Unity，等待 trial_started，再按时长结束。
2) pretrain：预训练采集。发布 cue/stim/rest 指令给 Unity，记录目标频率。

设计原则：
- 控制端固定在 ROS：所有 trial 时序、切换与结束均由本节点负责。
- Unity 仅负责显示（图像/闪烁/提示）与按约定回传触发信号。
"""

import random
import csv
import glob
import os
import re
import socket
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class CircularEEGBuffer:
    """Ring buffer for continuous EEG samples with absolute sample indexing."""

    def __init__(self, n_channels: int, fs: float, buffer_seconds: float):
        self.n_channels = int(n_channels)
        self.fs = float(fs)
        self.capacity = int(round(self.fs * float(buffer_seconds)))
        if self.capacity <= 0:
            raise ValueError("buffer capacity must be > 0")
        self.data = np.zeros((self.n_channels, self.capacity), dtype=np.float32)
        self.write_ptr = 0
        self.total_written = 0

    @property
    def latest_abs_index(self) -> int:
        return int(self.total_written)

    @property
    def oldest_abs_index(self) -> int:
        return max(0, int(self.total_written) - int(self.capacity))

    def append(self, chunk_ch_time: np.ndarray) -> Tuple[int, int]:
        x = np.asarray(chunk_ch_time, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk must have shape (n_channels, n_times), expected ({self.n_channels}, T)"
            )
        n = int(x.shape[1])
        if n == 0:
            return self.total_written, self.total_written
        if n >= self.capacity:
            x = x[:, -self.capacity :]
            n = self.capacity

        start_abs = int(self.total_written)
        first = min(n, self.capacity - self.write_ptr)
        second = n - first
        self.data[:, self.write_ptr : self.write_ptr + first] = x[:, :first]
        if second > 0:
            self.data[:, :second] = x[:, first:]
        self.write_ptr = (self.write_ptr + n) % self.capacity
        self.total_written += n
        return start_abs, int(self.total_written)

    def has_range(self, abs_start: int, abs_end: int) -> bool:
        if abs_end <= abs_start:
            return False
        return abs_start >= self.oldest_abs_index and abs_end <= self.latest_abs_index

    def get_range(self, abs_start: int, abs_end: int) -> np.ndarray:
        if not self.has_range(abs_start, abs_end):
            raise ValueError("requested range is not available in ring buffer")

        n = abs_end - abs_start
        rel_start = abs_start % self.capacity
        first = min(n, self.capacity - rel_start)
        second = n - first
        out = np.empty((self.n_channels, n), dtype=np.float32)
        out[:, :first] = self.data[:, rel_start : rel_start + first]
        if second > 0:
            out[:, first:] = self.data[:, :second]
        return out


class CentrlControllerSSVEPNode3(Node):
    """用于解码与预训练采集的统一 SSVEP 控制器。"""

    def __init__(self):
        super().__init__("centrl_controller_ssvep_node3")

        # ============================================================
        # 参数配置区
        # ============================================================
        # `run_mode`: 运行模式
        #   - 解码模式：默认，解码采集（图像映射 + 频率记录）
        #   - 预训练模式：预训练（提示/刺激/休息 + 目标频率记录）
        #
        # 通用参数：
        #   command_topic / image_topic / use_reliable_qos / loop_period_s / startup_delay / save_dir（通用配置项）
        #
        # 解码模式参数：
        #   解码参数与图像参数，控制图片加载、发布节奏、试次时长、确认信号等。
        #
        # 预训练模式参数：
        #   预训练参数与触发参数，控制提示/刺激/休息流程及触发记录。
        # ============================================================

        desc = lambda text: ParameterDescriptor(description=text)

        # ---------------------------
        # 通用参数
        # ---------------------------
        self.declare_parameter(
            "run_mode",
            "decode",
            descriptor=desc("运行模式：decode(默认) 或 pretrain。"),
        )
        self.declare_parameter(
            "image_topic",
            "/image_seg",
            descriptor=desc("图片发布话题（decode 模式）。"),
        )
        self.declare_parameter(
            "command_topic",
            "/ssvep_train_cmd",
            descriptor=desc("训练控制指令话题（pretrain 模式，cmd=cue/stim/rest/done）。"),
        )
        self.declare_parameter(
            "use_reliable_qos",
            True,
            descriptor=desc("是否使用 RELIABLE QoS。"),
        )
        self.declare_parameter(
            "loop_period_s",
            0.02,
            descriptor=desc("主状态机循环周期（秒）。"),
        )
        self.declare_parameter(
            "startup_delay",
            1.0,
            descriptor=desc("启动后等待 Unity 订阅就绪时间（秒）。"),
        )
        self.declare_parameter(
            "num_targets",
            8,
            descriptor=desc("目标数量。"),
        )
        self.declare_parameter(
            "ssvep_frequencies_hz",
            [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0],
            descriptor=desc("目标频率列表（Hz），长度需 >= num_targets。"),
        )
        self.declare_parameter(
            "save_dir",
            "data/central_controller_ssvep3",
            descriptor=desc("CSV 输出目录。"),
        )

        # ---------------------------
        # 解码模式参数
        # ---------------------------
        self.declare_parameter(
            "decode_image_publish_period",
            0.5,
            descriptor=desc("decode 模式下每张图片的发布间隔（秒）。"),
        )
        self.declare_parameter(
            "decode_inter_trial_interval",
            0.0,
            descriptor=desc("decode trial 之间额外等待时间（秒）。"),
        )
        self.declare_parameter(
            "decode_trial_duration_s",
            4.0,
            descriptor=desc("decode 刺激持续时长（秒）。"),
        )
        self.declare_parameter(
            "decode_pre_stim_hold_s",
            1.5,
            descriptor=desc("decode 模式图片显示后到开始闪烁前的停留时长（秒）。"),
        )
        self.declare_parameter(
            "decode_num_images",
            6,
            descriptor=desc("decode 模式每个 trial 发布的动态图数量（保持为 6）。"),
        )
        self.declare_parameter(
            "decode_max_trials",
            1,
            descriptor=desc("decode 最大 trial 数，0 表示无限。"),
        )
        self.declare_parameter(
            "decode_start_bind_ip",
            "0.0.0.0",
            descriptor=desc("接收 Unity trial_started UDP 的绑定 IP。"),
        )
        self.declare_parameter(
            "decode_start_port",
            10000,
            descriptor=desc("接收 Unity trial_started UDP 的端口。"),
        )
        self.declare_parameter(
            "decode_start_wait_timeout_s",
            15.0,
            descriptor=desc("decode 等待 trial_started 超时（秒）。"),
        )
        self.declare_parameter(
            "image_height",
            480,
            descriptor=desc("decode 图片高度。"),
        )
        self.declare_parameter(
            "image_width",
            640,
            descriptor=desc("decode 图片宽度。"),
        )
        self.declare_parameter(
            "image_paths",
            [],
            descriptor=desc("decode 图片路径列表；为空时从 image_dir 扫描。"),
        )
        self.declare_parameter(
            "image_dir",
            os.path.expanduser("~/workspace/eeg_robot/src/robot_ctr/graph/graph/results/segmentation_20260206_223629") ,# 也可改为使用本包 assets/stimuli 目录
            descriptor=desc("decode 图片目录（当 image_paths 为空时使用）。"),
        )

        # ---------------------------
        # 预训练模式参数
        # ---------------------------
        self.declare_parameter(
            "pretrain_repetitions_per_target",
            3,
            descriptor=desc("pretrain 每个目标采集次数，建议 3~5。"),
        )
        self.declare_parameter(
            "pretrain_cue_duration_s",
            2.0,
            descriptor=desc("pretrain cue 时长（秒）。"),
        )
        self.declare_parameter(
            "pretrain_stim_duration_s",
            1.5,
            descriptor=desc("pretrain stim 时长（秒）。"),
        )
        self.declare_parameter(
            "pretrain_rest_duration_s",
            1.0,
            descriptor=desc("pretrain rest 时长（秒）。"),
        )
        self.declare_parameter(
            "train_trigger_bind_ip",
            "0.0.0.0",
            descriptor=desc("接收 Unity trial_start UDP 的绑定 IP（pretrain）。"),
        )
        self.declare_parameter(
            "train_trigger_bind_port",
            10001,
            descriptor=desc("接收 Unity trial_start UDP 的端口（pretrain）。"),
        )
        self.declare_parameter(
            "trigger_local_ip",
            "0.0.0.0",
            descriptor=desc("发送到 Windows COM 转发器的 UDP trigger 本地绑定 IP。"),
        )
        self.declare_parameter(
            "trigger_local_port",
            5006,
            descriptor=desc("发送到 Windows COM 转发器的 UDP trigger 本地绑定端口。"),
        )
        self.declare_parameter(
            "trigger_remote_ip",
            "192.168.56.3",
            descriptor=desc("Windows COM 转发器 UDP IP。"),
        )
        self.declare_parameter(
            "trigger_remote_port",
            8888,
            descriptor=desc("Windows COM 转发器 UDP 端口。"),
        )
        self.declare_parameter(
            "eeg_server_ip",
            "192.168.56.3",
            descriptor=desc("EEG TCP 转发服务 IP。"),
        )
        self.declare_parameter(
            "eeg_server_port",
            8712,
            descriptor=desc("EEG TCP 转发服务端口。"),
        )
        self.declare_parameter(
            "eeg_recv_buffer_size",
            4096,
            descriptor=desc("EEG TCP 每次接收缓冲字节数。"),
        )
        self.declare_parameter(
            "eeg_n_channels",
            8,
            descriptor=desc("EEG 通道数（不含 trigger 列）。"),
        )
        self.declare_parameter(
            "eeg_frame_floats",
            9,
            descriptor=desc("TCP 每帧 float32 数量，默认 8 EEG + 1 trigger。"),
        )
        self.declare_parameter(
            "eeg_fs",
            250.0,
            descriptor=desc("EEG 采样率（Hz）。"),
        )

        # ---------------------------
        # 加载参数
        # ---------------------------
        self.run_mode = self.get_parameter("run_mode").get_parameter_value().string_value.strip().lower()
        if self.run_mode not in ("decode", "pretrain"):
            self.get_logger().warning(
                f"invalid run_mode='{self.run_mode}', fallback to decode"
            )
            self.run_mode = "decode"

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.command_topic = self.get_parameter("command_topic").get_parameter_value().string_value
        self.use_reliable_qos = (
            self.get_parameter("use_reliable_qos").get_parameter_value().bool_value
        )
        self.loop_period_s = self.get_parameter("loop_period_s").get_parameter_value().double_value
        self.startup_delay = self.get_parameter("startup_delay").get_parameter_value().double_value
        self.num_targets = int(self.get_parameter("num_targets").get_parameter_value().integer_value)
        self.ssvep_frequencies = [
            float(v)
            for v in self.get_parameter("ssvep_frequencies_hz")
            .get_parameter_value()
            .double_array_value
        ]
        save_dir_raw = self.get_parameter("save_dir").get_parameter_value().string_value

        if self.num_targets <= 0:
            raise ValueError("num_targets must be > 0")
        if len(self.ssvep_frequencies) < self.num_targets:
            raise ValueError("ssvep_frequencies_hz length must be >= num_targets")

        if os.path.isabs(save_dir_raw):
            self.save_dir = save_dir_raw
        else:
            self.save_dir = str((Path.cwd() / save_dir_raw).resolve())
        os.makedirs(self.save_dir, exist_ok=True)

        reliability = (
            QoSReliabilityPolicy.RELIABLE
            if self.use_reliable_qos
            else QoSReliabilityPolicy.BEST_EFFORT
        )
        qos = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # 解码模式使用 image_pub；预训练模式使用 command_pub。
        self.image_pub = self.create_publisher(Image, self.image_topic, qos)
        self.command_pub = self.create_publisher(Image, self.command_topic, qos)

        self.state = "init_wait"
        self.state_until = time.monotonic() + self.startup_delay

        # 当前试次运行时字段
        self.trial_idx = 0
        self.current_target_id = -1
        self.current_freq_hz = 0.0

        # 按模式初始化
        self._init_mode_specific()

        self.timer = self.create_timer(self.loop_period_s, self._on_timer)
        self.get_logger().info(
            "CentrlControllerSSVEPNode3 started: "
            f"mode={self.run_mode}, qos={'RELIABLE' if self.use_reliable_qos else 'BEST_EFFORT'}, "
            f"image_topic={self.image_topic}, command_topic={self.command_topic}, save_dir={self.save_dir}"
        )

    # ------------------------------------------------------------------
    # 模式初始化
    # ------------------------------------------------------------------
    def _init_mode_specific(self) -> None:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.run_mode == "decode":
            self.decode_image_period = (
                self.get_parameter("decode_image_publish_period").get_parameter_value().double_value
            )
            self.decode_iti = (
                self.get_parameter("decode_inter_trial_interval").get_parameter_value().double_value
            )
            self.decode_trial_duration_s = (
                self.get_parameter("decode_trial_duration_s").get_parameter_value().double_value
            )
            self.decode_max_trials = int(
                self.get_parameter("decode_max_trials").get_parameter_value().integer_value
            )
            self.decode_num_images = int(
                self.get_parameter("decode_num_images").get_parameter_value().integer_value
            )
            self.decode_pre_stim_hold_s = (
                self.get_parameter("decode_pre_stim_hold_s").get_parameter_value().double_value
            )
            self.decode_start_bind_ip = (
                self.get_parameter("decode_start_bind_ip").get_parameter_value().string_value
            )
            self.decode_start_port = int(
                self.get_parameter("decode_start_port").get_parameter_value().integer_value
            )
            self.decode_start_wait_timeout_s = (
                self.get_parameter("decode_start_wait_timeout_s").get_parameter_value().double_value
            )
            self.image_h = int(self.get_parameter("image_height").get_parameter_value().integer_value)
            self.image_w = int(self.get_parameter("image_width").get_parameter_value().integer_value)
            self.image_paths = list(
                self.get_parameter("image_paths").get_parameter_value().string_array_value
            )
            self.image_dir = self.get_parameter("image_dir").get_parameter_value().string_value

            if self.decode_num_images <= 0:
                raise ValueError("decode_num_images must be > 0")
            if self.decode_num_images > 6:
                raise ValueError("decode_num_images must be <= 6 for Unity decode image protocol")

            # 动态图像槽位固定为目标 1,2,3,5,6,7（cell3/cell7 使用静态图标）。
            base_dynamic_slots = [1, 2, 3, 5, 6, 7]
            self.decode_dynamic_target_ids = base_dynamic_slots[: self.decode_num_images]

            self.base_images = self._load_or_generate_images(self.decode_num_images)
            self.base_image_ids = list(range(1, self.decode_num_images + 1))

            self.current_trial_mapping: List[Tuple[int, int, float]] = []
            self.publish_idx = 0
            self.next_publish_at = 0.0
            self.waiting_start_trial_id = -1
            self.waiting_start_since = 0.0
            self.decode_hold_until = 0.0
            self.current_trial_start_mono = 0.0
            self.current_trial_prepared_wall = ""
            self.current_trial_start_wall = ""
            self.current_start_trial_id = -1
            self.current_start_status = "not_started"

            self.decode_start_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.decode_start_sock.bind((self.decode_start_bind_ip, self.decode_start_port))
            self.decode_start_sock.setblocking(False)

            self.mapping_csv_path = os.path.join(
                self.save_dir, f"ssvep2_decode_mapping_{run_stamp}.csv"
            )
            self.trials_csv_path = os.path.join(
                self.save_dir, f"ssvep2_decode_trials_{run_stamp}.csv"
            )
            self.mapping_csv_file = open(self.mapping_csv_path, "w", newline="", encoding="utf-8")
            self.trials_csv_file = open(self.trials_csv_path, "w", newline="", encoding="utf-8")
            self.mapping_writer = csv.writer(self.mapping_csv_file)
            self.trials_writer = csv.writer(self.trials_csv_file)
            self.mapping_writer.writerow(
                ["trial_id", "slot_id", "image_id", "frequency_hz", "trial_prepared_wall_time"]
            )
            self.trials_writer.writerow(
                [
                    "trial_id",
                    "target_id",
                    "target_frequency_hz",
                    "trial_prepared_wall_time",
                    "trial_start_wall_time",
                    "trial_end_wall_time",
                    "planned_duration_s",
                    "actual_duration_s",
                    "start_trial_id",
                    "start_status",
                ]
            )
            self.mapping_csv_file.flush()
            self.trials_csv_file.flush()

            self.get_logger().info(
                "decode mode ready: "
                f"trial_duration={self.decode_trial_duration_s:.2f}s, max_trials={self.decode_max_trials}, "
                f"decode_num_images={self.decode_num_images}, hold={self.decode_pre_stim_hold_s:.2f}s, "
                f"start_udp={self.decode_start_bind_ip}:{self.decode_start_port}, "
                f"mapping_csv={self.mapping_csv_path}, trials_csv={self.trials_csv_path}"
            )

        else:
            self.pretrain_reps = int(
                self.get_parameter("pretrain_repetitions_per_target").get_parameter_value().integer_value
            )
            self.pretrain_cue_s = (
                self.get_parameter("pretrain_cue_duration_s").get_parameter_value().double_value
            )
            self.pretrain_stim_s = (
                self.get_parameter("pretrain_stim_duration_s").get_parameter_value().double_value
            )
            self.pretrain_rest_s = (
                self.get_parameter("pretrain_rest_duration_s").get_parameter_value().double_value
            )
            self.train_trigger_bind_ip = (
                self.get_parameter("train_trigger_bind_ip").get_parameter_value().string_value
            )
            self.train_trigger_bind_port = int(
                self.get_parameter("train_trigger_bind_port").get_parameter_value().integer_value
            )
            self.trigger_local_ip = (
                self.get_parameter("trigger_local_ip").get_parameter_value().string_value
            )
            self.trigger_local_port = int(
                self.get_parameter("trigger_local_port").get_parameter_value().integer_value
            )
            self.trigger_remote_ip = (
                self.get_parameter("trigger_remote_ip").get_parameter_value().string_value
            )
            self.trigger_remote_port = int(
                self.get_parameter("trigger_remote_port").get_parameter_value().integer_value
            )
            self.eeg_server_ip = self.get_parameter("eeg_server_ip").get_parameter_value().string_value
            self.eeg_server_port = int(
                self.get_parameter("eeg_server_port").get_parameter_value().integer_value
            )
            self.eeg_recv_buffer_size = int(
                self.get_parameter("eeg_recv_buffer_size").get_parameter_value().integer_value
            )
            self.eeg_n_channels = int(
                self.get_parameter("eeg_n_channels").get_parameter_value().integer_value
            )
            self.eeg_frame_floats = int(
                self.get_parameter("eeg_frame_floats").get_parameter_value().integer_value
            )
            self.eeg_fs = self.get_parameter("eeg_fs").get_parameter_value().double_value

            if self.pretrain_reps <= 0:
                raise ValueError("pretrain_repetitions_per_target must be > 0")
            if self.eeg_n_channels <= 0:
                raise ValueError("eeg_n_channels must be > 0")
            if self.eeg_frame_floats < self.eeg_n_channels:
                raise ValueError("eeg_frame_floats must be >= eeg_n_channels")
            if self.eeg_fs <= 0:
                raise ValueError("eeg_fs must be > 0")

            # 试次计划：每个目标重复 N 次后打乱顺序。
            self.pretrain_trial_plan = []
            for target in range(1, self.num_targets + 1):
                for _ in range(self.pretrain_reps):
                    self.pretrain_trial_plan.append(target)
            random.shuffle(self.pretrain_trial_plan)
            self.pretrain_total_trials = len(self.pretrain_trial_plan)

            self.pretrain_trigger_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.pretrain_trigger_sock.bind((self.train_trigger_bind_ip, self.train_trigger_bind_port))
            self.pretrain_trigger_sock.setblocking(False)
            self.trigger_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.trigger_send_sock.bind((self.trigger_local_ip, self.trigger_local_port))
            self.trigger_send_sock.connect((self.trigger_remote_ip, self.trigger_remote_port))

            self.eeg_frame_bytes = self.eeg_frame_floats * 4
            self.eeg_unpack_fmt = f"<{self.eeg_frame_floats}f"
            self.eeg_tcp_buffer = bytearray()
            self.eeg_tcp_sock = None
            self.eeg_tcp_connected = False
            self.eeg_reconnect_at = 0.0
            self.eeg_ring = CircularEEGBuffer(
                n_channels=self.eeg_n_channels,
                fs=self.eeg_fs,
                buffer_seconds=max(20.0, self.pretrain_stim_s * 8.0),
            )
            self.target_samples = max(1, int(round(self.pretrain_stim_s * self.eeg_fs)))
            self.dataset_x: List[np.ndarray] = []
            self.dataset_y: List[int] = []
            self.dataset_saved = False

            self.pretrain_csv_path = os.path.join(
                self.save_dir, f"ssvep3_pretrain_trials_{run_stamp}.csv"
            )
            self.pretrain_csv_file = open(self.pretrain_csv_path, "w", newline="", encoding="utf-8")
            self.pretrain_writer = csv.writer(self.pretrain_csv_file)
            self.pretrain_writer.writerow(
                [
                    "trial_id",
                    "target_id",
                    "target_frequency_hz",
                    "cue_start_wall",
                    "stim_start_wall",
                    "stim_end_wall",
                    "trigger_received",
                    "trigger_wall",
                    "stim_start_trigger_sent",
                    "stim_start_trigger_wall",
                    "stim_end_trigger_sent",
                    "stim_end_trigger_wall",
                    "stim_start_abs",
                    "stim_end_abs",
                    "raw_samples",
                    "epoch_saved",
                ]
            )
            self.pretrain_csv_file.flush()
            self.pretrain_meta_csv_path = os.path.join(
                self.save_dir, f"ssvep3_pretrain_metadata_{run_stamp}.csv"
            )
            self.pretrain_meta_csv_file = open(
                self.pretrain_meta_csv_path, "w", newline="", encoding="utf-8"
            )
            self.pretrain_meta_writer = csv.writer(self.pretrain_meta_csv_file)
            self.pretrain_meta_writer.writerow(
                [
                    "trial_id",
                    "target_id",
                    "label",
                    "stim_start_wall",
                    "stim_end_wall",
                    "stim_start_abs",
                    "stim_end_abs",
                    "raw_samples",
                    "target_samples",
                    "sample_adjustment",
                ]
            )
            self.pretrain_meta_csv_file.flush()

            self.current_cue_start_wall = ""
            self.current_stim_start_wall = ""
            self.current_stim_end_wall = ""
            self.current_trigger_received = False
            self.current_trigger_wall = ""
            self.current_stim_start_trigger_sent = False
            self.current_stim_end_trigger_sent = False
            self.current_stim_start_trigger_wall = ""
            self.current_stim_end_trigger_wall = ""
            self.current_stim_start_abs = -1
            self.current_stim_end_abs = -1
            self.current_raw_samples = 0
            self.current_epoch_saved = False

            self._ensure_eeg_connected(force=True)

            self.get_logger().info(
                "pretrain mode ready: "
                f"trials={self.pretrain_total_trials} ({self.num_targets} targets x {self.pretrain_reps}), "
                f"cue={self.pretrain_cue_s:.2f}s, stim={self.pretrain_stim_s:.2f}s, rest={self.pretrain_rest_s:.2f}s, "
                f"trigger_recv_udp={self.train_trigger_bind_ip}:{self.train_trigger_bind_port}, "
                f"trigger_send_udp={self.trigger_local_ip}:{self.trigger_local_port}->{self.trigger_remote_ip}:{self.trigger_remote_port}, "
                f"eeg_tcp={self.eeg_server_ip}:{self.eeg_server_port}, "
                f"csv={self.pretrain_csv_path}, meta={self.pretrain_meta_csv_path}"
            )

    # ------------------------------------------------------------------
    # 解码辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _natural_sort_key(path: str):
        """按文件名中的数字做自然排序，例如 img2 < img10。"""
        basename = os.path.basename(path)
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', basename)]

    def _read_image_bgr(self, path: str) -> Optional[np.ndarray]:
        """读取图片并转换为 BGR，同时垂直翻转以适配 Unity 坐标系。

        OpenCV/ROS 坐标系原点在左上角（Y 轴向下），
        Unity 坐标系原点在左下角（Y 轴向上），因此需要上下翻转。
        """
        if not os.path.isfile(path) or PILImage is None:
            return None
        try:
            img = PILImage.open(path).convert("RGB").resize((self.image_w, self.image_h))
            rgb = np.asarray(img, dtype=np.uint8)
            bgr = rgb[:, :, ::-1].copy()
            bgr = np.flipud(bgr).copy()  # 垂直翻转：OpenCV 左上原点 → Unity 左下原点
            return bgr
        except Exception as e:
            self.get_logger().warning(f"Failed to read image {path}: {e}")
            return None

    def _generate_placeholders(self, n: int) -> List[np.ndarray]:
        colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ]
        out = []
        for i in range(n):
            img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
            img[:, :] = colors[i % len(colors)]
            box = 80
            x0 = self.image_w // 2 - box // 2
            y0 = self.image_h // 2 - box // 2
            img[y0 : y0 + box, x0 : x0 + box, :] = 255
            out.append(img)
        return out

    def _load_or_generate_images(self, n: int) -> List[np.ndarray]:
        candidates = []
        if self.image_paths:
            candidates = sorted(
                [p for p in self.image_paths if p], key=self._natural_sort_key
            )
        else:
            for pat in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                candidates.extend(glob.glob(os.path.join(self.image_dir, pat)))
            candidates = sorted(candidates, key=self._natural_sort_key)

        self.get_logger().info(
            f"Image candidates ({len(candidates)}): {[os.path.basename(p) for p in candidates[:n]]}"
        )

        frames = []
        for p in candidates:
            frame = self._read_image_bgr(p)
            if frame is not None:
                frames.append(frame)
            if len(frames) >= n:
                break

        if len(frames) < n:
            self.get_logger().warning(
                f"decode local images not enough ({len(frames)}/{n}), using generated placeholders"
            )
            frames.extend(self._generate_placeholders(n - len(frames)))
        return frames[:n]

    def _to_decode_image(
        self, trial_id: int, img_idx_1based: int, image_id: int, target_id: int, freq: float
    ) -> Image:
        bgr = self.base_images[image_id - 1]
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"trial={trial_id};img={img_idx_1based};image_id={image_id};target={target_id};"
            f"freq={freq:.3f};dur={self.decode_trial_duration_s:.3f}"
        )
        msg.height = int(bgr.shape[0])
        msg.width = int(bgr.shape[1])
        msg.encoding = "bgr8"
        msg.step = int(bgr.shape[1] * 3)
        msg.data = bgr.tobytes()
        return msg

    def _publish_decode_cmd(self, cmd: str, trial_id: int, target_id: int) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"cmd={cmd};trial={trial_id};target={target_id};dur={self.decode_trial_duration_s:.3f}"
        )
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.image_pub.publish(msg)

    def _publish_decode_stop(self, trial_id: int) -> None:
        self._publish_decode_cmd("decode_stop", trial_id, self.current_target_id)

    def _prepare_decode_trial(self) -> None:
        if self.decode_max_trials > 0 and self.trial_idx >= self.decode_max_trials:
            self.state = "done"
            self.get_logger().info("decode max_trials reached, stop scheduling")
            return

        self.trial_idx += 1
        self.publish_idx = 0
        self.next_publish_at = time.monotonic()
        self.current_trial_start_mono = 0.0
        self.current_trial_prepared_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_trial_start_wall = ""
        self.current_start_trial_id = -1
        self.current_start_status = "not_started"

        # 从 8 个闪烁目标中选择当前解码目标；Unity 用该目标编号做触发标记。
        self.current_target_id = random.randint(1, self.num_targets)
        self.current_freq_hz = self.ssvep_frequencies[self.current_target_id - 1]

        # 保持动态解码槽位固定（1,2,3,5,6,7），并在这些槽位上打乱图像编号。
        order = list(range(self.decode_num_images))
        random.shuffle(order)
        self.current_trial_mapping = []
        for i, slot_id in enumerate(self.decode_dynamic_target_ids):
            img_idx = order[i]
            image_id = self.base_image_ids[img_idx]
            freq = self.ssvep_frequencies[slot_id - 1]
            self.current_trial_mapping.append((slot_id, image_id, freq))

        for slot_id, image_id, freq in self.current_trial_mapping:
            self.mapping_writer.writerow(
                [self.trial_idx, slot_id, image_id, f"{freq:.3f}", self.current_trial_prepared_wall]
            )
        self.mapping_csv_file.flush()

        self.state = "decode_publishing"
        self.get_logger().info(
            f"[Decode Trial {self.trial_idx}] prepared target={self.current_target_id} "
            f"target_freq={self.current_freq_hz:.3f}Hz, start publishing {self.decode_num_images} images"
        )

    def _poll_decode_trial_started(self) -> int:
        while True:
            try:
                payload, _ = self.decode_start_sock.recvfrom(128)
            except BlockingIOError:
                return -1
            except OSError:
                return -1
            text = payload.decode("utf-8", errors="ignore").strip().lower()
            if not text.startswith("trial_started="):
                continue
            try:
                return int(text.split("=", 1)[1].strip())
            except ValueError:
                continue

    def _finalize_decode_trial(self) -> None:
        end_wall = datetime.now().isoformat(timespec="milliseconds")
        actual_s = 0.0
        if self.current_trial_start_mono > 0.0:
            actual_s = max(0.0, time.monotonic() - self.current_trial_start_mono)

        self.trials_writer.writerow(
            [
                self.trial_idx,
                self.current_target_id,
                f"{self.current_freq_hz:.3f}",
                self.current_trial_prepared_wall,
                self.current_trial_start_wall,
                end_wall,
                f"{self.decode_trial_duration_s:.3f}",
                f"{actual_s:.3f}",
                self.current_start_trial_id if self.current_start_trial_id > 0 else "",
                self.current_start_status,
            ]
        )
        self.trials_csv_file.flush()

        self.state = "waiting"
        self.state_until = time.monotonic() + max(0.0, self.decode_iti)

    # ------------------------------------------------------------------
    # 预训练辅助函数
    # ------------------------------------------------------------------
    def _ensure_eeg_connected(self, force: bool = False) -> None:
        if self.run_mode != "pretrain":
            return
        now = time.monotonic()
        if self.eeg_tcp_connected:
            return
        if not force and now < self.eeg_reconnect_at:
            return

        if self.eeg_tcp_sock is not None:
            try:
                self.eeg_tcp_sock.close()
            except Exception:
                pass
            self.eeg_tcp_sock = None

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((self.eeg_server_ip, self.eeg_server_port))
            sock.settimeout(0.001)
            self.eeg_tcp_sock = sock
            self.eeg_tcp_connected = True
            self.eeg_reconnect_at = 0.0
            self.get_logger().info(
                f"EEG TCP connected: {self.eeg_server_ip}:{self.eeg_server_port}"
            )
        except Exception as e:
            try:
                sock.close()
            except Exception:
                pass
            self.eeg_tcp_sock = None
            self.eeg_tcp_connected = False
            self.eeg_reconnect_at = now + 1.0
            self.get_logger().warning(
                f"EEG TCP connect failed: {e}; retry in 1s"
            )

    def _poll_eeg_tcp(self) -> None:
        if self.run_mode != "pretrain":
            return
        self._ensure_eeg_connected()
        if not self.eeg_tcp_connected or self.eeg_tcp_sock is None:
            return

        try:
            chunk = self.eeg_tcp_sock.recv(self.eeg_recv_buffer_size)
            if not chunk:
                raise ConnectionError("server closed connection")
            self.eeg_tcp_buffer.extend(chunk)
        except socket.timeout:
            pass
        except BlockingIOError:
            pass
        except Exception as e:
            self.eeg_tcp_connected = False
            self.eeg_reconnect_at = time.monotonic() + 1.0
            self.get_logger().warning(f"EEG TCP recv error: {e}")
            if self.eeg_tcp_sock is not None:
                try:
                    self.eeg_tcp_sock.close()
                except Exception:
                    pass
                self.eeg_tcp_sock = None
            return

        n_frames = len(self.eeg_tcp_buffer) // self.eeg_frame_bytes
        if n_frames <= 0:
            return

        eeg_chunk = np.empty((self.eeg_n_channels, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.eeg_frame_bytes
            end = start + self.eeg_frame_bytes
            vals = struct.unpack(self.eeg_unpack_fmt, self.eeg_tcp_buffer[start:end])
            eeg_chunk[:, i] = vals[: self.eeg_n_channels]
        del self.eeg_tcp_buffer[: n_frames * self.eeg_frame_bytes]
        self.eeg_ring.append(eeg_chunk)

    def _send_trigger(self, value: int) -> Tuple[bool, str]:
        wall = datetime.now().isoformat(timespec="milliseconds")
        payload = int(value).to_bytes(1, byteorder="little", signed=False)
        try:
            self.trigger_send_sock.send(payload)
            return True, wall
        except Exception as e:
            self.get_logger().warning(f"send trigger {value} failed: {e}")
            return False, wall

    def _normalize_epoch_samples(self, epoch: np.ndarray) -> Tuple[np.ndarray, str]:
        x = np.asarray(epoch, dtype=np.float32)
        n = x.shape[1]
        if n == self.target_samples:
            return x, "none"
        if n > self.target_samples:
            return x[:, : self.target_samples], "truncate"
        out = np.zeros((self.eeg_n_channels, self.target_samples), dtype=np.float32)
        out[:, :n] = x
        return out, "pad"

    def _capture_pretrain_epoch(self) -> None:
        self.current_epoch_saved = False
        if self.current_stim_start_abs < 0 or self.current_stim_end_abs <= self.current_stim_start_abs:
            self.get_logger().warning(f"[Trial {self.trial_idx}] invalid epoch abs range")
            return
        if not self.eeg_ring.has_range(self.current_stim_start_abs, self.current_stim_end_abs):
            self.get_logger().warning(
                f"[Trial {self.trial_idx}] EEG range unavailable "
                f"[{self.current_stim_start_abs}, {self.current_stim_end_abs})"
            )
            return
        try:
            raw = self.eeg_ring.get_range(self.current_stim_start_abs, self.current_stim_end_abs)
        except Exception as e:
            self.get_logger().warning(f"[Trial {self.trial_idx}] EEG epoch extraction failed: {e}")
            return

        self.current_raw_samples = int(raw.shape[1])
        fixed, adjust = self._normalize_epoch_samples(raw)
        self.dataset_x.append(fixed.astype(np.float32))
        self.dataset_y.append(int(self.current_target_id))
        self.current_epoch_saved = True

        self.pretrain_meta_writer.writerow(
            [
                self.trial_idx,
                self.current_target_id,
                self.current_target_id,
                self.current_stim_start_wall,
                self.current_stim_end_wall,
                self.current_stim_start_abs,
                self.current_stim_end_abs,
                self.current_raw_samples,
                self.target_samples,
                adjust,
            ]
        )
        self.pretrain_meta_csv_file.flush()

    def _save_pretrain_dataset(self) -> None:
        if self.run_mode != "pretrain" or self.dataset_saved:
            return
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.dataset_x:
            X = np.stack(self.dataset_x, axis=0).astype(np.float32)
            y = np.asarray(self.dataset_y, dtype=np.int32)
        else:
            X = np.zeros((0, self.eeg_n_channels, self.target_samples), dtype=np.float32)
            y = np.zeros((0,), dtype=np.int32)
        save_path = os.path.join(self.save_dir, f"ssvep3_pretrain_dataset_{run_stamp}.npy")
        np.save(save_path, {"x": X, "y": y}, allow_pickle=True)
        self.dataset_saved = True
        self.get_logger().info(
            f"pretrain dataset saved: {save_path}, x_shape={X.shape}, y_shape={y.shape}"
        )

    def _publish_pretrain_cmd(self, cmd: str, trial_id: int, target_id: int) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"cmd={cmd};trial={trial_id};target={target_id};"
            f"cue={self.pretrain_cue_s:.3f};stim={self.pretrain_stim_s:.3f};rest={self.pretrain_rest_s:.3f};"
            "freqs=" + ",".join(f"{v:.3f}" for v in self.ssvep_frequencies[: self.num_targets])
        )
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.command_pub.publish(msg)

    def _poll_pretrain_trigger(self) -> Optional[Tuple[int, int, str]]:
        while True:
            try:
                payload, _ = self.pretrain_trigger_sock.recvfrom(256)
            except BlockingIOError:
                return None
            except OSError:
                return None

            text = payload.decode("utf-8", errors="ignore").strip().lower()
            if not text.startswith("trial_start="):
                continue

            trial_id = -1
            target_id = -1
            for part in text.split(";"):
                kv = part.split("=", 1)
                if len(kv) != 2:
                    continue
                if kv[0].strip() == "trial_start":
                    try:
                        trial_id = int(kv[1].strip())
                    except ValueError:
                        trial_id = -1
                elif kv[0].strip() == "target":
                    try:
                        target_id = int(kv[1].strip())
                    except ValueError:
                        target_id = -1

            if trial_id > 0:
                wall = datetime.now().isoformat(timespec="milliseconds")
                return trial_id, target_id, wall

    def _start_pretrain_trial(self) -> None:
        if self.trial_idx >= self.pretrain_total_trials:
            self._save_pretrain_dataset()
            self._publish_pretrain_cmd("done", self.trial_idx, 0)
            self.state = "done"
            self.get_logger().info("pretrain trials finished")
            return

        self.trial_idx += 1
        self.current_target_id = self.pretrain_trial_plan[self.trial_idx - 1]
        self.current_freq_hz = self.ssvep_frequencies[self.current_target_id - 1]

        self.current_cue_start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_stim_start_wall = ""
        self.current_stim_end_wall = ""
        self.current_trigger_received = False
        self.current_trigger_wall = ""
        self.current_stim_start_trigger_sent = False
        self.current_stim_end_trigger_sent = False
        self.current_stim_start_trigger_wall = ""
        self.current_stim_end_trigger_wall = ""
        self.current_stim_start_abs = -1
        self.current_stim_end_abs = -1
        self.current_raw_samples = 0
        self.current_epoch_saved = False

        self._publish_pretrain_cmd("cue", self.trial_idx, self.current_target_id)
        self.state = "pretrain_cueing"
        self.state_until = time.monotonic() + self.pretrain_cue_s

    def _enter_pretrain_stim(self) -> None:
        self.current_stim_start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_stim_start_abs = self.eeg_ring.latest_abs_index
        self.current_stim_start_trigger_sent, self.current_stim_start_trigger_wall = self._send_trigger(1)
        self._publish_pretrain_cmd("stim", self.trial_idx, self.current_target_id)
        self.state = "pretrain_stimulating"
        self.state_until = time.monotonic() + self.pretrain_stim_s

    def _enter_pretrain_rest(self) -> None:
        self.current_stim_end_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_stim_end_abs = self.eeg_ring.latest_abs_index
        self.current_stim_end_trigger_sent, self.current_stim_end_trigger_wall = self._send_trigger(2)
        self._capture_pretrain_epoch()
        self._publish_pretrain_cmd("rest", self.trial_idx, self.current_target_id)

        self.pretrain_writer.writerow(
            [
                self.trial_idx,
                self.current_target_id,
                f"{self.current_freq_hz:.3f}",
                self.current_cue_start_wall,
                self.current_stim_start_wall,
                self.current_stim_end_wall,
                int(self.current_trigger_received),
                self.current_trigger_wall,
                int(self.current_stim_start_trigger_sent),
                self.current_stim_start_trigger_wall,
                int(self.current_stim_end_trigger_sent),
                self.current_stim_end_trigger_wall,
                self.current_stim_start_abs,
                self.current_stim_end_abs,
                self.current_raw_samples,
                int(self.current_epoch_saved),
            ]
        )
        self.pretrain_csv_file.flush()

        self.state = "pretrain_resting"
        self.state_until = time.monotonic() + self.pretrain_rest_s

    # ------------------------------------------------------------------
    # 主定时器循环
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        now = time.monotonic()
        self._poll_eeg_tcp()

        if self.state == "done":
            return

        if self.run_mode == "decode":
            self._on_timer_decode(now)
        else:
            self._on_timer_pretrain(now)

    def _on_timer_decode(self, now: float) -> None:
        if self.state == "init_wait":
            if now < self.state_until:
                return
            try:
                sub_count = self.image_pub.get_subscription_count()
            except Exception:
                sub_count = 1
            if sub_count < 1:
                return
            self._prepare_decode_trial()
            return

        if self.state == "waiting":
            if now >= self.state_until:
                self._prepare_decode_trial()
            return

        if self.state == "decode_publishing":
            if now < self.next_publish_at:
                return

            slot_id, image_id, freq = self.current_trial_mapping[self.publish_idx]
            self.image_pub.publish(
                self._to_decode_image(
                    trial_id=self.trial_idx,
                    img_idx_1based=self.publish_idx + 1,
                    image_id=image_id,
                    target_id=self.current_target_id,
                    freq=self.current_freq_hz,
                )
            )

            self.publish_idx += 1
            self.next_publish_at = now + max(0.0, self.decode_image_period)
            if self.publish_idx >= self.decode_num_images:
                self.state = "decode_hold"
                self.decode_hold_until = now + max(0.0, self.decode_pre_stim_hold_s)
                self._publish_decode_cmd("decode_prepare", self.trial_idx, self.current_target_id)
            return

        if self.state == "decode_hold":
            if now < self.decode_hold_until:
                return
            self._publish_decode_cmd("decode_stim", self.trial_idx, self.current_target_id)
            self.state = "decode_wait_start"
            self.waiting_start_trial_id = self.trial_idx
            self.waiting_start_since = now
            return

        if self.state == "decode_wait_start":
            started_trial = self._poll_decode_trial_started()
            if started_trial == self.waiting_start_trial_id:
                self.state = "decode_stimulating"
                self.current_trial_start_mono = now
                self.current_trial_start_wall = datetime.now().isoformat(timespec="milliseconds")
                self.current_start_trial_id = started_trial
                self.current_start_status = "started"
                return

            timed_out = (
                self.decode_start_wait_timeout_s > 0.0
                and now - self.waiting_start_since >= self.decode_start_wait_timeout_s
            )
            if timed_out:
                self.state = "decode_stimulating"
                self.current_trial_start_mono = now
                self.current_trial_start_wall = datetime.now().isoformat(timespec="milliseconds")
                self.current_start_trial_id = -1
                self.current_start_status = "timeout_force_start"
            return

        if self.state == "decode_stimulating":
            if now - self.current_trial_start_mono >= self.decode_trial_duration_s:
                self._publish_decode_stop(self.trial_idx)
                self._finalize_decode_trial()
            return

    def _on_timer_pretrain(self, now: float) -> None:
        if self.state == "init_wait":
            if now < self.state_until:
                return
            self._start_pretrain_trial()
            return

        if self.state == "pretrain_cueing":
            if now >= self.state_until:
                self._enter_pretrain_stim()
            return

        if self.state == "pretrain_stimulating":
            trig = self._poll_pretrain_trigger()
            if trig is not None and not self.current_trigger_received:
                trial_id, target_id, wall = trig
                if trial_id == self.trial_idx:
                    self.current_trigger_received = True
                    self.current_trigger_wall = wall
                    if target_id > 0:
                        self.current_target_id = target_id
                        self.current_freq_hz = self.ssvep_frequencies[self.current_target_id - 1]
            if now >= self.state_until:
                self._enter_pretrain_rest()
            return

        if self.state == "pretrain_resting":
            if now >= self.state_until:
                self._start_pretrain_trial()
            return

    # ------------------------------------------------------------------
    # 生命周期清理
    # ------------------------------------------------------------------
    def destroy_node(self):
        self._save_pretrain_dataset()

        for sock_name in ["decode_start_sock", "pretrain_trigger_sock", "trigger_send_sock", "eeg_tcp_sock"]:
            sock = getattr(self, sock_name, None)
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        for file_name in [
            "mapping_csv_file",
            "trials_csv_file",
            "pretrain_csv_file",
            "pretrain_meta_csv_file",
        ]:
            f = getattr(self, file_name, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CentrlControllerSSVEPNode3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
