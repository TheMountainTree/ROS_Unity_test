#!/usr/bin/env python3
"""Refactored SSVEP communication node with grouped config and enum state machine."""

import random
import csv
import glob
import json
import os
import re
import socket
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

from .utils import (
    CircularEEGBuffer,
    NodeState,
    TrialState,
)
from .ssvep_communication_node2_config import make_default_config


class CentralControllerSSVEPNode2(Node):
    """用于解码与预训练采集的统一 SSVEP 控制器。"""

    def __init__(self):
        super().__init__("central_controller_ssvep_node2")

        # 仅保留运行时高频覆盖项；静态默认配置见 ssvep_communication_node2_config.py。
        desc = lambda text: ParameterDescriptor(description=text)
        self.config = make_default_config()
        self._declare_runtime_parameters(desc)
        self._load_all_configs()

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

        # 解码图片、解码控制、预训练控制分别使用独立 topic。
        self.image_pub = self.create_publisher(Image, self.image_topic, qos)
        self.decode_command_pub = self.create_publisher(Image, self.decode_command_topic, qos)
        self.command_pub = self.create_publisher(Image, self.command_topic, qos)
        self.history_pub = self.create_publisher(Image, self.history_image_topic, qos)
        self.reasoner_pub = self.create_publisher(Image, self.reasoner_output_topic, qos)
        self.reasoner_sub = self.create_subscription(
            Image, self.reasoner_input_topic, self._on_reasoner_image, qos
        )

        self.state = NodeState.INIT_WAIT
        self.state_until = time.monotonic() + self.startup_delay

        # 当前试次运行时字段
        self.trial_idx = 0
        self.current_target_id = -1
        self.current_freq_hz = 0.0
        self.trial_state = TrialState()

        # 按模式初始化
        self._init_mode_specific()

        self.timer = self.create_timer(self.loop_period_s, self._on_timer)
        self.get_logger().info(
            "CentralControllerSSVEPNode2 started: "
            f"mode={self.run_mode}, qos={'RELIABLE' if self.use_reliable_qos else 'BEST_EFFORT'}, "
            f"image_topic={self.image_topic}, decode_command_topic={self.decode_command_topic}, "
            f"command_topic={self.command_topic}, save_dir={self.save_dir}"
        )

    def _param_str(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _param_int(self, name: str) -> int:
        return int(self.get_parameter(name).get_parameter_value().integer_value)

    def _param_bool(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _declare_runtime_parameters(self, desc) -> None:
        self.declare_parameter(
            "run_mode",
            "decode",
            descriptor=desc("运行模式：decode(默认) 或 pretrain。"),
        )
        self.declare_parameter(
            "reasoner_mode_enabled",
            self.config.reasoner.enabled,
            descriptor=desc("是否启用 reasoner 外部图片分组模式。"),
        )
        self.declare_parameter(
            "mock_selected_index",
            self.config.reasoner.mock_selected_index,
            descriptor=desc("测试参数：模拟 EEG 已判定的用户选择槽位，0..7，消费后自动重置为 -1。"),
        )
        self.declare_parameter(
            "save_dir",
            self.config.general.save_dir,
            descriptor=desc("CSV 输出目录。"),
        )
        self.declare_parameter(
            "image_dir",
            os.path.expanduser(self.config.decode.image_dir),
            descriptor=desc("decode 图片目录（覆盖静态配置）。"),
        )
        self.declare_parameter(
            "decode_max_trials",
            self.config.decode.max_trials,
            descriptor=desc("decode 最大 trial 数，0 表示无限。"),
        )

    def _load_all_configs(self) -> None:
        cfg = self.config
        self.run_mode = self._param_str("run_mode").strip().lower()
        if self.run_mode not in ("decode", "pretrain"):
            self.get_logger().warning(f"invalid run_mode='{self.run_mode}', fallback to decode")
            self.run_mode = "decode"

        cfg.reasoner.enabled = self._param_bool("reasoner_mode_enabled")
        cfg.reasoner.mock_selected_index = self._param_int("mock_selected_index")
        cfg.general.save_dir = self._param_str("save_dir")
        cfg.decode.image_dir = self._param_str("image_dir")
        cfg.decode.max_trials = self._param_int("decode_max_trials")

        self.image_topic = cfg.unity.image_topic
        self.command_topic = cfg.unity.command_topic
        self.decode_command_topic = cfg.unity.decode_command_topic
        self.use_reliable_qos = cfg.general.use_reliable_qos
        self.loop_period_s = cfg.general.loop_period_s
        self.startup_delay = cfg.general.startup_delay
        self.num_targets = cfg.general.num_targets
        self.ssvep_frequencies = list(cfg.general.ssvep_frequencies_hz)
        save_dir_raw = self._param_str("save_dir")

        if self.num_targets <= 0:
            raise ValueError("num_targets must be > 0")
        if len(self.ssvep_frequencies) < self.num_targets:
            raise ValueError("ssvep_frequencies_hz length must be >= num_targets")

        if os.path.isabs(save_dir_raw):
            self.save_dir = save_dir_raw
        else:
            self.save_dir = str((Path.cwd() / save_dir_raw).resolve())
        os.makedirs(self.save_dir, exist_ok=True)

        self.reasoner_config = cfg.reasoner

        if cfg.eeg_server.n_channels <= 0:
            raise ValueError("eeg_n_channels must be > 0")
        if cfg.eeg_server.frame_floats != cfg.eeg_server.n_channels + 1:
            raise ValueError("eeg_frame_floats must equal eeg_n_channels + 1 (EEG + trigger)")
        if cfg.eeg_server.fs <= 0:
            raise ValueError("eeg_fs must be > 0")
        if (
            self.reasoner_config.history_image_width <= 0
            or self.reasoner_config.history_image_height <= 0
        ):
            raise ValueError("history_image_width and history_image_height must be > 0")

        self.reasoner_input_topic = self.reasoner_config.input_topic
        self.reasoner_output_topic = self.reasoner_config.output_topic
        self.history_image_topic = self.reasoner_config.history_image_topic
        self.history_image_width = self.reasoner_config.history_image_width
        self.history_image_height = self.reasoner_config.history_image_height
        self.reasoner_mode_enabled = self.reasoner_config.enabled
        self.history_udp_target_ip = self.reasoner_config.history_udp_ip
        self.history_udp_target_port = self.reasoner_config.history_udp_port

        self.decode_start_bind_ip = cfg.unity.host_ip
        self.decode_start_port = cfg.unity.decode_start_port
        self.train_trigger_bind_ip = cfg.unity.host_ip
        self.train_trigger_bind_port = cfg.unity.pretrain_start_port
        self.trigger_local_ip = cfg.trigger_forward.local_ip
        self.trigger_local_port = cfg.trigger_forward.local_port
        self.trigger_remote_ip = cfg.trigger_forward.remote_ip
        self.trigger_remote_port = cfg.trigger_forward.remote_port
        self.eeg_server_ip = cfg.eeg_server.server_ip
        self.eeg_server_port = cfg.eeg_server.server_port
        self.eeg_recv_buffer_size = cfg.eeg_server.recv_buffer_size
        self.eeg_n_channels = cfg.eeg_server.n_channels
        self.eeg_frame_floats = cfg.eeg_server.frame_floats
        self.eeg_fs = cfg.eeg_server.fs

    def _init_trigger_sender(self) -> None:
        if getattr(self, "trigger_send_sock", None) is not None:
            return
        self.trigger_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.trigger_send_sock.bind((self.trigger_local_ip, self.trigger_local_port))
        self.trigger_send_sock.connect((self.trigger_remote_ip, self.trigger_remote_port))

    def _init_eeg_streaming(self, buffer_seconds: float) -> None:
        self.eeg_frame_bytes = self.eeg_frame_floats * 4
        self.eeg_unpack_fmt = f"<{self.eeg_frame_floats}f"
        self.eeg_tcp_buffer = bytearray()
        self.eeg_tcp_sock = None
        self.eeg_tcp_connected = False
        self.eeg_reconnect_at = 0.0
        self.eeg_ring = CircularEEGBuffer(
            n_channels=self.eeg_n_channels,
            fs=self.eeg_fs,
            buffer_seconds=buffer_seconds,
        )
        self.dataset_x = []
        self.dataset_y = []
        self.dataset_saved = False

    def _init_decode_ack_receiver(self) -> None:
        self.decode_start_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.decode_start_sock.bind((self.decode_start_bind_ip, self.decode_start_port))
        self.decode_start_sock.setblocking(False)

    def _init_pretrain_ack_receiver(self) -> None:
        self.pretrain_trigger_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pretrain_trigger_sock.bind((self.train_trigger_bind_ip, self.train_trigger_bind_port))
        self.pretrain_trigger_sock.setblocking(False)

    # ------------------------------------------------------------------
    # 模式初始化
    # ------------------------------------------------------------------
    def _init_mode_specific(self) -> None:
        if self.run_mode == "decode":
            self._init_decode_mode()
        else:
            self._init_pretrain_mode()

    def _load_decode_config(self) -> None:
        self.decode_config = self.config.decode
        self.decode_image_period = self.decode_config.image_publish_period
        self.decode_iti = self.decode_config.inter_trial_interval
        self.decode_trial_duration_s = self.decode_config.trial_duration_s
        self.decode_max_trials = self.decode_config.max_trials
        self.decode_num_images = self.decode_config.num_images
        self.decode_pre_stim_hold_s = self.decode_config.pre_stim_hold_s
        self.decode_start_wait_timeout_s = self.decode_config.start_wait_timeout_s
        self.decode_capture_wait_timeout_s = self.decode_config.capture_wait_timeout_s
        self.image_h = self.decode_config.image_height
        self.image_w = self.decode_config.image_width
        self.image_paths = list(self.decode_config.image_paths)
        self.image_dir = os.path.expanduser(self.decode_config.image_dir)
        if self.decode_num_images <= 0:
            raise ValueError("decode_num_images must be > 0")
        if self.decode_num_images > 6:
            raise ValueError("decode_num_images must be <= 6 for Unity decode image protocol")

    def _load_pretrain_config(self) -> None:
        self.pretrain_config = self.config.pretrain
        self.pretrain_reps = self.pretrain_config.repetitions_per_target
        self.pretrain_cue_s = self.pretrain_config.cue_duration_s
        self.pretrain_stim_s = self.pretrain_config.stim_duration_s
        self.pretrain_rest_s = self.pretrain_config.rest_duration_s
        if self.pretrain_reps <= 0:
            raise ValueError("pretrain_repetitions_per_target must be > 0")

    def _reset_trial_state(self) -> None:
        self.trial_state = TrialState()
        self.current_trial_start_mono = self.trial_state.trial_start_mono
        self.current_trial_prepared_wall = self.trial_state.prepared_wall
        self.current_trial_start_wall = self.trial_state.start_wall
        self.current_start_trial_id = self.trial_state.start_trial_id
        self.current_start_status = self.trial_state.start_status
        self.current_decode_stop_mono = self.trial_state.decode_stop_mono
        self.current_epoch_mode = self.trial_state.epoch_mode
        self.current_cue_start_wall = self.trial_state.cue_start_wall
        self.current_stim_start_wall = self.trial_state.stim_start_wall
        self.current_stim_end_wall = self.trial_state.stim_end_wall
        self.current_trigger_received = self.trial_state.trigger_received
        self.current_trigger_wall = self.trial_state.trigger_wall
        self.current_stim_start_trigger_sent = self.trial_state.stim_start_trigger_sent
        self.current_stim_end_trigger_sent = self.trial_state.stim_end_trigger_sent
        self.current_stim_start_trigger_wall = self.trial_state.stim_start_trigger_wall
        self.current_stim_end_trigger_wall = self.trial_state.stim_end_trigger_wall
        self.current_stim_enter_abs = self.trial_state.stim_enter_abs
        self.current_stim_exit_abs = self.trial_state.stim_exit_abs
        self.current_stim_start_abs = self.trial_state.stim_start_abs
        self.current_stim_end_abs_inclusive = self.trial_state.stim_end_abs_inclusive
        self.current_raw_samples = self.trial_state.raw_samples
        self.current_epoch_complete = self.trial_state.epoch_complete
        self.current_epoch_saved = self.trial_state.epoch_saved
        self.current_trial_record_written = self.trial_state.trial_record_written
        self.last_trigger_value = self.trial_state.last_trigger_value
        self.epoch_start_pending = self.trial_state.epoch_start_pending

    def _init_decode_state(self) -> None:
        base_dynamic_slots = [1, 2, 3, 5, 6, 7]
        self.decode_dynamic_target_ids = base_dynamic_slots[: self.decode_num_images]
        self.history_stack: List[Dict[str, object]] = []
        self.next_history_id = 0
        self.previous_reasoner_group_images: List[Dict[str, object]] = []
        self.current_reasoner_group_images: List[Dict[str, object]] = []
        self.ready_reasoner_batches: List[List[Dict[str, object]]] = []
        self.reasoner_building_group_id = -1
        self.reasoner_building_images: Dict[int, np.ndarray] = {}
        self.reasoner_building_meta: Dict[int, Dict[str, object]] = {}
        self.reasoner_handshake_complete = False
        self.reasoner_ready_last_sent = 0.0
        self.pending_mock_selection = -1
        self.base_images = (
            self._generate_placeholders(self.decode_num_images)
            if self.reasoner_mode_enabled
            else self._load_or_generate_images(self.decode_num_images)
        )
        self.base_image_ids = list(range(1, self.decode_num_images + 1))
        self.current_trial_mapping: List[Tuple[int, int, float]] = []
        self.publish_idx = 0
        self.next_publish_at = 0.0
        self.waiting_start_trial_id = -1
        self.waiting_start_since = 0.0
        self.decode_hold_until = 0.0
        self._reset_trial_state()

    def _init_pretrain_state(self) -> None:
        self.pretrain_trial_plan = []
        for target in range(1, self.num_targets + 1):
            for _ in range(self.pretrain_reps):
                self.pretrain_trial_plan.append(target)
        random.shuffle(self.pretrain_trial_plan)
        self.pretrain_total_trials = len(self.pretrain_trial_plan)
        self._reset_trial_state()

    def _init_decode_sockets(self) -> None:
        self.history_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._init_decode_ack_receiver()
        self._init_trigger_sender()
        self._init_eeg_streaming(buffer_seconds=max(20.0, self.decode_trial_duration_s * 8.0))

    def _init_pretrain_sockets(self) -> None:
        self._init_pretrain_ack_receiver()
        self._init_trigger_sender()
        self._init_eeg_streaming(buffer_seconds=max(20.0, self.pretrain_stim_s * 8.0))

    def _init_decode_csv_files(self) -> None:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mapping_csv_path = os.path.join(self.save_dir, f"ssvep4_decode_mapping_{run_stamp}.csv")
        self.trials_csv_path = os.path.join(self.save_dir, f"ssvep4_decode_trials_{run_stamp}.csv")
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
        self.decode_eeg_csv_path = os.path.join(self.save_dir, f"ssvep4_decode_eeg_trials_{run_stamp}.csv")
        self.decode_eeg_csv_file = open(self.decode_eeg_csv_path, "w", newline="", encoding="utf-8")
        self.decode_eeg_writer = csv.writer(self.decode_eeg_csv_file)
        self.decode_eeg_writer.writerow(
            [
                "trial_id",
                "target_id",
                "target_frequency_hz",
                "trial_start_wall",
                "trial_end_wall",
                "start_status",
                "stim_start_trigger_sent",
                "stim_start_trigger_wall",
                "stim_end_trigger_sent",
                "stim_end_trigger_wall",
                "stim_enter_abs",
                "stim_exit_abs",
                "epoch_start_abs",
                "epoch_end_abs_inclusive",
                "raw_samples",
                "epoch_complete",
                "epoch_saved",
            ]
        )
        self.decode_eeg_csv_file.flush()
        self.decode_meta_csv_path = os.path.join(self.save_dir, f"ssvep4_decode_metadata_{run_stamp}.csv")
        self.decode_meta_csv_file = open(self.decode_meta_csv_path, "w", newline="", encoding="utf-8")
        self.decode_meta_writer = csv.writer(self.decode_meta_csv_file)
        self.decode_meta_writer.writerow(
            [
                "trial_id",
                "target_id",
                "label",
                "stim_start_wall",
                "stim_end_wall",
                "epoch_start_abs",
                "epoch_end_abs_inclusive",
                "n_samples",
                "epoch_complete",
            ]
        )
        self.decode_meta_csv_file.flush()

    def _init_pretrain_csv_files(self) -> None:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pretrain_csv_path = os.path.join(self.save_dir, f"ssvep4_pretrain_trials_{run_stamp}.csv")
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
                "stim_enter_abs",
                "stim_exit_abs",
                "epoch_start_abs",
                "epoch_end_abs_inclusive",
                "raw_samples",
                "epoch_complete",
                "epoch_saved",
            ]
        )
        self.pretrain_csv_file.flush()
        self.pretrain_meta_csv_path = os.path.join(self.save_dir, f"ssvep4_pretrain_metadata_{run_stamp}.csv")
        self.pretrain_meta_csv_file = open(self.pretrain_meta_csv_path, "w", newline="", encoding="utf-8")
        self.pretrain_meta_writer = csv.writer(self.pretrain_meta_csv_file)
        self.pretrain_meta_writer.writerow(
            [
                "trial_id",
                "target_id",
                "label",
                "stim_start_wall",
                "stim_end_wall",
                "epoch_start_abs",
                "epoch_end_abs_inclusive",
                "n_samples",
                "epoch_complete",
            ]
        )
        self.pretrain_meta_csv_file.flush()

    def _init_decode_mode(self) -> None:
        self._load_decode_config()
        self._init_decode_state()
        self._init_decode_sockets()
        self._init_decode_csv_files()
        self._ensure_eeg_connected(force=True)
        self.get_logger().info(
            "decode mode ready: "
            f"trial_duration={self.decode_trial_duration_s:.2f}s, max_trials={self.decode_max_trials}, "
            f"decode_num_images={self.decode_num_images}, hold={self.decode_pre_stim_hold_s:.2f}s, "
            f"decode_ack_udp={self.decode_start_bind_ip}:{self.decode_start_port}, "
            f"trigger_send_udp={self.trigger_local_ip}:{self.trigger_local_port}"
            f"->{self.trigger_remote_ip}:{self.trigger_remote_port}, "
            f"eeg_tcp={self.eeg_server_ip}:{self.eeg_server_port}, "
            f"mapping_csv={self.mapping_csv_path}, trials_csv={self.trials_csv_path}, "
            f"decode_eeg_csv={self.decode_eeg_csv_path}, decode_meta={self.decode_meta_csv_path}"
        )

    def _init_pretrain_mode(self) -> None:
        self._load_pretrain_config()
        self._init_pretrain_state()
        self._init_pretrain_sockets()
        self._init_pretrain_csv_files()
        self._ensure_eeg_connected(force=True)
        self.get_logger().info(
            "pretrain mode ready: "
            f"trials={self.pretrain_total_trials} ({self.num_targets} targets x {self.pretrain_reps}), "
            f"cue={self.pretrain_cue_s:.2f}s, stim={self.pretrain_stim_s:.2f}s, rest={self.pretrain_rest_s:.2f}s, "
            f"pretrain_ack_udp={self.train_trigger_bind_ip}:{self.train_trigger_bind_port}, "
            f"trigger_send_udp={self.trigger_local_ip}:{self.trigger_local_port}"
            f"->{self.trigger_remote_ip}:{self.trigger_remote_port}, "
            f"eeg_tcp={self.eeg_server_ip}:{self.eeg_server_port}, epoch_source=tcp_trigger_channel(1->2 inclusive), "
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

    @staticmethod
    def _parse_frame_id(frame_id: str) -> Dict[str, str]:
        meta: Dict[str, str] = {}
        if not frame_id:
            return meta
        for part in frame_id.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            meta[key.strip()] = value.strip()
        return meta

    def _on_reasoner_image(self, msg: Image) -> None:
        if not self.reasoner_mode_enabled or self.run_mode != "decode":
            return

        meta = self._parse_frame_id(msg.header.frame_id)
        cmd = meta.get("cmd", "")
        if cmd == "reasoner_ready":
            if not self.reasoner_handshake_complete:
                self.reasoner_handshake_complete = True
                self.get_logger().info("reasoner handshake complete")
            return

        if msg.encoding != "bgr8" or msg.width <= 0 or msg.height <= 0:
            return

        try:
            group_id = int(meta.get("group", "-1"))
            image_index = int(meta.get("index", "-1"))
        except ValueError:
            self.get_logger().warning(f"invalid reasoner frame_id: {msg.header.frame_id}")
            return
        if image_index < 0 or image_index >= 6:
            self.get_logger().warning(f"invalid reasoner image index={image_index}")
            return

        expected_size = int(msg.width) * int(msg.height) * 3
        if len(msg.data) != expected_size:
            self.get_logger().warning(
                f"invalid reasoner image payload size={len(msg.data)}, expected={expected_size}"
            )
            return

        if self.reasoner_building_group_id != group_id:
            if self.reasoner_building_images:
                self.get_logger().warning(
                    f"dropping incomplete reasoner group={self.reasoner_building_group_id}, "
                    f"received={len(self.reasoner_building_images)}"
                )
            self.reasoner_building_group_id = group_id
            self.reasoner_building_images = {}
            self.reasoner_building_meta = {}

        bgr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((int(msg.height), int(msg.width), 3)).copy()
        self.reasoner_building_images[image_index] = bgr
        self.reasoner_building_meta[image_index] = {
            "group": group_id,
            "index": image_index,
            "source_path": meta.get("image_path", ""),
        }

        end_flag = meta.get("end", "0") == "1"
        if len(self.reasoner_building_images) == 6 or end_flag:
            if len(self.reasoner_building_images) != 6:
                self.get_logger().warning(
                    f"reasoner group={group_id} ended before 6 images, received={len(self.reasoner_building_images)}"
                )
                return
            batch = []
            for idx in range(6):
                batch.append(
                    {
                        "group": group_id,
                        "index": idx,
                        "source_path": self.reasoner_building_meta[idx]["source_path"],
                        "image": self.reasoner_building_images[idx],
                    }
                )
            self.ready_reasoner_batches.append(batch)
            self.reasoner_building_group_id = -1
            self.reasoner_building_images = {}
            self.reasoner_building_meta = {}
            self.get_logger().info(
                f"received reasoner batch group={group_id}, queued_batches={len(self.ready_reasoner_batches)}"
            )

    def _publish_reasoner_cmd(self, cmd: str) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"cmd={cmd}"
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.reasoner_pub.publish(msg)

    def _publish_history_image_msg(self, image: np.ndarray, history_id: int) -> None:
        history_img = image
        if (
            image.shape[1] != self.history_image_width
            or image.shape[0] != self.history_image_height
        ):
            if PILImage is not None:
                rgb = image[:, :, ::-1]
                pil_img = PILImage.fromarray(rgb, mode="RGB")
                pil_img = pil_img.resize(
                    (self.history_image_width, self.history_image_height)
                )
                resized_rgb = np.asarray(pil_img, dtype=np.uint8)
                history_img = resized_rgb[:, :, ::-1].copy()
            else:
                self.get_logger().warning(
                    "PIL unavailable; history image will be published without resize"
                )
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"hist_id={history_id}"
        msg.height = int(history_img.shape[0])
        msg.width = int(history_img.shape[1])
        msg.encoding = "bgr8"
        msg.step = int(history_img.shape[1] * 3)
        msg.data = history_img.tobytes()
        self.history_pub.publish(msg)

    def _send_history_udp_command(self, payload: Dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        try:
            self.history_udp_sock.sendto(
                data, (self.history_udp_target_ip, self.history_udp_target_port)
            )
        except Exception as e:
            self.get_logger().warning(f"send history udp command failed: {e}")

    def _push_ready_reasoner_batch(self) -> bool:
        if not self.ready_reasoner_batches:
            return False
        batch = self.ready_reasoner_batches.pop(0)
        self.current_reasoner_group_images = batch
        self.base_images = [item["image"] for item in batch]
        self.base_image_ids = list(range(1, len(batch) + 1))
        self.get_logger().info(
            f"activate reasoner batch group={batch[0]['group']}, size={len(batch)}"
        )
        return True

    def _slot_to_group_image(self, slot_index: int) -> Optional[Dict[str, object]]:
        slot_map = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
        group_idx = slot_map.get(slot_index)
        if group_idx is None:
            return None
        if group_idx >= len(self.current_reasoner_group_images):
            return None
        return self.current_reasoner_group_images[group_idx]

    def _poll_mock_selected_index(self) -> None:
        value = self._param_int("mock_selected_index")
        if value == -1:
            return
        self.pending_mock_selection = value
        self.set_parameters([Parameter("mock_selected_index", value=-1)])
        self.get_logger().info(
            f"cached mock_selected_index={value}, current_state={self.state}"
        )

    def _consume_cached_mock_selection(self) -> int:
        value = self.pending_mock_selection
        self.pending_mock_selection = -1
        return value

    def _start_next_decode_trial_with_current_images(self) -> None:
        if self.reasoner_mode_enabled:
            self.state = NodeState.WAITING
            self.state_until = time.monotonic() + max(0.0, self.decode_iti)
        else:
            self.state = NodeState.WAITING
            self.state_until = time.monotonic() + max(0.0, self.decode_iti)

    def _handle_reasoner_selection(self, selection: int) -> None:
        if selection not in range(8):
            self.get_logger().warning(f"ignore invalid mock_selected_index={selection}")
            return

        if selection in (0, 1, 2, 4, 5, 6):
            selected = self._slot_to_group_image(selection)
            if selected is None:
                self.get_logger().warning(f"no current group image for selection slot={selection}")
                return
            self.next_history_id += 1
            history_item = {
                "history_id": self.next_history_id,
                "selection_slot": selection,
                "image": selected["image"],
                "source_path": selected.get("source_path", ""),
            }
            self.history_stack.append(history_item)
            self._publish_history_image_msg(history_item["image"], history_item["history_id"])
            self.previous_reasoner_group_images = [
                {
                    "group": item["group"],
                    "index": item["index"],
                    "source_path": item.get("source_path", ""),
                    "image": item["image"].copy(),
                }
                for item in self.current_reasoner_group_images
            ]
            self.current_reasoner_group_images = []
            self._publish_reasoner_cmd("request_next_group")
            self.state = NodeState.REASONER_WAIT_BATCH
            self.get_logger().info(
                f"selection slot={selection} accepted, history_id={history_item['history_id']}, "
                f"history_size={len(self.history_stack)}, waiting next reasoner batch"
            )
            return

        if selection == 3:
            history_total = len(self.history_stack)
            if history_total == 0:
                self.get_logger().info("selection slot=3, history is empty")
            for idx, item in enumerate(self.history_stack):
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = (
                    f"kind=history_return;hist_id={item['history_id']};"
                    f"source_path={os.path.basename(item.get('source_path', ''))};"
                    f"end={1 if idx == history_total - 1 else 0}"
                )
                image = item["image"]
                msg.height = int(image.shape[0])
                msg.width = int(image.shape[1])
                msg.encoding = "bgr8"
                msg.step = int(image.shape[1] * 3)
                msg.data = image.tobytes()
                self.reasoner_pub.publish(msg)
            self.get_logger().info(f"selection slot=3, returned history bundle size={history_total}")
            self._start_next_decode_trial_with_current_images()
            return

        if selection == 7:
            if not self.history_stack:
                self.get_logger().warning("selection slot=7 ignored because history is empty")
                return
            if not self.previous_reasoner_group_images:
                self.get_logger().warning("selection slot=7 ignored because previous group is unavailable")
                return
            self.history_stack.pop()
            self._send_history_udp_command({"cmd": "delete_last"})
            self.current_reasoner_group_images = [
                {
                    "group": item["group"],
                    "index": item["index"],
                    "source_path": item.get("source_path", ""),
                    "image": item["image"].copy(),
                }
                for item in self.previous_reasoner_group_images
            ]
            self.base_images = [item["image"] for item in self.current_reasoner_group_images]
            self.base_image_ids = list(range(1, len(self.current_reasoner_group_images) + 1))
            self.get_logger().info(
                f"selection slot=7, rollback history_size={len(self.history_stack)}"
            )
            self._start_next_decode_trial_with_current_images()

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
        self.decode_command_pub.publish(msg)

    def _publish_decode_stop(self, trial_id: int) -> None:
        self._publish_decode_cmd("stop", trial_id, self.current_target_id)

    def _publish_decode_done(self) -> None:
        self._publish_decode_cmd("done", self.trial_idx, 0)

    def _prepare_decode_trial(self) -> None:
        if self.reasoner_mode_enabled:
            if not self.current_reasoner_group_images:
                if not self._push_ready_reasoner_batch():
                    self.state = NodeState.REASONER_WAIT_BATCH
                    return
            self.base_images = [item["image"] for item in self.current_reasoner_group_images]
            self.base_image_ids = list(range(1, len(self.current_reasoner_group_images) + 1))

        if (not self.reasoner_mode_enabled) and self.decode_max_trials > 0 and self.trial_idx >= self.decode_max_trials:
            self.state = NodeState.DONE
            self._publish_decode_done()
            self._write_mode_trial_row()
            self._save_mode_dataset()
            self.get_logger().info("decode max_trials reached, stop scheduling")
            return

        self.trial_idx += 1
        self.publish_idx = 0
        self.next_publish_at = time.monotonic()
        self._reset_trial_state()
        self.current_trial_prepared_wall = datetime.now().isoformat(timespec="milliseconds")

        # 从 8 个闪烁目标中选择当前解码目标；Unity 用该目标编号做触发标记。
        self.current_target_id = random.randint(1, self.num_targets)
        self.current_freq_hz = self.ssvep_frequencies[self.current_target_id - 1]

        # 默认 decode 会打乱图像到动态槽位的映射；
        # reasoner 模式下需要保持屏幕槽位与输入组顺序一致，便于按 UI 索引回写 history。
        if self.reasoner_mode_enabled:
            order = list(range(self.decode_num_images))
        else:
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

        self.state = NodeState.DECODE_PUBLISHING
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
        end_wall = self.current_stim_end_wall or datetime.now().isoformat(timespec="milliseconds")
        if self.current_stim_end_wall == "":
            self.current_stim_end_wall = end_wall
        actual_s = 0.0
        if self.current_trial_start_mono > 0.0:
            stop_mono = self.current_decode_stop_mono if self.current_decode_stop_mono > 0.0 else time.monotonic()
            actual_s = max(0.0, stop_mono - self.current_trial_start_mono)

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

        if self.reasoner_mode_enabled:
            self.state = NodeState.REASONER_WAIT_SELECTION
            self.state_until = 0.0
            self.get_logger().info(
                f"[Decode Trial {self.trial_idx}] waiting mock_selected_index, "
                f"cached={self.pending_mock_selection}"
            )
        else:
            self.state = NodeState.WAITING
            self.state_until = time.monotonic() + max(0.0, self.decode_iti)

    def _enter_decode_stimulating(self, now: float, start_status: str, start_trial_id: int) -> None:
        self.state = NodeState.DECODE_STIMULATING
        self.current_trial_start_mono = now
        self.current_trial_start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_start_trial_id = start_trial_id
        self.current_start_status = start_status
        self.current_stim_start_wall = self.current_trial_start_wall
        self.current_stim_enter_abs = self.eeg_ring.latest_abs_index
        self.current_epoch_mode = "decode"
        self.current_stim_start_trigger_sent, self.current_stim_start_trigger_wall = self._send_trigger(1)

    def _enter_decode_wait_capture(self, now: float) -> None:
        self._publish_decode_stop(self.trial_idx)
        self.current_decode_stop_mono = now
        self.current_stim_end_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_stim_exit_abs = self.eeg_ring.latest_abs_index
        self.current_stim_end_trigger_sent, self.current_stim_end_trigger_wall = self._send_trigger(2)
        self.state = NodeState.DECODE_WAIT_CAPTURE
        self.state_until = now + max(0.0, self.decode_capture_wait_timeout_s)

    # ------------------------------------------------------------------
    # 预训练辅助函数
    # ------------------------------------------------------------------
    def _ensure_eeg_connected(self, force: bool = False) -> None:
        if self.run_mode not in ("pretrain", "decode"):
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
        if self.run_mode not in ("pretrain", "decode"):
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
        trigger_values: List[int] = []
        for i in range(n_frames):
            start = i * self.eeg_frame_bytes
            end = start + self.eeg_frame_bytes
            vals = struct.unpack(self.eeg_unpack_fmt, self.eeg_tcp_buffer[start:end])
            eeg_chunk[:, i] = vals[: self.eeg_n_channels]
            trigger_values.append(int(round(vals[self.eeg_n_channels])))
        del self.eeg_tcp_buffer[: n_frames * self.eeg_frame_bytes]
        start_abs, _ = self.eeg_ring.append(eeg_chunk)
        for i, trigger_value in enumerate(trigger_values):
            self._process_eeg_trigger_sample(start_abs + i, trigger_value)

    def _send_trigger(self, value: int) -> Tuple[bool, str]:
        wall = datetime.now().isoformat(timespec="milliseconds")
        payload = int(value).to_bytes(1, byteorder="little", signed=False)
        try:
            self.trigger_send_sock.send(payload)
            return True, wall
        except Exception as e:
            self.get_logger().warning(f"send trigger {value} failed: {e}")
            return False, wall

    def _process_eeg_trigger_sample(self, abs_index: int, trigger_value: int) -> None:
        if trigger_value == self.last_trigger_value:
            return

        self.last_trigger_value = trigger_value
        if self.current_epoch_mode is None:
            return

        if trigger_value == 1:
            if self.current_epoch_complete:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] ignore trigger=1 after completed epoch at abs={abs_index}"
                )
                return
            if self.epoch_start_pending:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] duplicate trigger=1 ignored at abs={abs_index}"
                )
                return
            self.current_stim_start_abs = int(abs_index)
            self.epoch_start_pending = True
            return

        if trigger_value == 2:
            if not self.epoch_start_pending:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] trigger=2 without open epoch at abs={abs_index}"
                )
                return
            if self.current_epoch_complete:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] duplicate trigger=2 ignored at abs={abs_index}"
                )
                return
            self.current_stim_end_abs_inclusive = int(abs_index)
            self.current_epoch_complete = True
            self.epoch_start_pending = False
            self._capture_epoch()

    def _capture_epoch(self) -> None:
        self.current_epoch_saved = False
        if self.current_stim_start_abs < 0 or self.current_stim_end_abs_inclusive < self.current_stim_start_abs:
            self.get_logger().warning(f"[Trial {self.trial_idx}] invalid epoch abs range")
            return
        epoch_end_exclusive = self.current_stim_end_abs_inclusive + 1
        if not self.eeg_ring.has_range(self.current_stim_start_abs, epoch_end_exclusive):
            self.get_logger().warning(
                f"[Trial {self.trial_idx}] EEG range unavailable "
                f"[{self.current_stim_start_abs}, {epoch_end_exclusive})"
            )
            return
        try:
            raw = self.eeg_ring.get_range(self.current_stim_start_abs, epoch_end_exclusive)
        except Exception as e:
            self.get_logger().warning(f"[Trial {self.trial_idx}] EEG epoch extraction failed: {e}")
            return

        self.current_raw_samples = int(raw.shape[1])
        self.dataset_x.append(raw.astype(np.float32))
        self.dataset_y.append(int(self.current_target_id))
        self.current_epoch_saved = True

        if self.run_mode == "pretrain":
            self.pretrain_meta_writer.writerow(
                [
                    self.trial_idx,
                    self.current_target_id,
                    self.current_target_id,
                    self.current_stim_start_wall,
                    self.current_stim_end_wall,
                    self.current_stim_start_abs,
                    self.current_stim_end_abs_inclusive,
                    self.current_raw_samples,
                    int(self.current_epoch_complete),
                ]
            )
            self.pretrain_meta_csv_file.flush()
        elif self.run_mode == "decode":
            self.decode_meta_writer.writerow(
                [
                    self.trial_idx,
                    self.current_target_id,
                    self.current_target_id,
                    self.current_stim_start_wall,
                    self.current_stim_end_wall,
                    self.current_stim_start_abs,
                    self.current_stim_end_abs_inclusive,
                    self.current_raw_samples,
                    int(self.current_epoch_complete),
                ]
            )
            self.decode_meta_csv_file.flush()

    def _write_mode_trial_row(self) -> None:
        if self.current_trial_record_written:
            return

        if self.run_mode == "pretrain":
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
                    self.current_stim_enter_abs,
                    self.current_stim_exit_abs,
                    self.current_stim_start_abs,
                    self.current_stim_end_abs_inclusive,
                    self.current_raw_samples,
                    int(self.current_epoch_complete),
                    int(self.current_epoch_saved),
                ]
            )
            self.pretrain_csv_file.flush()
        elif self.run_mode == "decode":
            self.decode_eeg_writer.writerow(
                [
                    self.trial_idx,
                    self.current_target_id,
                    f"{self.current_freq_hz:.3f}",
                    self.current_stim_start_wall,
                    self.current_stim_end_wall,
                    self.current_start_status,
                    int(self.current_stim_start_trigger_sent),
                    self.current_stim_start_trigger_wall,
                    int(self.current_stim_end_trigger_sent),
                    self.current_stim_end_trigger_wall,
                    self.current_stim_enter_abs,
                    self.current_stim_exit_abs,
                    self.current_stim_start_abs,
                    self.current_stim_end_abs_inclusive,
                    self.current_raw_samples,
                    int(self.current_epoch_complete),
                    int(self.current_epoch_saved),
                ]
            )
            self.decode_eeg_csv_file.flush()
        self.current_trial_record_written = True
        self.current_epoch_mode = None

    def _save_mode_dataset(self) -> None:
        if self.run_mode not in ("pretrain", "decode") or self.dataset_saved:
            return
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.dataset_x:
            X = np.empty(len(self.dataset_x), dtype=object)
            for i, epoch in enumerate(self.dataset_x):
                X[i] = epoch
            y = np.asarray(self.dataset_y, dtype=np.int32)
        else:
            X = np.asarray([], dtype=object)
            y = np.zeros((0,), dtype=np.int32)
        if self.run_mode == "pretrain":
            save_path = os.path.join(self.save_dir, f"ssvep4_pretrain_dataset_{run_stamp}.npy")
        else:
            save_path = os.path.join(self.save_dir, f"ssvep4_decode_dataset_{run_stamp}.npy")
        np.save(save_path, {"x": X, "y": y}, allow_pickle=True)
        self.dataset_saved = True
        self.get_logger().info(
            f"{self.run_mode} dataset saved: {save_path}, x_shape={X.shape}, y_shape={y.shape}"
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
            self._write_mode_trial_row()
            self._save_mode_dataset()
            self._publish_pretrain_cmd("done", self.trial_idx, 0)
            self.state = NodeState.DONE
            self.get_logger().info("pretrain trials finished")
            return

        self.trial_idx += 1
        self.current_target_id = self.pretrain_trial_plan[self.trial_idx - 1]
        self.current_freq_hz = self.ssvep_frequencies[self.current_target_id - 1]

        self._reset_trial_state()
        self.current_cue_start_wall = datetime.now().isoformat(timespec="milliseconds")

        self._publish_pretrain_cmd("cue", self.trial_idx, self.current_target_id)
        self.state = NodeState.PRETRAIN_CUEING
        self.state_until = time.monotonic() + self.pretrain_cue_s

    def _enter_pretrain_stim(self) -> None:
        self.current_stim_start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_stim_enter_abs = self.eeg_ring.latest_abs_index
        self.current_epoch_mode = "pretrain"
        self.current_stim_start_trigger_sent, self.current_stim_start_trigger_wall = self._send_trigger(1)
        self._publish_pretrain_cmd("stim", self.trial_idx, self.current_target_id)
        self.state = NodeState.PRETRAIN_STIMULATING
        self.state_until = time.monotonic() + self.pretrain_stim_s

    def _enter_pretrain_rest(self) -> None:
        self.current_stim_end_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_stim_exit_abs = self.eeg_ring.latest_abs_index
        self.current_stim_end_trigger_sent, self.current_stim_end_trigger_wall = self._send_trigger(2)
        self._publish_pretrain_cmd("rest", self.trial_idx, self.current_target_id)

        self.state = NodeState.PRETRAIN_RESTING
        self.state_until = time.monotonic() + self.pretrain_rest_s

    # ------------------------------------------------------------------
    # 主定时器循环
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        now = time.monotonic()
        if self.reasoner_mode_enabled and self.run_mode == "decode":
            self._poll_mock_selected_index()
            if (
                not self.reasoner_handshake_complete
                and now - self.reasoner_ready_last_sent >= 1.0
            ):
                self._publish_reasoner_cmd("ssvep_ready")
                self.reasoner_ready_last_sent = now
        self._poll_eeg_tcp()

        if self.state == NodeState.DONE:
            return

        if self.run_mode == "decode":
            self._handle_decode_state(now)
        else:
            self._handle_pretrain_state(now)

    def _handle_decode_state(self, now: float) -> None:
        if self.state == NodeState.INIT_WAIT:
            if now < self.state_until:
                return
            try:
                sub_count = self.image_pub.get_subscription_count()
            except Exception:
                sub_count = 1
            if sub_count < 1:
                return
            if self.reasoner_mode_enabled and not self.reasoner_handshake_complete:
                return
            if self.reasoner_mode_enabled and not self.current_reasoner_group_images:
                if not self._push_ready_reasoner_batch():
                    self.state = NodeState.REASONER_WAIT_BATCH
                    return
            self._prepare_decode_trial()
            return

        if self.state == NodeState.WAITING:
            if now >= self.state_until:
                self._prepare_decode_trial()
            return

        if self.state == NodeState.REASONER_WAIT_BATCH:
            if not self.reasoner_handshake_complete:
                return
            if self._push_ready_reasoner_batch():
                self._start_next_decode_trial_with_current_images()
            return

        if self.state == NodeState.REASONER_WAIT_SELECTION:
            selection = self._consume_cached_mock_selection()
            if selection == -1:
                return
            self._handle_reasoner_selection(selection)
            return

        if self.state == NodeState.DECODE_PUBLISHING:
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
                self.state = NodeState.DECODE_HOLD
                self.decode_hold_until = now + max(0.0, self.decode_pre_stim_hold_s)
                self._publish_decode_cmd("prepare", self.trial_idx, self.current_target_id)
            return

        if self.state == NodeState.DECODE_HOLD:
            if now < self.decode_hold_until:
                return
            self._publish_decode_cmd("stim", self.trial_idx, self.current_target_id)
            self.state = NodeState.DECODE_WAIT_START
            self.waiting_start_trial_id = self.trial_idx
            self.waiting_start_since = now
            return

        if self.state == NodeState.DECODE_WAIT_START:
            started_trial = self._poll_decode_trial_started()
            if started_trial == self.waiting_start_trial_id:
                self._enter_decode_stimulating(
                    now=now,
                    start_status="started",
                    start_trial_id=started_trial,
                )
                return

            timed_out = (
                self.decode_start_wait_timeout_s > 0.0
                and now - self.waiting_start_since >= self.decode_start_wait_timeout_s
            )
            if timed_out:
                self._enter_decode_stimulating(
                    now=now,
                    start_status="timeout_force_start",
                    start_trial_id=-1,
                )
            return

        if self.state == NodeState.DECODE_STIMULATING:
            if now - self.current_trial_start_mono >= self.decode_trial_duration_s:
                self._enter_decode_wait_capture(now)
            return

        if self.state == NodeState.DECODE_WAIT_CAPTURE:
            if self.current_epoch_complete or now >= self.state_until:
                self._write_mode_trial_row()
                self._finalize_decode_trial()
            return

    def _handle_pretrain_state(self, now: float) -> None:
        if self.state == NodeState.INIT_WAIT:
            if now < self.state_until:
                return
            self._start_pretrain_trial()
            return

        if self.state == NodeState.PRETRAIN_CUEING:
            if now >= self.state_until:
                self._enter_pretrain_stim()
            return

        if self.state == NodeState.PRETRAIN_STIMULATING:
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

        if self.state == NodeState.PRETRAIN_RESTING:
            if now >= self.state_until:
                self._write_mode_trial_row()
                self._start_pretrain_trial()
            return

    # ------------------------------------------------------------------
    # 生命周期清理
    # ------------------------------------------------------------------
    def destroy_node(self):
        if self.run_mode in ("pretrain", "decode") and self.trial_idx > 0:
            self._write_mode_trial_row()
        self._save_mode_dataset()

        for sock_name in [
            "decode_start_sock",
            "pretrain_trigger_sock",
            "trigger_send_sock",
            "eeg_tcp_sock",
            "history_udp_sock",
        ]:
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
            "decode_eeg_csv_file",
            "decode_meta_csv_file",
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
    node = CentralControllerSSVEPNode2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
