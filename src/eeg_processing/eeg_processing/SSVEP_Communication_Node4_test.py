#!/usr/bin/env python3
"""SSVEP Communication Node4_test - With Real eTRCA EEG Decoding.

This node extends Node3_1 with integrated eTRCA decoding for real-time
EEG-based selection, replacing the mock_selected_index parameter approach.

Key features:
- Automatic model training after pretrain completion (24 trials: 3 reps x 8 targets)
- Real-time EEG decoding during decode mode
- Strict mode: requires pre-trained model for decode mode

Usage:
  # Pretrain mode (collect data and train model)
  ros2 run eeg_processing ssvep_communication_node4_test --ros-args \\
    -p run_mode:=pretrain \\
    -p eeg_bypass_debug:=true

  # Decode mode (use trained model for real-time selection)
  ros2 run eeg_processing ssvep_communication_node4_test --ros-args \\
    -p run_mode:=decode \\
    -p reasoner_mode_enabled:=true \\
    -p eeg_bypass_debug:=true
"""

import os
import socket
import time
from pathlib import Path

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String as StringMsg

from .decode_2_test import DecodeModule
from .pretrain_2_test import PretrainModule
from .reasoner_2_test import ReasonerModule
from .ssvep_communication_node4_test_config import make_default_config
from .utils import CircularEEGBuffer, NodeState, TrialState


class CentralControllerSSVEPNode4Test(DecodeModule, PretrainModule, ReasonerModule, Node):
    """Unified SSVEP controller with integrated eTRCA decoding.

    This node combines:
    - DecodeModule: Decode mode with real EEG decoding
    - PretrainModule: Pretrain mode with auto-training
    - ReasonerModule: Reasoner communication for image batches
    """

    def __init__(self):
        super().__init__("central_controller_ssvep_node4_test")

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

        self.image_pub = self.create_publisher(Image, self.image_topic, qos)
        self.decode_command_pub = self.create_publisher(Image, self.decode_command_topic, qos)
        self.command_pub = self.create_publisher(Image, self.command_topic, qos)
        self.history_pub = self.create_publisher(Image, self.history_image_topic, qos)
        self.reasoner_pub = self.create_publisher(Image, self.reasoner_output_topic, qos)
        self.reasoner_sub = self.create_subscription(
            Image, self.reasoner_input_topic, self._on_reasoner_image, qos
        )
        self.llm_stream_pub = self.create_publisher(
            StringMsg, self.llm_stream_output_topic, qos
        )
        self.llm_stream_sub = self.create_subscription(
            StringMsg, self.llm_stream_input_topic, self._on_reasoner_llm_stream, qos
        )

        self.state = NodeState.INIT_WAIT
        self.state_until = time.monotonic() + self.startup_delay

        self.trial_idx = 0
        self.current_target_id = -1
        self.current_freq_hz = 0.0
        self.trial_state = TrialState()

        self._init_mode_specific()

        self.timer = self.create_timer(self.loop_period_s, self._on_timer)
        self.get_logger().info(
            "CentralControllerSSVEPNode4Test started: "
            f"mode={self.run_mode}, qos={'RELIABLE' if self.use_reliable_qos else 'BEST_EFFORT'}, "
            f"image_topic={self.image_topic}, decode_command_topic={self.decode_command_topic}, "
            f"command_topic={self.command_topic}, save_dir={self.save_dir}, "
            f"eeg_bypass_debug={self.eeg_bypass_debug}"
        )
        self.get_logger().info(
            "Node4_test eTRCA integration: "
            f"model_path={self.etrca_config.model_path}, "
            f"auto_train={self.etrca_config.auto_train_after_pretrain}"
        )
        self.get_logger().info(
            "Node4_test llm stream forward: "
            f"input={self.llm_stream_input_topic}, output={self.llm_stream_output_topic}"
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
        self.declare_parameter(
            "eeg_bypass_debug",
            False,
            descriptor=desc("调试解耦模式：开启后旁路 EEG TCP 与 trigger 发送。"),
        )
        self.declare_parameter(
            "etrca_model_path",
            self.config.etrca_decoder.model_path,
            descriptor=desc("eTRCA 模型文件路径（覆盖静态配置）。"),
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
        self.eeg_bypass_debug = self._param_bool("eeg_bypass_debug")

        # Override eTRCA model path if parameter is set
        model_path_override = self._param_str("etrca_model_path")
        if model_path_override != self.config.etrca_decoder.model_path:
            cfg.etrca_decoder.model_path = model_path_override

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
        self.etrca_config = cfg.etrca_decoder

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
        self.llm_stream_input_topic = self.reasoner_config.llm_stream_input_topic
        self.llm_stream_output_topic = self.reasoner_config.llm_stream_output_topic
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

    def _on_reasoner_llm_stream(self, msg: StringMsg) -> None:
        payload = str(msg.data or "")
        if not payload.strip():
            return
        try:
            forward = StringMsg()
            forward.data = payload
            self.llm_stream_pub.publish(forward)
        except Exception as exc:
            self.get_logger().warning(f"llm stream forward failed: {exc}")

    def _init_trigger_sender(self) -> None:
        if getattr(self, "eeg_bypass_debug", False):
            self.trigger_send_sock = None
            return
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

    def _init_mode_specific(self) -> None:
        if self.run_mode == "decode":
            self._init_decode_mode()
        else:
            self._init_pretrain_mode()

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
            handle = getattr(self, file_name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CentralControllerSSVEPNode4Test()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
