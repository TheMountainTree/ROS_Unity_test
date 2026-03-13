#!/usr/bin/env python3
import csv
import os
import random
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class CircularEEGBuffer:
    """Ring buffer for continuous EEG samples with absolute sample indexing."""

    def __init__(self, n_channels: int, fs: float, buffer_seconds: float):
        self.n_channels = int(n_channels)
        self.fs = float(fs)
        self.capacity = int(round(self.fs * float(buffer_seconds)))
        if self.capacity <= 0:
            raise ValueError("buffer capacity must be > 0")
        self.data = np.zeros((self.n_channels, self.capacity), dtype=float)
        self.write_ptr = 0
        self.total_written = 0
        self.lock = threading.Lock()

    @property
    def latest_abs_index(self) -> int:
        return int(self.total_written)

    @property
    def oldest_abs_index(self) -> int:
        return max(0, int(self.total_written) - int(self.capacity))

    def append(self, chunk_ch_time: np.ndarray) -> Tuple[int, int]:
        x = np.asarray(chunk_ch_time, dtype=float)
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

        with self.lock:
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
        out = np.empty((self.n_channels, n), dtype=float)
        with self.lock:
            out[:, :first] = self.data[:, rel_start : rel_start + first]
            if second > 0:
                out[:, first:] = self.data[:, :second]
        return out


@dataclass
class PendingCapture:
    trial_id: int
    target_id: int
    trigger_abs_index: int
    start_abs_index: int
    end_abs_index: int
    trigger_wall_time: str


class SSVEPTrialDataCollector:
    """
    Receives Trial_Start trigger and cuts fixed-length EEG trial from circular buffer.

    External EEG source should continuously call `push_eeg_chunk(chunk_ch_time)`.
    """

    def __init__(self, fs: float, n_channels: int, save_dir: str, stim_duration_s: float):
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.stim_duration_s = float(stim_duration_s)
        self.buffer = CircularEEGBuffer(
            n_channels=n_channels,
            fs=fs,
            buffer_seconds=max(20.0, self.stim_duration_s * 6.0),
        )
        self.pending: List[PendingCapture] = []

        self.save_dir = save_dir
        self.trials_dir = os.path.join(save_dir, "train_trials_npy")
        os.makedirs(self.trials_dir, exist_ok=True)
        self.index_csv_path = os.path.join(save_dir, "ssvep_train_eeg_index.csv")
        self.index_csv_file = open(self.index_csv_path, "w", newline="", encoding="utf-8")
        self.index_writer = csv.writer(self.index_csv_file)
        self.index_writer.writerow(
            [
                "trial_id",
                "target_id",
                "label",
                "trigger_wall_time",
                "trigger_abs_index",
                "start_abs_index",
                "end_abs_index",
                "n_channels",
                "n_samples",
                "data_path",
            ]
        )
        self.index_csv_file.flush()

    def close(self) -> None:
        try:
            self.index_csv_file.close()
        except Exception:
            pass

    def set_stim_duration(self, stim_duration_s: float) -> None:
        self.stim_duration_s = float(stim_duration_s)

    def push_eeg_chunk(self, chunk_ch_time: np.ndarray) -> None:
        self.buffer.append(chunk_ch_time)

    def add_trial_start_trigger(
        self,
        trial_id: int,
        target_id: int,
        stim_duration_s: Optional[float] = None,
        trigger_wall_time: Optional[str] = None,
    ) -> None:
        duration = float(stim_duration_s) if stim_duration_s is not None else self.stim_duration_s
        n_samples = int(round(duration * self.fs))
        if n_samples <= 1:
            n_samples = int(round(self.fs))

        trigger_abs = self.buffer.latest_abs_index
        start_abs = trigger_abs
        end_abs = trigger_abs + n_samples
        self.pending.append(
            PendingCapture(
                trial_id=int(trial_id),
                target_id=int(target_id),
                trigger_abs_index=int(trigger_abs),
                start_abs_index=int(start_abs),
                end_abs_index=int(end_abs),
                trigger_wall_time=trigger_wall_time
                if trigger_wall_time
                else datetime.now().isoformat(timespec="milliseconds"),
            )
        )

    def pop_ready_captures(self) -> List[Dict]:
        ready: List[Dict] = []
        next_pending: List[PendingCapture] = []

        oldest = self.buffer.oldest_abs_index
        latest = self.buffer.latest_abs_index

        for item in self.pending:
            if item.end_abs_index > latest:
                next_pending.append(item)
                continue
            if item.start_abs_index < oldest:
                continue
            if not self.buffer.has_range(item.start_abs_index, item.end_abs_index):
                continue

            eeg = self.buffer.get_range(item.start_abs_index, item.end_abs_index)
            fname = f"trial_{item.trial_id:04d}_label_{item.target_id:02d}.npy"
            fpath = os.path.join(self.trials_dir, fname)
            np.save(fpath, eeg.astype(np.float32))

            self.index_writer.writerow(
                [
                    item.trial_id,
                    item.target_id,
                    item.target_id,
                    item.trigger_wall_time,
                    item.trigger_abs_index,
                    item.start_abs_index,
                    item.end_abs_index,
                    eeg.shape[0],
                    eeg.shape[1],
                    fpath,
                ]
            )
            self.index_csv_file.flush()

            ready.append(
                {
                    "trial_id": item.trial_id,
                    "target_id": item.target_id,
                    "label": item.target_id,
                    "shape": tuple(eeg.shape),
                    "path": fpath,
                }
            )

        self.pending = next_pending
        return ready


class CentralControllerSSVEPTrainNode(Node):
    """
    ROS-controlled SSVEP eTRCA data collection.

    Flow per trial:
    1) cue(2s): ROS sends cue command, Unity only highlights target.
    2) stim(1~1.5s): ROS sends stim command, Unity starts flicker and sends UDP trial_start.
    3) Python receives trial_start UDP, captures EEG segment from CircularEEGBuffer and labels it.
    4) rest(1s): ROS sends rest command then schedules next trial.
    """

    def __init__(self):
        super().__init__("central_controller_ssvep_train_node")

        desc = lambda text: ParameterDescriptor(description=text)

        self.declare_parameter(
            "command_topic",
            "/ssvep_train_cmd",
            descriptor=desc("发布给 Unity 的训练控制话题，消息头中包含 cmd/trial/target。"),
        )
        self.declare_parameter(
            "use_reliable_qos",
            True,
            descriptor=desc("是否使用 RELIABLE QoS 发布控制指令。"),
        )
        self.declare_parameter(
            "loop_period_s",
            0.02,
            descriptor=desc("主状态机循环周期（秒）。"),
        )
        self.declare_parameter(
            "startup_delay",
            1.0,
            descriptor=desc("节点启动后等待 Unity 就绪的延迟（秒）。"),
        )

        self.declare_parameter(
            "num_targets",
            6,
            descriptor=desc("SSVEP 目标数量。"),
        )
        self.declare_parameter(
            "repetitions_per_target",
            3,
            descriptor=desc("每个目标采集次数，建议 3~5。"),
        )
        self.declare_parameter(
            "cue_duration_s",
            1.0,
            descriptor=desc("提示阶段时长（秒）：只高亮目标红框。"),
        )
        self.declare_parameter(
            "stim_duration_s",
            4.0,
            descriptor=desc("刺激阶段时长（秒）：所有目标按频率闪烁。"),
        )
        self.declare_parameter(
            "rest_duration_s",
            1.0,
            descriptor=desc("试次间休息时长（秒）。"),
        )
        self.declare_parameter(
            "ssvep_frequencies_hz",
            [8.0, 10.0, 12.0, 15.0, 20.0, 30.0],
            descriptor=desc("目标频率列表（Hz），长度需 >= num_targets。"),
        )

        self.declare_parameter(
            "trigger_bind_ip",
            "0.0.0.0",
            descriptor=desc("接收 Unity trial_start UDP 触发的绑定 IP。"),
        )
        self.declare_parameter(
            "trigger_bind_port",
            10001,
            descriptor=desc("接收 Unity trial_start UDP 触发的端口。"),
        )
        self.declare_parameter(
            "trigger_wait_timeout_s",
            1.0,
            descriptor=desc("刺激阶段等待 trial_start 的超时告警阈值（秒）。"),
        )

        self.declare_parameter(
            "eeg_fs",
            250.0,
            descriptor=desc("EEG 采样率（Hz），用于按秒换算采样点数。"),
        )
        self.declare_parameter(
            "eeg_n_channels",
            8,
            descriptor=desc("EEG 通道数。"),
        )
        self.declare_parameter(
            "save_dir",
            "/home/themountaintree/workspace/ROS_Unity_test/data/central_controller_ssvep_train",
            descriptor=desc("训练数据输出目录（CSV 索引与 trial npy）。"),
        )

        self.command_topic = self.get_parameter("command_topic").get_parameter_value().string_value
        self.use_reliable_qos = (
            self.get_parameter("use_reliable_qos").get_parameter_value().bool_value
        )
        self.loop_period_s = self.get_parameter("loop_period_s").get_parameter_value().double_value
        self.startup_delay = self.get_parameter("startup_delay").get_parameter_value().double_value

        self.num_targets = int(self.get_parameter("num_targets").get_parameter_value().integer_value)
        self.repetitions_per_target = int(
            self.get_parameter("repetitions_per_target").get_parameter_value().integer_value
        )
        self.cue_duration_s = self.get_parameter("cue_duration_s").get_parameter_value().double_value
        self.stim_duration_s = self.get_parameter("stim_duration_s").get_parameter_value().double_value
        self.rest_duration_s = self.get_parameter("rest_duration_s").get_parameter_value().double_value
        self.ssvep_frequencies = [
            float(v)
            for v in self.get_parameter("ssvep_frequencies_hz")
            .get_parameter_value()
            .double_array_value
        ]

        self.trigger_bind_ip = self.get_parameter("trigger_bind_ip").get_parameter_value().string_value
        self.trigger_bind_port = int(
            self.get_parameter("trigger_bind_port").get_parameter_value().integer_value
        )
        self.trigger_wait_timeout_s = (
            self.get_parameter("trigger_wait_timeout_s").get_parameter_value().double_value
        )

        self.eeg_fs = self.get_parameter("eeg_fs").get_parameter_value().double_value
        self.eeg_n_channels = int(self.get_parameter("eeg_n_channels").get_parameter_value().integer_value)
        self.save_dir = self.get_parameter("save_dir").get_parameter_value().string_value

        if self.num_targets <= 0:
            raise ValueError("num_targets must be > 0")
        if self.repetitions_per_target <= 0:
            raise ValueError("repetitions_per_target must be > 0")
        if len(self.ssvep_frequencies) < self.num_targets:
            raise ValueError("ssvep_frequencies_hz length must be >= num_targets")

        os.makedirs(self.save_dir, exist_ok=True)
        self.trial_plan = self._build_trial_plan(self.num_targets, self.repetitions_per_target)
        self.total_trials = len(self.trial_plan)

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
        self.cmd_pub = self.create_publisher(Image, self.command_topic, qos)

        self.trigger_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.trigger_sock.bind((self.trigger_bind_ip, self.trigger_bind_port))
        self.trigger_sock.setblocking(False)

        self.collector = SSVEPTrialDataCollector(
            fs=self.eeg_fs,
            n_channels=self.eeg_n_channels,
            save_dir=self.save_dir,
            stim_duration_s=self.stim_duration_s,
        )

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trial_csv_path = os.path.join(self.save_dir, f"ssvep_train_trials_{run_stamp}.csv")
        self.trial_csv_file = open(self.trial_csv_path, "w", newline="", encoding="utf-8")
        self.trial_writer = csv.writer(self.trial_csv_file)
        self.trial_writer.writerow(
            [
                "trial_id",
                "target_id",
                "frequency_hz",
                "cue_start_wall",
                "stim_start_wall",
                "stim_end_wall",
                "trigger_received",
                "trigger_wall",
                "capture_saved",
                "capture_path",
            ]
        )
        self.trial_csv_file.flush()

        self.state = "init_wait"
        self.state_until = time.monotonic() + self.startup_delay
        self.trial_idx = 0
        self.current_target = -1
        self.current_freq = 0.0
        self.current_cue_start_wall = ""
        self.current_stim_start_wall = ""
        self.current_stim_end_wall = ""
        self.current_trigger_received = False
        self.current_trigger_wall = ""
        self.current_capture_path = ""
        self.current_stim_enter_mono = 0.0
        self.current_trigger_timeout_warned = False

        self.timer = self.create_timer(self.loop_period_s, self._on_timer)

        self.get_logger().info(
            "CentralControllerSSVEPTrainNode started: "
            f"topic={self.command_topic}, qos={'RELIABLE' if self.use_reliable_qos else 'BEST_EFFORT'}, "
            f"trials={self.total_trials} ({self.num_targets} targets x {self.repetitions_per_target} reps), "
            f"cue={self.cue_duration_s:.2f}s, stim={self.stim_duration_s:.2f}s, rest={self.rest_duration_s:.2f}s, "
            f"trigger_udp={self.trigger_bind_ip}:{self.trigger_bind_port}, "
            f"save_dir={self.save_dir}"
        )

    def _build_trial_plan(self, num_targets: int, reps: int) -> List[int]:
        plan = []
        for target in range(1, num_targets + 1):
            for _ in range(reps):
                plan.append(target)
        random.shuffle(plan)
        return plan

    def push_eeg_chunk(self, chunk_ch_time: np.ndarray) -> None:
        """External EEG stream should call this continuously during runtime."""
        self.collector.push_eeg_chunk(chunk_ch_time)

    def _publish_command(self, cmd: str, trial_id: int, target_id: int) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()

        parts = [
            f"cmd={cmd}",
            f"trial={trial_id}",
            f"target={target_id}",
            f"cue={self.cue_duration_s:.3f}",
            f"stim={self.stim_duration_s:.3f}",
            f"rest={self.rest_duration_s:.3f}",
            "freqs=" + ",".join(f"{v:.3f}" for v in self.ssvep_frequencies[: self.num_targets]),
        ]
        msg.header.frame_id = ";".join(parts)
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.cmd_pub.publish(msg)

    def _poll_trial_start_trigger(self) -> Optional[Tuple[int, int, str]]:
        while True:
            try:
                payload, addr = self.trigger_sock.recvfrom(256)
            except BlockingIOError:
                return None
            except OSError:
                return None

            text = payload.decode("utf-8", errors="ignore").strip().lower()
            # expected: trial_start=12;target=2
            if not text.startswith("trial_start="):
                continue

            trial_id = -1
            target_id = -1
            for part in text.split(";"):
                kv = part.split("=", 1)
                if len(kv) != 2:
                    continue
                key = kv[0].strip()
                val = kv[1].strip()
                if key == "trial_start":
                    try:
                        trial_id = int(val)
                    except ValueError:
                        trial_id = -1
                elif key == "target":
                    try:
                        target_id = int(val)
                    except ValueError:
                        target_id = -1

            if trial_id <= 0:
                continue

            wall = datetime.now().isoformat(timespec="milliseconds")
            self.get_logger().info(f"trigger recv from {addr}: '{text}'")
            return trial_id, target_id, wall

    def _start_next_trial(self) -> None:
        if self.trial_idx >= self.total_trials:
            self.state = "done"
            self._publish_command("done", self.trial_idx, 0)
            self.get_logger().info("all planned trials finished")
            return

        self.trial_idx += 1
        self.current_target = self.trial_plan[self.trial_idx - 1]
        self.current_freq = self.ssvep_frequencies[self.current_target - 1]
        self.current_cue_start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_stim_start_wall = ""
        self.current_stim_end_wall = ""
        self.current_trigger_received = False
        self.current_trigger_wall = ""
        self.current_capture_path = ""
        self.current_stim_enter_mono = 0.0
        self.current_trigger_timeout_warned = False

        self._publish_command("cue", self.trial_idx, self.current_target)
        self.state = "cueing"
        self.state_until = time.monotonic() + self.cue_duration_s
        self.get_logger().info(
            f"[Trial {self.trial_idx}/{self.total_trials}] cue target={self.current_target} "
            f"freq={self.current_freq:.2f}Hz for {self.cue_duration_s:.2f}s"
        )

    def _enter_stim(self) -> None:
        self.current_stim_start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.collector.set_stim_duration(self.stim_duration_s)
        self._publish_command("stim", self.trial_idx, self.current_target)
        self.state = "stimulating"
        self.current_stim_enter_mono = time.monotonic()
        self.current_trigger_timeout_warned = False
        self.state_until = time.monotonic() + self.stim_duration_s
        self.get_logger().info(
            f"[Trial {self.trial_idx}] stimulating {self.stim_duration_s:.2f}s, wait trial_start trigger"
        )

    def _enter_rest(self) -> None:
        self.current_stim_end_wall = datetime.now().isoformat(timespec="milliseconds")
        self._publish_command("rest", self.trial_idx, self.current_target)

        capture_saved = bool(self.current_capture_path)
        self.trial_writer.writerow(
            [
                self.trial_idx,
                self.current_target,
                f"{self.current_freq:.3f}",
                self.current_cue_start_wall,
                self.current_stim_start_wall,
                self.current_stim_end_wall,
                int(self.current_trigger_received),
                self.current_trigger_wall,
                int(capture_saved),
                self.current_capture_path,
            ]
        )
        self.trial_csv_file.flush()

        self.state = "resting"
        self.state_until = time.monotonic() + self.rest_duration_s
        self.get_logger().info(
            f"[Trial {self.trial_idx}] rest {self.rest_duration_s:.2f}s, "
            f"trigger_received={self.current_trigger_received}, capture_saved={capture_saved}"
        )

    def _on_timer(self) -> None:
        now = time.monotonic()

        # Drain pending ready captures each loop
        for item in self.collector.pop_ready_captures():
            if item["trial_id"] == self.trial_idx:
                self.current_capture_path = item["path"]
            self.get_logger().info(
                f"capture saved trial={item['trial_id']} label={item['label']} shape={item['shape']}"
            )

        if self.state == "done":
            return

        if self.state == "init_wait":
            if now < self.state_until:
                return
            self._start_next_trial()
            return

        if self.state == "cueing":
            if now >= self.state_until:
                self._enter_stim()
            return

        if self.state == "stimulating":
            trigger = self._poll_trial_start_trigger()
            if trigger is not None:
                trial_id, target_id, wall = trigger
                if trial_id == self.trial_idx and not self.current_trigger_received:
                    self.current_trigger_received = True
                    self.current_trigger_wall = wall
                    labeled_target = target_id if target_id > 0 else self.current_target
                    self.collector.add_trial_start_trigger(
                        trial_id=trial_id,
                        target_id=labeled_target,
                        stim_duration_s=self.stim_duration_s,
                        trigger_wall_time=wall,
                    )
            if (
                not self.current_trigger_received
                and not self.current_trigger_timeout_warned
                and self.trigger_wait_timeout_s > 0.0
                and now - self.current_stim_enter_mono >= self.trigger_wait_timeout_s
            ):
                self.current_trigger_timeout_warned = True
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] no trial_start trigger after "
                    f"{self.trigger_wait_timeout_s:.2f}s"
                )
            if now >= self.state_until:
                self._enter_rest()
            return

        if self.state == "resting":
            if now >= self.state_until:
                self._start_next_trial()
            return

    def destroy_node(self):
        try:
            self.trigger_sock.close()
        except Exception:
            pass
        try:
            self.collector.close()
        except Exception:
            pass
        try:
            self.trial_csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CentralControllerSSVEPTrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
