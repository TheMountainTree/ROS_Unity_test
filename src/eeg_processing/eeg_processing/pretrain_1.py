#!/usr/bin/env python3
"""Pretrain and shared EEG capture module for SSVEP communication node."""

import csv
import os
import random
import socket
import struct
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sensor_msgs.msg import Image

from .utils import NodeState, TrialState


class PretrainModule:
    """Mix-in for pretrain state machine and shared EEG/epoch handling."""

    def _load_pretrain_config(self) -> None:
        self.pretrain_config = self.config.pretrain
        self.pretrain_reps = self.pretrain_config.repetitions_per_target
        self.pretrain_cue_s = self.pretrain_config.cue_duration_s
        self.pretrain_stim_s = self.pretrain_config.stim_duration_s
        self.pretrain_rest_s = self.pretrain_config.rest_duration_s
        if self.pretrain_reps <= 0:
            raise ValueError("pretrain_repetitions_per_target must be > 0")

    def _reset_trial_state(self) -> None:
        """Reset per-trial state by creating a fresh TrialState instance."""
        self.trial_state = TrialState()

    def _init_pretrain_state(self) -> None:
        self.pretrain_trial_plan = []
        for target in range(1, self.num_targets + 1):
            for _ in range(self.pretrain_reps):
                self.pretrain_trial_plan.append(target)
        random.shuffle(self.pretrain_trial_plan)
        self.pretrain_total_trials = len(self.pretrain_trial_plan)
        self._reset_trial_state()

    def _init_pretrain_sockets(self) -> None:
        self._init_pretrain_ack_receiver()
        self._init_trigger_sender()
        self._init_eeg_streaming(buffer_seconds=max(20.0, self.pretrain_stim_s * 8.0))

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

    def _init_pretrain_mode(self) -> None:
        self._load_pretrain_config()
        self._init_pretrain_state()
        self._init_pretrain_sockets()
        self._init_pretrain_csv_files()
        self._ensure_eeg_connected(force=True)
        eeg_source_text = (
            "bypass_debug(no_tcp_no_trigger)" if getattr(self, "eeg_bypass_debug", False)
            else "tcp_trigger_channel(1->2 inclusive)"
        )
        self.get_logger().info(
            "pretrain mode ready: "
            f"trials={self.pretrain_total_trials} ({self.num_targets} targets x {self.pretrain_reps}), "
            f"cue={self.pretrain_cue_s:.2f}s, stim={self.pretrain_stim_s:.2f}s, rest={self.pretrain_rest_s:.2f}s, "
            f"pretrain_ack_udp={self.train_trigger_bind_ip}:{self.train_trigger_bind_port}, "
            f"trigger_send_udp={self.trigger_local_ip}:{self.trigger_local_port}"
            f"->{self.trigger_remote_ip}:{self.trigger_remote_port}, "
            f"eeg_tcp={self.eeg_server_ip}:{self.eeg_server_port}, epoch_source={eeg_source_text}, "
            f"csv={self.pretrain_csv_path}, meta={self.pretrain_meta_csv_path}"
        )

    def _ensure_eeg_connected(self, force: bool = False) -> None:
        if getattr(self, "eeg_bypass_debug", False):
            if self.eeg_tcp_sock is not None:
                try:
                    self.eeg_tcp_sock.close()
                except Exception:
                    pass
                self.eeg_tcp_sock = None
            self.eeg_tcp_connected = False
            self.eeg_reconnect_at = 0.0
            return
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
            self.get_logger().info(f"EEG TCP connected: {self.eeg_server_ip}:{self.eeg_server_port}")
        except Exception as e:
            try:
                sock.close()
            except Exception:
                pass
            self.eeg_tcp_sock = None
            self.eeg_tcp_connected = False
            self.eeg_reconnect_at = now + 1.0
            self.get_logger().warning(f"EEG TCP connect failed: {e}; retry in 1s")

    def _poll_eeg_tcp(self) -> None:
        if getattr(self, "eeg_bypass_debug", False):
            return
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
        if getattr(self, "eeg_bypass_debug", False):
            return False, wall
        payload = int(value).to_bytes(1, byteorder="little", signed=False)
        try:
            self.trigger_send_sock.send(payload)
            return True, wall
        except Exception as e:
            self.get_logger().warning(f"send trigger {value} failed: {e}")
            return False, wall

    def _process_eeg_trigger_sample(self, abs_index: int, trigger_value: int) -> None:
        if trigger_value == self.trial_state.last_trigger_value:
            return

        self.trial_state.last_trigger_value = trigger_value
        if self.trial_state.epoch_mode is None:
            return

        if trigger_value == 1:
            if self.trial_state.epoch_complete:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] ignore trigger=1 after completed epoch at abs={abs_index}"
                )
                return
            if self.trial_state.epoch_start_pending:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] duplicate trigger=1 ignored at abs={abs_index}"
                )
                return
            self.trial_state.stim_start_abs = int(abs_index)
            self.trial_state.epoch_start_pending = True
            return

        if trigger_value == 2:
            if not self.trial_state.epoch_start_pending:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] trigger=2 without open epoch at abs={abs_index}"
                )
                return
            if self.trial_state.epoch_complete:
                self.get_logger().warning(
                    f"[Trial {self.trial_idx}] duplicate trigger=2 ignored at abs={abs_index}"
                )
                return
            self.trial_state.stim_end_abs_inclusive = int(abs_index)
            self.trial_state.epoch_complete = True
            self.trial_state.epoch_start_pending = False
            self._capture_epoch()

    def _capture_epoch(self) -> None:
        self.trial_state.epoch_saved = False
        if (
            self.trial_state.stim_start_abs < 0
            or self.trial_state.stim_end_abs_inclusive < self.trial_state.stim_start_abs
        ):
            self.get_logger().warning(f"[Trial {self.trial_idx}] invalid epoch abs range")
            return
        epoch_end_exclusive = self.trial_state.stim_end_abs_inclusive + 1
        if not self.eeg_ring.has_range(self.trial_state.stim_start_abs, epoch_end_exclusive):
            self.get_logger().warning(
                f"[Trial {self.trial_idx}] EEG range unavailable "
                f"[{self.trial_state.stim_start_abs}, {epoch_end_exclusive})"
            )
            return
        try:
            raw = self.eeg_ring.get_range(
                self.trial_state.stim_start_abs,
                epoch_end_exclusive,
            )
        except Exception as e:
            self.get_logger().warning(f"[Trial {self.trial_idx}] EEG epoch extraction failed: {e}")
            return

        self.trial_state.raw_samples = int(raw.shape[1])
        self.dataset_x.append(raw.astype(np.float32))
        self.dataset_y.append(int(self.current_target_id))
        self.trial_state.epoch_saved = True

        if self.run_mode == "pretrain":
            self.pretrain_meta_writer.writerow(
                [
                    self.trial_idx,
                    self.current_target_id,
                    self.current_target_id,
                    self.trial_state.stim_start_wall,
                    self.trial_state.stim_end_wall,
                    self.trial_state.stim_start_abs,
                    self.trial_state.stim_end_abs_inclusive,
                    self.trial_state.raw_samples,
                    int(self.trial_state.epoch_complete),
                ]
            )
            self.pretrain_meta_csv_file.flush()
        elif self.run_mode == "decode":
            self.decode_meta_writer.writerow(
                [
                    self.trial_idx,
                    self.current_target_id,
                    self.current_target_id,
                    self.trial_state.stim_start_wall,
                    self.trial_state.stim_end_wall,
                    self.trial_state.stim_start_abs,
                    self.trial_state.stim_end_abs_inclusive,
                    self.trial_state.raw_samples,
                    int(self.trial_state.epoch_complete),
                ]
            )
            self.decode_meta_csv_file.flush()

    def _write_mode_trial_row(self) -> None:
        if self.trial_state.trial_record_written:
            return

        if self.run_mode == "pretrain":
            self.pretrain_writer.writerow(
                [
                    self.trial_idx,
                    self.current_target_id,
                    f"{self.current_freq_hz:.3f}",
                    self.trial_state.cue_start_wall,
                    self.trial_state.stim_start_wall,
                    self.trial_state.stim_end_wall,
                    int(self.trial_state.trigger_received),
                    self.trial_state.trigger_wall,
                    int(self.trial_state.stim_start_trigger_sent),
                    self.trial_state.stim_start_trigger_wall,
                    int(self.trial_state.stim_end_trigger_sent),
                    self.trial_state.stim_end_trigger_wall,
                    self.trial_state.stim_enter_abs,
                    self.trial_state.stim_exit_abs,
                    self.trial_state.stim_start_abs,
                    self.trial_state.stim_end_abs_inclusive,
                    self.trial_state.raw_samples,
                    int(self.trial_state.epoch_complete),
                    int(self.trial_state.epoch_saved),
                ]
            )
            self.pretrain_csv_file.flush()
        elif self.run_mode == "decode":
            self.decode_eeg_writer.writerow(
                [
                    self.trial_idx,
                    self.current_target_id,
                    f"{self.current_freq_hz:.3f}",
                    self.trial_state.stim_start_wall,
                    self.trial_state.stim_end_wall,
                    self.trial_state.start_status,
                    int(self.trial_state.stim_start_trigger_sent),
                    self.trial_state.stim_start_trigger_wall,
                    int(self.trial_state.stim_end_trigger_sent),
                    self.trial_state.stim_end_trigger_wall,
                    self.trial_state.stim_enter_abs,
                    self.trial_state.stim_exit_abs,
                    self.trial_state.stim_start_abs,
                    self.trial_state.stim_end_abs_inclusive,
                    self.trial_state.raw_samples,
                    int(self.trial_state.epoch_complete),
                    int(self.trial_state.epoch_saved),
                ]
            )
            self.decode_eeg_csv_file.flush()
        self.trial_state.trial_record_written = True
        self.trial_state.epoch_mode = None

    def _save_mode_dataset(self) -> None:
        if self.run_mode not in ("pretrain", "decode") or self.dataset_saved:
            return
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.dataset_x:
            x_data = np.empty(len(self.dataset_x), dtype=object)
            for i, epoch in enumerate(self.dataset_x):
                x_data[i] = epoch
            y_data = np.asarray(self.dataset_y, dtype=np.int32)
        else:
            x_data = np.asarray([], dtype=object)
            y_data = np.zeros((0,), dtype=np.int32)
        if self.run_mode == "pretrain":
            save_path = os.path.join(self.save_dir, f"ssvep4_pretrain_dataset_{run_stamp}.npy")
        else:
            save_path = os.path.join(self.save_dir, f"ssvep4_decode_dataset_{run_stamp}.npy")
        np.save(save_path, {"x": x_data, "y": y_data}, allow_pickle=True)
        self.dataset_saved = True
        self.get_logger().info(
            f"{self.run_mode} dataset saved: {save_path}, x_shape={x_data.shape}, y_shape={y_data.shape}"
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
        self.trial_state.cue_start_wall = datetime.now().isoformat(timespec="milliseconds")

        self._publish_pretrain_cmd("cue", self.trial_idx, self.current_target_id)
        self.state = NodeState.PRETRAIN_CUEING
        self.state_until = time.monotonic() + self.pretrain_cue_s

    def _enter_pretrain_stim(self) -> None:
        self.trial_state.stim_start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.trial_state.stim_enter_abs = self.eeg_ring.latest_abs_index
        self.trial_state.epoch_mode = "pretrain"
        self.trial_state.stim_start_trigger_sent, self.trial_state.stim_start_trigger_wall = self._send_trigger(1)
        self._publish_pretrain_cmd("stim", self.trial_idx, self.current_target_id)
        self.state = NodeState.PRETRAIN_STIMULATING
        self.state_until = time.monotonic() + self.pretrain_stim_s

    def _enter_pretrain_rest(self) -> None:
        self.trial_state.stim_end_wall = datetime.now().isoformat(timespec="milliseconds")
        self.trial_state.stim_exit_abs = self.eeg_ring.latest_abs_index
        self.trial_state.stim_end_trigger_sent, self.trial_state.stim_end_trigger_wall = self._send_trigger(2)
        self._publish_pretrain_cmd("rest", self.trial_idx, self.current_target_id)

        self.state = NodeState.PRETRAIN_RESTING
        self.state_until = time.monotonic() + self.pretrain_rest_s

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
            if trig is not None and not self.trial_state.trigger_received:
                trial_id, target_id, wall = trig
                if trial_id == self.trial_idx:
                    self.trial_state.trigger_received = True
                    self.trial_state.trigger_wall = wall
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
