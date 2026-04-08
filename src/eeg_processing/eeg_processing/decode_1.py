#!/usr/bin/env python3
"""Decode mode behavior module for SSVEP communication node."""

import csv
import glob
import os
import random
import re
import socket
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sensor_msgs.msg import Image

from .utils import NodeState

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class DecodeModule:
    """Mix-in that encapsulates decode configuration and state machine."""

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

    def _init_decode_state(self) -> None:
        base_dynamic_slots = [1, 2, 3, 5, 6, 7]
        self.decode_dynamic_target_ids = base_dynamic_slots[: self.decode_num_images]
        self.current_decode_num_images = self.decode_num_images
        self.history_stack: List[Dict[str, object]] = []
        self.reasoner_action_stack: List[Dict[str, object]] = []
        self.next_history_id = 0
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

    def _init_decode_sockets(self) -> None:
        self.history_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._init_decode_ack_receiver()
        self._init_trigger_sender()
        self._init_eeg_streaming(buffer_seconds=max(20.0, self.decode_trial_duration_s * 8.0))

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

    @staticmethod
    def _natural_sort_key(path: str):
        """Natural sort by filename number segments, such as img2 < img10."""
        basename = os.path.basename(path)
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", basename)]

    def _read_image_bgr(self, path: str) -> Optional[np.ndarray]:
        """Read image and convert to vertically flipped BGR for Unity."""
        if not os.path.isfile(path) or PILImage is None:
            return None
        try:
            img = PILImage.open(path).convert("RGB").resize((self.image_w, self.image_h))
            rgb = np.asarray(img, dtype=np.uint8)
            bgr = rgb[:, :, ::-1].copy()
            bgr = np.flipud(bgr).copy()
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
            img[y0: y0 + box, x0: x0 + box, :] = 255
            out.append(img)
        return out

    def _load_or_generate_images(self, n: int) -> List[np.ndarray]:
        candidates = []
        if self.image_paths:
            candidates = sorted([p for p in self.image_paths if p], key=self._natural_sort_key)
        else:
            for pat in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                candidates.extend(glob.glob(os.path.join(self.image_dir, pat)))
            candidates = sorted(candidates, key=self._natural_sort_key)

        self.get_logger().info(
            f"Image candidates ({len(candidates)}): {[os.path.basename(p) for p in candidates[:n]]}"
        )

        frames = []
        for path in candidates:
            frame = self._read_image_bgr(path)
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
        self.decode_command_pub.publish(msg)

    def _publish_decode_batch_cmd(self, cmd: str, trial_id: int, target_id: int, count: int) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"cmd={cmd};trial={trial_id};target={target_id};count={count};"
            f"dur={self.decode_trial_duration_s:.3f}"
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
            self.current_decode_num_images = max(1, min(6, len(self.current_reasoner_group_images)))
            trial_dynamic_slots = [1, 2, 3, 5, 6, 7][: self.current_decode_num_images]
        else:
            self.current_decode_num_images = self.decode_num_images
            trial_dynamic_slots = self.decode_dynamic_target_ids

        if (
            (not self.reasoner_mode_enabled)
            and self.decode_max_trials > 0
            and self.trial_idx >= self.decode_max_trials
        ):
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
        self.trial_state.prepared_wall = datetime.now().isoformat(timespec="milliseconds")

        self.current_target_id = random.randint(1, self.num_targets)
        self.current_freq_hz = self.ssvep_frequencies[self.current_target_id - 1]

        if self.reasoner_mode_enabled:
            order = list(range(self.current_decode_num_images))
        else:
            order = list(range(self.current_decode_num_images))
            random.shuffle(order)
        self.current_trial_mapping = []
        for i, slot_id in enumerate(trial_dynamic_slots):
            img_idx = order[i]
            image_id = self.base_image_ids[img_idx]
            freq = self.ssvep_frequencies[slot_id - 1]
            self.current_trial_mapping.append((slot_id, image_id, freq))

        for slot_id, image_id, freq in self.current_trial_mapping:
            self.mapping_writer.writerow(
                [self.trial_idx, slot_id, image_id, f"{freq:.3f}", self.trial_state.prepared_wall]
            )
        self.mapping_csv_file.flush()

        self.state = NodeState.DECODE_PUBLISHING
        self.get_logger().info(
            f"[Decode Trial {self.trial_idx}] prepared target={self.current_target_id} "
            f"target_freq={self.current_freq_hz:.3f}Hz, start publishing {self.current_decode_num_images} images"
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
        end_wall = self.trial_state.stim_end_wall or datetime.now().isoformat(timespec="milliseconds")
        if self.trial_state.stim_end_wall == "":
            self.trial_state.stim_end_wall = end_wall
        actual_s = 0.0
        if self.trial_state.trial_start_mono > 0.0:
            stop_mono = (
                self.trial_state.decode_stop_mono
                if self.trial_state.decode_stop_mono > 0.0
                else time.monotonic()
            )
            actual_s = max(0.0, stop_mono - self.trial_state.trial_start_mono)

        self.trials_writer.writerow(
            [
                self.trial_idx,
                self.current_target_id,
                f"{self.current_freq_hz:.3f}",
                self.trial_state.prepared_wall,
                self.trial_state.start_wall,
                end_wall,
                f"{self.decode_trial_duration_s:.3f}",
                f"{actual_s:.3f}",
                self.trial_state.start_trial_id if self.trial_state.start_trial_id > 0 else "",
                self.trial_state.start_status,
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
        self.trial_state.trial_start_mono = now
        self.trial_state.start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.trial_state.start_trial_id = start_trial_id
        self.trial_state.start_status = start_status
        self.trial_state.stim_start_wall = self.trial_state.start_wall
        self.trial_state.stim_enter_abs = self.eeg_ring.latest_abs_index
        self.trial_state.epoch_mode = "decode"
        self.trial_state.stim_start_trigger_sent, self.trial_state.stim_start_trigger_wall = self._send_trigger(1)

    def _enter_decode_wait_capture(self, now: float) -> None:
        self._publish_decode_stop(self.trial_idx)
        self.trial_state.decode_stop_mono = now
        self.trial_state.stim_end_wall = datetime.now().isoformat(timespec="milliseconds")
        self.trial_state.stim_exit_abs = self.eeg_ring.latest_abs_index
        self.trial_state.stim_end_trigger_sent, self.trial_state.stim_end_trigger_wall = self._send_trigger(2)
        self.state = NodeState.DECODE_WAIT_CAPTURE
        self.state_until = now + max(0.0, self.decode_capture_wait_timeout_s)

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

            if self.publish_idx == 0:
                self._publish_decode_batch_cmd(
                    "batch_start",
                    self.trial_idx,
                    self.current_target_id,
                    self.current_decode_num_images,
                )

            _, image_id, _ = self.current_trial_mapping[self.publish_idx]
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
            if self.publish_idx >= self.current_decode_num_images:
                self._publish_decode_batch_cmd(
                    "batch_end",
                    self.trial_idx,
                    self.current_target_id,
                    self.current_decode_num_images,
                )
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
            if now - self.trial_state.trial_start_mono >= self.decode_trial_duration_s:
                self._enter_decode_wait_capture(now)
            return

        if self.state == NodeState.DECODE_WAIT_CAPTURE:
            if self.trial_state.epoch_complete or now >= self.state_until:
                self._write_mode_trial_row()
                self._finalize_decode_trial()
            return
