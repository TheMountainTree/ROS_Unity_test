#!/usr/bin/env python3
"""Decode mode behavior module for SSVEP communication node (Node4_test).

Node4_test decode now uses runtime FBCCA with unified preprocess/decode APIs.
"""

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

try:
    from .ssvep_runtime_fbcca import SSVEPFBCCARuntime
except ImportError:
    from ssvep_runtime_fbcca import SSVEPFBCCARuntime


class DecodeModule:
    """Mix-in that encapsulates decode configuration and state machine.

    Uses runtime FBCCA decoding for real EEG-based selection.
    """
    # UI slot conventions (0-based): image slots 0/1/2/4/5/6, function slots 3(confirm)/7(undo).
    _UI_DYNAMIC_IMAGE_SLOTS = (0, 1, 2, 4, 5, 6)
    _UI_IMAGE_SLOTS = (0, 1, 2, 4, 5, 6)

    def _consume_reasoner_selection(self) -> int:
        """Resolve reasoner selection source for the current trial.

        Returns:
            UI slot index (0..7) when a selection is ready, or -1 when the
            controller should keep waiting/restart.
        """
        selection = self._consume_cached_mock_selection()
        if selection != -1:
            self.get_logger().info(
                f"[Decode Trial {self.trial_idx}] consume mock_selected_index={selection}"
            )
            return selection

        if getattr(self, "eeg_bypass_debug", False):
            return -1

        return self._perform_eeg_decoding()

    def _load_decode_config(self) -> None:
        self.decode_config = self.config.decode
        self.decode_image_period = self.decode_config.image_publish_period
        self.decode_iti = self.decode_config.inter_trial_interval
        self.decode_trial_duration_s = self.decode_config.trial_duration_s
        self.decode_max_trials = self.decode_config.max_trials
        self.decode_num_images = self.decode_config.num_images
        self.decode_pre_stim_hold_s = self.decode_config.pre_stim_hold_s
        self.decode_capture_wait_timeout_s = self.decode_config.capture_wait_timeout_s
        self.image_h = self.decode_config.image_height
        self.image_w = self.decode_config.image_width
        self.image_paths = list(self.decode_config.image_paths)
        self.image_dir = os.path.expanduser(self.decode_config.image_dir)
        if self.decode_num_images <= 0:
            raise ValueError("decode_num_images must be > 0")
        if self.decode_num_images > 6:
            raise ValueError("decode_num_images must be <= 6 for Unity decode image protocol")

        self.fbcca_runtime_config = self.config.fbcca_runtime

    def _init_decode_state(self) -> None:
        self.decode_dynamic_ui_slots = list(self._UI_DYNAMIC_IMAGE_SLOTS[: self.decode_num_images])
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
        self.base_image_ids = list(range(self.decode_num_images))
        self.current_trial_mapping: List[Tuple[int, int, float]] = []
        self.current_active_ui_image_slots: List[int] = []
        self.publish_idx = 0
        self.next_publish_at = 0.0
        self.waiting_start_trial_id = -1
        self.waiting_start_since = 0.0
        self.decode_hold_until = 0.0
        self._reset_trial_state()

        self.runtime_decoder = SSVEPFBCCARuntime(
            config=self.fbcca_runtime_config,
            frequencies=self.ssvep_frequencies[: self.num_targets],
            logger=self.get_logger(),
        )

    def _init_decode_sockets(self) -> None:
        self.history_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
            ["trial_id", "slot_index", "image_index", "frequency_hz", "trial_prepared_wall_time"]
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

    def _decode_epoch(self, epoch: np.ndarray) -> int:
        """Decode EEG epoch to predicted class label (1-based) via FBCCA."""
        return int(
            self.runtime_decoder.decode_epoch(
                epoch=epoch,
                input_fs=float(self.eeg_fs),
                active_ui_slots=self.current_active_ui_image_slots,
            )
        )

    def _map_predicted_to_slot(self, predicted_label: int) -> int:
        """Map predicted class label to UI slot index.

        Label semantics:
        - labels 1..8 correspond to fixed frequency classes.
        - UI slot index is 0-based and equals label-1.
        - image slots are 0,1,2,4,5,6 (must also be active in current batch).
        - function slots are 3(confirm), 7(undo).

        Args:
            predicted_label: Predicted class label (1-based, 1-8)

        Returns:
            UI slot index (0..7), or -1 if this prediction is invalid/inactive
        """
        if predicted_label < 1 or predicted_label > len(self.ssvep_frequencies):
            self.get_logger().warning(f"Invalid predicted_label={predicted_label}")
            return -1

        ui_slot = predicted_label - 1

        # For image slots, only accept currently active slots in this batch.
        if ui_slot in self._UI_IMAGE_SLOTS:
            if ui_slot not in self.current_active_ui_image_slots:
                self.get_logger().warning(
                    f"Predicted image slot={ui_slot} is inactive for current batch; "
                    f"active_slots={self.current_active_ui_image_slots}"
                )
                return -1
            return ui_slot

        # Function slots (3=confirm, 7=undo) are always valid UI selections.
        return ui_slot

    def _perform_eeg_decoding(self) -> int:
        """Perform real EEG decoding and return predicted UI slot index.

        Returns:
            UI slot index (0..7), or -1 if decoding failed
        """
        if not self.trial_state.epoch_saved:
            self.get_logger().warning("No epoch captured, cannot decode")
            return -1

        if not self.dataset_x:
            self.get_logger().warning("No epochs in dataset, cannot decode")
            return -1

        # Get the last captured epoch
        epoch = self.dataset_x[-1]

        # Decode the epoch
        predicted_label = self._decode_epoch(epoch)
        if predicted_label < 0:
            self.get_logger().error("Decoding failed")
            return -1

        # Map predicted label to slot
        slot_index = self._map_predicted_to_slot(predicted_label)

        self.get_logger().info(
            f"EEG decode: predicted_label={predicted_label} -> "
            f"freq={self.ssvep_frequencies[predicted_label-1]:.2f}Hz -> slot={slot_index}"
        )

        return slot_index

    def _init_decode_mode(self) -> None:
        self._load_decode_config()
        self._init_decode_state()
        self._init_decode_sockets()
        self._init_decode_csv_files()
        self._ensure_eeg_connected(force=True)
        bypass_debug = getattr(self, "eeg_bypass_debug", False)
        eeg_source_text = (
            "bypass_debug(wait_mock_selected_index)"
            if bypass_debug
            else "runtime_fbcca(eeg_tcp+trigger)"
        )
        transport_text = (
            "trigger_send_udp=disabled, eeg_tcp=disabled"
            if bypass_debug
            else (
                f"trigger_send_udp={self.trigger_local_ip}:{self.trigger_local_port}"
                f"->{self.trigger_remote_ip}:{self.trigger_remote_port}, "
                f"eeg_tcp={self.eeg_server_ip}:{self.eeg_server_port}"
            )
        )

        self.get_logger().info(
            "decode mode ready: "
            f"trial_duration={self.decode_trial_duration_s:.2f}s, max_trials={self.decode_max_trials}, "
            f"decode_num_images={self.decode_num_images}, hold={self.decode_pre_stim_hold_s:.2f}s, "
            f"{transport_text}, "
            f"mapping_csv={self.mapping_csv_path}, trials_csv={self.trials_csv_path}, "
            f"decode_eeg_csv={self.decode_eeg_csv_path}, decode_meta={self.decode_meta_csv_path}, "
            f"runtime=FBCCA, epoch_source={eeg_source_text}"
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
        self,
        trial_id: int,
        img_idx_0based: int,
        image_id: int,
        slot_index: int,
        target_id: int,
        freq: float,
    ) -> Image:
        bgr = self.base_images[image_id]
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"trial={trial_id};img={img_idx_0based};image_id={image_id};slot={slot_index};target={target_id};"
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
            self.base_image_ids = list(range(len(self.current_reasoner_group_images)))
            self.current_decode_num_images = max(1, min(6, len(self.current_reasoner_group_images)))
            trial_dynamic_slots = list(self._UI_DYNAMIC_IMAGE_SLOTS[: self.current_decode_num_images])
        else:
            self.current_decode_num_images = self.decode_num_images
            trial_dynamic_slots = self.decode_dynamic_ui_slots

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
        self.current_active_ui_image_slots = []
        for i, slot_id in enumerate(trial_dynamic_slots):
            img_idx = order[i]
            image_id = self.base_image_ids[img_idx]
            freq = self.ssvep_frequencies[slot_id]
            self.current_trial_mapping.append((slot_id, image_id, freq))
            self.current_active_ui_image_slots.append(slot_id)

        for slot_id, image_id, freq in self.current_trial_mapping:
            self.mapping_writer.writerow(
                [self.trial_idx, slot_id, image_id, f"{freq:.3f}", self.trial_state.prepared_wall]
            )
        self.mapping_csv_file.flush()

        self.state = NodeState.DECODE_PUBLISHING
        self.get_logger().info(
            f"[Decode Trial {self.trial_idx}] prepared target={self.current_target_id} "
            f"target_freq={self.current_freq_hz:.3f}Hz, start publishing {self.current_decode_num_images} images; "
            f"active_image_slots={self.current_active_ui_image_slots}"
        )

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
            if getattr(self, "eeg_bypass_debug", False):
                self.get_logger().info(
                    f"[Decode Trial {self.trial_idx}] waiting mock_selected_index, "
                    f"cached={self.pending_mock_selection}"
                )
            else:
                self.get_logger().info(
                    f"[Decode Trial {self.trial_idx}] waiting for EEG decode..."
                )
        else:
            self.state = NodeState.WAITING
            self.state_until = time.monotonic() + max(0.0, self.decode_iti)

    def _enter_decode_stimulating(self, now: float) -> None:
        self.state = NodeState.DECODE_STIMULATING
        self.trial_state.trial_start_mono = now
        self.trial_state.start_wall = datetime.now().isoformat(timespec="milliseconds")
        self.trial_state.start_trial_id = self.trial_idx
        self.trial_state.start_status = "command_sync"
        self.trial_state.stim_start_wall = self.trial_state.start_wall
        self.trial_state.stim_enter_abs = self.eeg_ring.latest_abs_index
        self.trial_state.epoch_mode = "decode"
        self.trial_state.stim_start_trigger_sent, self.trial_state.stim_start_trigger_wall = self._send_trigger(1)
        self._publish_decode_cmd("stim", self.trial_idx, self.current_target_id)

    def _enter_decode_wait_capture(self, now: float) -> None:
        self.trial_state.decode_stop_mono = now
        self.trial_state.stim_end_wall = datetime.now().isoformat(timespec="milliseconds")
        self.trial_state.stim_exit_abs = self.eeg_ring.latest_abs_index
        self.trial_state.stim_end_trigger_sent, self.trial_state.stim_end_trigger_wall = self._send_trigger(2)
        self._publish_decode_stop(self.trial_idx)
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
            selection = self._consume_reasoner_selection()
            if selection == -1:
                if getattr(self, "eeg_bypass_debug", False):
                    return
                self.get_logger().info(
                    "EEG decode returned invalid/empty slot; restart flashing current page."
                )
                self._start_next_decode_trial_with_current_images()
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

            slot_id, image_id, slot_freq = self.current_trial_mapping[self.publish_idx]
            self.image_pub.publish(
                self._to_decode_image(
                    trial_id=self.trial_idx,
                    img_idx_0based=self.publish_idx,
                    image_id=image_id,
                    slot_index=slot_id,
                    target_id=self.current_target_id,
                    freq=slot_freq,
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
            self._enter_decode_stimulating(now)
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
