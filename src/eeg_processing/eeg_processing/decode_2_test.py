#!/usr/bin/env python3
"""Decode mode behavior module for SSVEP communication node (Node4_test).

Extended from decode_1.py with eTRCA decoder integration for real EEG decoding.
"""

import csv
import collections
import glob
import json
import os
import random
import re
import socket
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from sensor_msgs.msg import Image

from .utils import NodeState

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

try:
    from .ssvep_pipeline import SSVEPDecoder
except ImportError:
    from ssvep_pipeline import SSVEPDecoder


class DecodeModule:
    """Mix-in that encapsulates decode configuration and state machine.

    Extended with eTRCA decoder integration for real EEG-based selection.
    """
    # Frequency-class label(1..8) -> Unity/UI slot index(0..7).
    _FREQ_SLOT_TO_UI_SLOT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    _UI_IMAGE_SLOTS = (0, 1, 2, 4, 5, 6)

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

        # Load eTRCA decoder config
        self.etrca_config = self.config.etrca_decoder
        self.decode_filter_enabled = bool(getattr(self.etrca_config, "decode_filter_enabled", True))
        self.decode_bandpass_low_hz = float(getattr(self.etrca_config, "decode_bandpass_low_hz", 6.0))
        self.decode_bandpass_high_hz = float(getattr(self.etrca_config, "decode_bandpass_high_hz", 48.0))
        self.decode_bandpass_order = int(getattr(self.etrca_config, "decode_bandpass_order", 4))
        self.decode_notch_hz = [float(v) for v in getattr(self.etrca_config, "decode_notch_hz", [50.0, 100.0])]
        self.decode_notch_q = float(getattr(self.etrca_config, "decode_notch_q", 35.0))
        self.decode_robust_norm_enabled = bool(
            getattr(self.etrca_config, "decode_robust_norm_enabled", True)
        )
        self.decode_bad_channel_suppress_enabled = bool(
            getattr(self.etrca_config, "decode_bad_channel_suppress_enabled", True)
        )
        self.decode_bad_channel_low_ratio = float(
            getattr(self.etrca_config, "decode_bad_channel_low_ratio", 0.2)
        )
        self.decode_bad_channel_high_ratio = float(
            getattr(self.etrca_config, "decode_bad_channel_high_ratio", 10.0)
        )
        self.decode_bad_channel_suppress_factor = float(
            getattr(self.etrca_config, "decode_bad_channel_suppress_factor", 0.0)
        )
        # Candidate decode window derived from pretrain stim duration.
        pretrain_cfg = getattr(self.config, "pretrain", None)
        if pretrain_cfg is not None and float(pretrain_cfg.stim_duration_s) > 0.0:
            self.decode_model_window_s = float(pretrain_cfg.stim_duration_s)
        else:
            self.decode_model_window_s = float(self.decode_trial_duration_s)

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
        self.current_active_ui_image_slots: List[int] = []
        self.publish_idx = 0
        self.next_publish_at = 0.0
        self.waiting_start_trial_id = -1
        self.waiting_start_since = 0.0
        self.decode_hold_until = 0.0
        self._reset_trial_state()

        # Initialize decoder state
        self.decoder = None
        self.model_loaded = False
        self.decode_success_samples = 0
        self.decode_model_sample_candidates: List[int] = []
        self.decode_length_search_done = False
        self._decode_bandpass_sos = None
        self._decode_notch_ba: List[Tuple[np.ndarray, np.ndarray]] = []
        self.decode_profile_bad_channels: List[int] = []

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

    def _load_decoder_model(self) -> None:
        """Load pre-trained eTRCA model for decoding.

        Raises:
            RuntimeError: If model file not found (strict mode - must run pretrain first)
        """
        model_path = self.etrca_config.model_path

        if not os.path.isfile(model_path):
            self.get_logger().error(
                f"No pre-trained model found at {model_path}. "
                "Please run pretrain mode first to train the model."
            )
            raise RuntimeError(f"Model file not found: {model_path}")

        try:
            self.decoder = SSVEPDecoder.from_file(model_path)
            self.model_loaded = True
            self.decode_model_sample_candidates = self._infer_model_sample_candidates()
            self._load_channel_profile_sidecar(model_path)
            self.get_logger().info(f"Loaded eTRCA decoder from {model_path}")
            if self.decode_model_sample_candidates:
                self.get_logger().info(
                    f"Decoder model sample candidates inferred: {self.decode_model_sample_candidates}"
                )
        except Exception as e:
            self.get_logger().error(f"Failed to load decoder model: {e}")
            raise RuntimeError(f"Failed to load decoder model: {e}")

    def _load_channel_profile_sidecar(self, model_path: str) -> None:
        """Load bad-channel profile exported by pretrain."""
        self.decode_profile_bad_channels = []
        sidecar = model_path + ".channel_profile.json"
        if not os.path.isfile(sidecar):
            return
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                payload = json.load(f)
            indices = payload.get("bad_channels", [])
            if isinstance(indices, list):
                self.decode_profile_bad_channels = sorted(
                    {
                        int(v)
                        for v in indices
                        if isinstance(v, (int, float)) and int(v) >= 0
                    }
                )
            if self.decode_profile_bad_channels:
                self.get_logger().info(
                    f"Loaded bad-channel profile: {self.decode_profile_bad_channels}"
                )
        except Exception as e:
            self.get_logger().warning(f"Failed to load channel profile sidecar: {e}")

    def _infer_model_sample_candidates(self) -> List[int]:
        """Try to infer sample-length candidates from estimator internals."""
        candidates: List[int] = []
        estimator = getattr(self.decoder, "_estimator", None)
        if estimator is None:
            return candidates

        def _collect(value) -> None:
            if isinstance(value, np.ndarray):
                if value.ndim >= 2:
                    sample_len = int(value.shape[-1])
                    if 16 <= sample_len <= 8192 and sample_len not in candidates:
                        candidates.append(sample_len)
                return
            if isinstance(value, (list, tuple)):
                for item in value[:64]:
                    _collect(item)
                return
            if isinstance(value, dict):
                for item in list(value.values())[:64]:
                    _collect(item)

        for value in vars(estimator).values():
            _collect(value)
            if len(candidates) >= 16:
                break

        candidates.sort()
        return candidates

    def _resample_epoch_for_decode(self, epoch: np.ndarray) -> np.ndarray:
        """Resample single epoch from acquisition rate to model rate.

        Args:
            epoch: EEG data with shape (n_channels, n_samples)

        Returns:
            Resampled epoch with shape (n_channels, n_samples_resampled)
        """
        orig_fs = self.eeg_fs
        target_fs = self.etrca_config.srate

        n_samples = int(epoch.shape[1] * target_fs / orig_fs)
        return signal.resample(epoch, n_samples, axis=1)

    def _build_decode_filters(self, fs: float) -> None:
        """Build bandpass and notch filters for decode preprocessing."""
        self._decode_bandpass_sos = None
        self._decode_notch_ba = []
        if not self.decode_filter_enabled or fs <= 0:
            return

        nyquist = fs * 0.5
        low = max(0.1, self.decode_bandpass_low_hz)
        high = min(self.decode_bandpass_high_hz, nyquist - 0.5)
        if high > low:
            self._decode_bandpass_sos = signal.butter(
                self.decode_bandpass_order,
                [low, high],
                btype="bandpass",
                fs=fs,
                output="sos",
            )

        for f0 in self.decode_notch_hz:
            if f0 <= 0.0 or f0 >= nyquist:
                continue
            b, a = signal.iirnotch(w0=f0, Q=self.decode_notch_q, fs=fs)
            self._decode_notch_ba.append((b, a))

        self.get_logger().info(
            "decode filters ready: "
            f"enabled={self.decode_filter_enabled}, bandpass={low:.1f}-{high:.1f}Hz, "
            f"notch={[round(v, 2) for v in self.decode_notch_hz if 0.0 < v < nyquist]}"
        )

    def _apply_decode_filters(self, epoch: np.ndarray, fs: float) -> np.ndarray:
        """Apply basic decode filters to suppress mains and display noise."""
        if not self.decode_filter_enabled or epoch.ndim != 2:
            return epoch
        if self._decode_bandpass_sos is None and not self._decode_notch_ba:
            self._build_decode_filters(fs)

        out = epoch.astype(np.float64, copy=False)
        try:
            if self._decode_bandpass_sos is not None:
                out = signal.sosfiltfilt(self._decode_bandpass_sos, out, axis=1)
            for b, a in self._decode_notch_ba:
                out = signal.filtfilt(b, a, out, axis=1)
        except Exception as e:
            self.get_logger().warning(f"decode filtering failed, fallback to raw epoch: {e}")
            return epoch
        return out.astype(np.float32, copy=False)

    def _apply_decode_channel_suppression(self, epoch: np.ndarray) -> np.ndarray:
        """Robust channel normalization + static/dynamic bad-channel suppression."""
        if epoch.ndim != 2:
            return epoch

        out = epoch.astype(np.float64, copy=True)
        out -= np.median(out, axis=1, keepdims=True)
        mad = np.median(np.abs(out), axis=1)
        scale = 1.4826 * mad
        valid = scale > 1e-12
        global_scale = float(np.median(scale[valid])) if np.any(valid) else 1.0
        floor = max(1e-6, global_scale * 0.05)
        safe = np.maximum(scale, floor)

        if self.decode_robust_norm_enabled:
            out = out / safe[:, np.newaxis]

        if not self.decode_bad_channel_suppress_enabled:
            return out.astype(np.float32, copy=False)

        dynamic_bad = (
            (scale < global_scale * self.decode_bad_channel_low_ratio)
            | (scale > global_scale * self.decode_bad_channel_high_ratio)
        )
        bad_idx = set(np.where(dynamic_bad)[0].astype(int).tolist())
        bad_idx.update(
            idx for idx in self.decode_profile_bad_channels if 0 <= idx < out.shape[0]
        )
        if bad_idx:
            factor = min(1.0, max(0.0, float(self.decode_bad_channel_suppress_factor)))
            for idx in sorted(bad_idx):
                out[idx, :] *= factor
            self.get_logger().warning(
                f"decode bad-channel suppression: bad={sorted(bad_idx)}, factor={factor}"
            )

        return out.astype(np.float32, copy=False)

    def _fit_epoch_to_samples(self, epoch: np.ndarray, expected_samples: int) -> np.ndarray:
        """Return epoch truncated/padded to expected sample length."""
        if expected_samples <= 0:
            return epoch

        current_samples = int(epoch.shape[1])
        if current_samples == expected_samples:
            return epoch

        if current_samples > expected_samples:
            return epoch[:, :expected_samples]

        pad_width = expected_samples - current_samples
        return np.pad(epoch, ((0, 0), (0, pad_width)), mode="constant")

    def _predict_label_for_samples(
        self,
        epoch: np.ndarray,
        expected_samples: int,
    ) -> int:
        """Predict one label using a fixed sample length."""
        epoch_try = self._fit_epoch_to_samples(epoch, expected_samples)
        if epoch_try.ndim == 2:
            epoch_try = epoch_try[np.newaxis, :, :]
        predicted = self.decoder.decode(epoch_try)
        return int(predicted[0])

    def _decode_epoch_with_window_voting(
        self,
        epoch: np.ndarray,
        expected_samples: int,
    ) -> int:
        """Decode variable-length epochs by fixed-window voting."""
        if expected_samples <= 0:
            return -1

        current_samples = int(epoch.shape[1])
        if current_samples <= expected_samples:
            return self._predict_label_for_samples(epoch, expected_samples)

        n_full = current_samples // expected_samples
        labels: List[int] = []
        for i in range(n_full):
            start = i * expected_samples
            end = start + expected_samples
            try:
                labels.append(self._predict_label_for_samples(epoch[:, start:end], expected_samples))
            except Exception:
                continue

        # If no full window succeeded, fall back to one fixed-length attempt.
        if not labels:
            return self._predict_label_for_samples(epoch, expected_samples)

        voted = collections.Counter(labels).most_common(1)[0][0]
        return int(voted)

    def _search_working_decode_length(self, epoch: np.ndarray, center: int) -> int:
        """Best-effort discovery of a model-compatible sample length."""
        if center <= 0:
            return -1
        current_samples = int(epoch.shape[1])
        low = max(128, center - 160)
        high = min(current_samples, center + 160)
        if high < low:
            high = low

        search_order: List[int] = []
        for delta in range(0, max(center - low, high - center) + 1):
            left = center - delta
            right = center + delta
            if low <= left <= high and left not in search_order:
                search_order.append(left)
            if low <= right <= high and right not in search_order:
                search_order.append(right)

        for expected_samples in search_order:
            try:
                label = self._decode_epoch_with_window_voting(epoch, expected_samples)
                if label > 0:
                    self.get_logger().info(
                        f"Discovered decoder-compatible sample length: {expected_samples}"
                    )
                    return expected_samples
            except Exception:
                continue
        return -1

    def _decode_epoch(self, epoch: np.ndarray) -> int:
        """Decode EEG epoch to predicted class label.

        Args:
            epoch: EEG data with shape (n_channels, n_samples)

        Returns:
            Predicted class label (1-based), or -1 if decoding failed
        """
        if not self.model_loaded or self.decoder is None:
            self.get_logger().error("No trained model available for decoding")
            return -1

        # Resample if needed (EEG at 1000Hz, model at 256Hz)
        if self.eeg_fs != self.etrca_config.srate:
            epoch = self._resample_epoch_for_decode(epoch)

        # Basic preprocessing: bandpass + notch(50/100Hz) to suppress mains/monitor noise.
        epoch = self._apply_decode_filters(epoch, float(self.etrca_config.srate))
        epoch = self._apply_decode_channel_suppression(epoch)

        # Build sample-length candidates:
        # 1) last successful length, 2) current epoch length, 3) pretrain-window length.
        current_samples = int(epoch.shape[1])
        pretrain_samples = int(round(self.decode_model_window_s * self.etrca_config.srate))
        candidate_lengths: List[int] = []
        for value in (
            self.decode_success_samples,
            current_samples,
            pretrain_samples,
            *self.decode_model_sample_candidates,
        ):
            if value > 0 and value not in candidate_lengths:
                candidate_lengths.append(value)

        last_error: Optional[Exception] = None
        for expected_samples in candidate_lengths:
            try:
                predicted = self._decode_epoch_with_window_voting(epoch, expected_samples)
                if expected_samples != current_samples:
                    self.get_logger().warning(
                        f"Decode epoch length mismatch: got {current_samples}, "
                        f"using {expected_samples} for decoder input."
                    )
                self.decode_success_samples = expected_samples
                return int(predicted)
            except Exception as e:
                last_error = e
                continue

        # One-time adaptive search around pretrain-derived center.
        if not self.decode_length_search_done:
            self.decode_length_search_done = True
            discovered = self._search_working_decode_length(epoch, pretrain_samples)
            if discovered > 0:
                try:
                    predicted = self._decode_epoch_with_window_voting(epoch, discovered)
                    self.decode_success_samples = discovered
                    self.get_logger().warning(
                        f"Decode epoch length mismatch: got {current_samples}, "
                        f"auto-discovered {discovered} for decoder input."
                    )
                    return int(predicted)
                except Exception as e:
                    last_error = e

        self.get_logger().error(f"Decoding failed: {last_error}")
        return -1

    def _map_predicted_to_slot(self, predicted_label: int) -> int:
        """Map predicted class label to UI slot index.

        Label semantics:
        - labels 1..8 correspond to fixed frequency classes / fixed UI slots 0..7.
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

        ui_slot = self._FREQ_SLOT_TO_UI_SLOT.get(predicted_label, -1)
        if ui_slot < 0:
            self.get_logger().warning(f"No UI slot mapping for predicted_label={predicted_label}")
            return -1

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

        if not self.model_loaded:
            self.get_logger().error("Model not loaded, cannot perform decoding")
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

        # Load decoder model (will raise RuntimeError if not found)
        self._load_decoder_model()

        self.get_logger().info(
            "decode mode ready: "
            f"trial_duration={self.decode_trial_duration_s:.2f}s, max_trials={self.decode_max_trials}, "
            f"decode_num_images={self.decode_num_images}, hold={self.decode_pre_stim_hold_s:.2f}s, "
            f"decode_ack_udp={self.decode_start_bind_ip}:{self.decode_start_port}, "
            f"trigger_send_udp={self.trigger_local_ip}:{self.trigger_local_port}"
            f"->{self.trigger_remote_ip}:{self.trigger_remote_port}, "
            f"eeg_tcp={self.eeg_server_ip}:{self.eeg_server_port}, "
            f"mapping_csv={self.mapping_csv_path}, trials_csv={self.trials_csv_path}, "
            f"decode_eeg_csv={self.decode_eeg_csv_path}, decode_meta={self.decode_meta_csv_path}, "
            f"model_loaded={self.model_loaded}"
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
        self.current_active_ui_image_slots = []
        for i, slot_id in enumerate(trial_dynamic_slots):
            img_idx = order[i]
            image_id = self.base_image_ids[img_idx]
            freq = self.ssvep_frequencies[slot_id - 1]
            self.current_trial_mapping.append((slot_id, image_id, freq))
            ui_slot = self._FREQ_SLOT_TO_UI_SLOT.get(slot_id, -1)
            if ui_slot >= 0:
                self.current_active_ui_image_slots.append(ui_slot)

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
                f"[Decode Trial {self.trial_idx}] waiting for EEG decode..."
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
            # Use real EEG decoding instead of mock selection
            selection = self._perform_eeg_decoding()
            if selection == -1:
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

            _, image_id, slot_freq = self.current_trial_mapping[self.publish_idx]
            self.image_pub.publish(
                self._to_decode_image(
                    trial_id=self.trial_idx,
                    img_idx_1based=self.publish_idx + 1,
                    image_id=image_id,
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
