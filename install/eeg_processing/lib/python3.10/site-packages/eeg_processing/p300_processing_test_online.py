#!/usr/bin/env python3
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Keep MNE cache/config writes inside writable sandbox paths.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_DATA", "/tmp/mne_data")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

try:
    # In newer brainda versions, the dataset name is updated.
    from brainda.datasets import Cattan2019 as Cattan_P300
    from brainda.paradigms.p300 import P300
except ImportError:
    raise ImportError("Could not import brainda. Please run 'pip install brainda'.")


BASE_EEG_CHANNELS = [
    "FP1",
    "FP2",
    "FC5",
    "FZ",
    "FC6",
    "T7",
    "CZ",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "OZ",
    "O2",
]
TRUE_EOG_CHANNELS = ["VEOG", "HEOG", "EOG1", "EOG2"]


def normalize_trial_labels(yi):
    yi = np.asarray(yi)
    if yi.ndim == 1:
        return yi.astype(int)
    if yi.ndim == 2 and yi.shape[1] == 1:
        return yi[:, 0].astype(int)
    if yi.ndim == 2 and yi.shape[1] > 1:
        return np.argmax(yi, axis=1).astype(int)
    return yi.reshape(-1).astype(int)


def trial_to_features(xi, downsample=2):
    return xi[:, :, ::downsample].reshape(xi.shape[0], -1)


def stimulus_to_binary_labels(stimulus_ids, code):
    code = np.asarray(code).astype(int).reshape(-1)
    stimulus_ids = normalize_trial_labels(stimulus_ids)
    return np.array([1 if code[int(stim_id) - 1] == 2 else 0 for stim_id in stimulus_ids])


def get_target_classes_from_code(code):
    code = np.asarray(code).astype(int).reshape(-1)
    return [i + 1 for i, v in enumerate(code) if v == 2]


def build_binary_train_set(x_trials, y_trials, meta, trial_indices, downsample=2):
    features = []
    labels = []
    for idx in trial_indices:
        xi = x_trials[idx]
        yi_stim = normalize_trial_labels(y_trials[idx])
        yi_bin = stimulus_to_binary_labels(yi_stim, meta.iloc[idx]["code"])
        features.append(trial_to_features(xi, downsample=downsample))
        labels.append(yi_bin)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def remove_artifacts_by_amplitude_threshold(
    x_trials, y_trials, ptp_threshold=120.0, rms_z_threshold=4.0
):
    x_clean_trials = []
    y_clean_trials = []
    for xi, yi in zip(x_trials, y_trials):
        ptp_per_channel = np.ptp(xi, axis=-1)
        max_ptp_per_epoch = np.max(ptp_per_channel, axis=1)
        rms_per_epoch = np.sqrt(np.mean(xi**2, axis=(1, 2)))
        rms_std = np.std(rms_per_epoch)
        if rms_std < 1e-12:
            rms_z = np.zeros_like(rms_per_epoch)
        else:
            rms_z = (rms_per_epoch - np.mean(rms_per_epoch)) / rms_std
        keep_mask = (max_ptp_per_epoch < ptp_threshold) & (np.abs(rms_z) < rms_z_threshold)
        x_clean_trials.append(xi[keep_mask])
        y_clean_trials.append(yi[keep_mask])
    return x_clean_trials, y_clean_trials


def remove_artifacts_by_eog_regression(x_trials, channel_names):
    eog_indices = [i for i, ch in enumerate(channel_names) if ch in TRUE_EOG_CHANNELS]
    if not eog_indices:
        return None, None
    eeg_indices = [i for i, ch in enumerate(channel_names) if ch not in TRUE_EOG_CHANNELS]
    cleaned_trials = []
    for xi in x_trials:
        xi_clean = xi.copy()
        for epoch_idx in range(xi.shape[0]):
            eog = xi[epoch_idx, eog_indices, :].T
            design = np.column_stack([np.ones(eog.shape[0]), eog])
            for ch_idx in eeg_indices:
                eeg_signal = xi[epoch_idx, ch_idx, :]
                beta, *_ = np.linalg.lstsq(design, eeg_signal, rcond=None)
                artifact = eog @ beta[1:]
                xi_clean[epoch_idx, ch_idx, :] = eeg_signal - artifact
        cleaned_trials.append(xi_clean[:, eeg_indices, :])
    cleaned_channels = [channel_names[i] for i in eeg_indices]
    return cleaned_trials, cleaned_channels


def load_data_with_eog_fallback(subject_id=1):
    dataset = Cattan_P300()
    channels_with_eog = BASE_EEG_CHANNELS + TRUE_EOG_CHANNELS
    try:
        paradigm = P300(channels=channels_with_eog)
        x, y, meta = paradigm.get_data(dataset, subjects=[subject_id], verbose=False)
        channel_names = channels_with_eog
    except Exception:
        paradigm = P300(channels=BASE_EEG_CHANNELS)
        x, y, meta = paradigm.get_data(dataset, subjects=[subject_id], verbose=False)
        channel_names = BASE_EEG_CHANNELS
    return x, y, meta, channel_names


def train_offline_calibration_model(
    subject_id=1,
    downsample=2,
    test_size=0.3,
    random_state=42,
    ptp_threshold=120.0,
    rms_z_threshold=4.0,
):
    x, y, meta, channel_names = load_data_with_eog_fallback(subject_id=subject_id)

    x_artifact_free, cleaned_channels = remove_artifacts_by_eog_regression(x, channel_names)
    if x_artifact_free is None:
        x_artifact_free, y_clean = remove_artifacts_by_amplitude_threshold(
            x, y, ptp_threshold=ptp_threshold, rms_z_threshold=rms_z_threshold
        )
        cleaned_channels = channel_names
        removal_method = "Amplitude threshold"
    else:
        y_clean = y
        removal_method = "EOG regression"

    trial_ids = np.arange(len(x_artifact_free))
    train_ids, _ = train_test_split(
        trial_ids, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_train, y_train = build_binary_train_set(
        x_artifact_free, y_clean, meta, train_ids, downsample=downsample
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="multinomial"),
    )
    clf.fit(X_train, y_train)
    classes = clf.named_steps["logisticregression"].classes_
    return clf, classes, removal_method, cleaned_channels


@dataclass
class OnlinePreprocessConfig:
    fs: float = 100.0
    low: float = 0.5
    high: float = 30.0
    order: int = 4
    baseline_samples: int = 20


@dataclass
class EpochWindowConfig:
    pre_seconds: float = 0.2
    post_seconds: float = 0.8
    buffer_seconds: float = 12.0
    trigger_latency_correction_s: float = 0.0


@dataclass
class UDPTriggerConfig:
    bind_ip: str = "0.0.0.0"
    bind_port: int = 9999
    timeout_s: float = 0.05
    valid_trigger_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)


@dataclass
class OnlineRunConfig:
    downsample: int = 2
    num_stimuli: int = 6
    expected_triggers_per_trial: int = 18
    topk: int = 2
    min_confidence_gap: float = 0.0


class OnlineEpochPreprocessor:
    """
    Online preprocessing per epoch:
    - causal filter only
    - baseline correction only with epoch start samples
    """

    def __init__(self, n_channels: int, cfg: OnlinePreprocessConfig):
        self.cfg = cfg
        self.sos = butter(
            cfg.order, [cfg.low, cfg.high], fs=cfg.fs, btype="bandpass", output="sos"
        )
        self.zi = np.array(
            [lfilter_zi(section[:3], section[3:]) for section in self.sos], dtype=float
        )
        self.zi = np.repeat(self.zi[:, :, None], n_channels, axis=2)

    def process_epoch(self, epoch_ch_time: np.ndarray) -> np.ndarray:
        x = np.asarray(epoch_ch_time, dtype=float)
        if x.ndim != 2:
            raise ValueError("epoch_ch_time must have shape (n_channels, n_times)")

        y = np.empty_like(x)
        for ch in range(x.shape[0]):
            signal = x[ch]
            for sec_idx, section in enumerate(self.sos):
                b = section[:3]
                a = section[3:]
                zi_ch = self.zi[sec_idx, :, ch]
                signal, zi_new = lfilter(b, a, signal, zi=zi_ch)
                self.zi[sec_idx, :, ch] = zi_new
            y[ch] = signal

        b = min(self.cfg.baseline_samples, y.shape[1])
        baseline = np.mean(y[:, :b], axis=1, keepdims=True)
        return y - baseline


class CircularEEGBuffer:
    """Ring buffer for continuous EEG samples with absolute sample indexing."""

    def __init__(self, n_channels: int, fs: float, buffer_seconds: float):
        self.n_channels = n_channels
        self.fs = fs
        self.capacity = int(round(fs * buffer_seconds))
        if self.capacity <= 0:
            raise ValueError("buffer capacity must be > 0")
        self.data = np.zeros((n_channels, self.capacity), dtype=float)
        self.write_ptr = 0
        self.total_written = 0
        self.lock = threading.Lock()

    @property
    def latest_abs_index(self) -> int:
        return self.total_written

    @property
    def oldest_abs_index(self) -> int:
        return max(0, self.total_written - self.capacity)

    def append(self, chunk_ch_time: np.ndarray) -> Tuple[int, int]:
        x = np.asarray(chunk_ch_time, dtype=float)
        if x.ndim != 2 or x.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk must have shape (n_channels, n_times), expected ({self.n_channels}, T)"
            )
        n = x.shape[1]
        if n == 0:
            return self.total_written, self.total_written

        if n >= self.capacity:
            x = x[:, -self.capacity :]
            n = self.capacity

        with self.lock:
            start_abs = self.total_written
            first = min(n, self.capacity - self.write_ptr)
            second = n - first
            self.data[:, self.write_ptr : self.write_ptr + first] = x[:, :first]
            if second > 0:
                self.data[:, :second] = x[:, first:]
            self.write_ptr = (self.write_ptr + n) % self.capacity
            self.total_written += n
            return start_abs, self.total_written

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


class TriggeredEpochExtractor:
    """Create epochs from continuous stream using UDP trigger IDs."""

    def __init__(self, eeg_buffer: CircularEEGBuffer, cfg: EpochWindowConfig):
        self.buffer = eeg_buffer
        self.cfg = cfg
        self.pre_samples = int(round(cfg.pre_seconds * eeg_buffer.fs))
        self.post_samples = int(round(cfg.post_seconds * eeg_buffer.fs))
        self.latency_samples = int(round(cfg.trigger_latency_correction_s * eeg_buffer.fs))
        self.pending: List[Tuple[int, int]] = []  # (trigger_id, trigger_abs_idx)

    def add_trigger_now(self, trigger_id: int) -> None:
        trigger_abs = self.buffer.latest_abs_index + self.latency_samples
        self.pending.append((trigger_id, trigger_abs))

    def pop_ready_epochs(self) -> List[Tuple[int, np.ndarray]]:
        ready: List[Tuple[int, np.ndarray]] = []
        next_pending: List[Tuple[int, int]] = []

        oldest = self.buffer.oldest_abs_index
        latest = self.buffer.latest_abs_index

        for trigger_id, trigger_abs in self.pending:
            start_abs = trigger_abs - self.pre_samples
            end_abs = trigger_abs + self.post_samples

            if end_abs > latest:
                next_pending.append((trigger_id, trigger_abs))
                continue
            if start_abs < oldest:
                continue
            if not self.buffer.has_range(start_abs, end_abs):
                continue

            epoch = self.buffer.get_range(start_abs, end_abs)
            ready.append((trigger_id, epoch))

        self.pending = next_pending
        return ready


class OnlineP300Decoder:
    def __init__(
        self,
        classifier,
        classes: np.ndarray,
        preprocessor: OnlineEpochPreprocessor,
        run_cfg: OnlineRunConfig,
    ):
        self.classifier = classifier
        self.classes = classes
        self.preprocessor = preprocessor
        self.run_cfg = run_cfg
        self.pos_idx = int(np.where(classes == 1)[0][0])
        self.score = {i: 0.0 for i in range(1, run_cfg.num_stimuli + 1)}
        self.epoch_count = 0

    def update(self, epoch_ch_time: np.ndarray, stimulus_id: int) -> float:
        if stimulus_id < 1 or stimulus_id > self.run_cfg.num_stimuli:
            raise ValueError(
                f"stimulus_id must be in [1, {self.run_cfg.num_stimuli}], got {stimulus_id}"
            )
        x = self.preprocessor.process_epoch(epoch_ch_time)
        feat = x[:, :: self.run_cfg.downsample].reshape(1, -1)
        p = self.classifier.predict_proba(feat)[0]
        p_target = float(p[self.pos_idx])
        self.score[stimulus_id] += p_target
        self.epoch_count += 1
        return p_target

    def predict_topk(self, k: int) -> List[int]:
        return sorted(self.score.keys(), key=lambda sid: self.score[sid], reverse=True)[:k]

    def confidence_gap(self) -> float:
        ordered = sorted(self.score.values(), reverse=True)
        if len(ordered) < 2:
            return 0.0
        return float(ordered[0] - ordered[1])

    def finalize_trial(self) -> Dict:
        pred = self.predict_topk(self.run_cfg.topk)
        gap = self.confidence_gap()
        accepted = gap >= self.run_cfg.min_confidence_gap
        result = {
            "predicted_target_class": pred if accepted else [],
            "trial_score": dict(self.score),
            "used_epochs": int(self.epoch_count),
            "confidence_gap": gap,
            "accepted": bool(accepted),
        }
        self.score = {i: 0.0 for i in range(1, self.run_cfg.num_stimuli + 1)}
        self.epoch_count = 0
        return result


class UDPTriggerReceiver:
    """Receive Unity UDP trigger ID (single byte)."""

    def __init__(self, cfg: UDPTriggerConfig):
        self.cfg = cfg
        self.sock: Optional[socket.socket] = None
        self.triggers: List[int] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.cfg.bind_ip, self.cfg.bind_port))
        self.sock.settimeout(self.cfg.timeout_s)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.sock is not None:
            self.sock.close()

    def pop_all(self) -> List[int]:
        with self.lock:
            out = list(self.triggers)
            self.triggers.clear()
        return out

    def _run(self) -> None:
        assert self.sock is not None
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(8)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            trigger_id = int(data[0])
            if trigger_id not in self.cfg.valid_trigger_ids:
                continue
            with self.lock:
                self.triggers.append(trigger_id)


class OnlineP300UDPSystem:
    """
    Integration entry:
    1) feed continuous EEG samples via `push_eeg_chunk(ch x time)`
    2) Unity sends UDP trigger ID (row/col) to this process
    3) system cuts [trigger-pre, trigger+post] epoch and updates decoder
    4) every expected_triggers_per_trial triggers, emits one trial result
    """

    def __init__(
        self,
        classifier,
        classes: np.ndarray,
        n_channels: int,
        preproc_cfg: OnlinePreprocessConfig,
        epoch_cfg: EpochWindowConfig,
        udp_cfg: UDPTriggerConfig,
        run_cfg: OnlineRunConfig,
    ):
        self.run_cfg = run_cfg
        self.buffer = CircularEEGBuffer(
            n_channels=n_channels, fs=preproc_cfg.fs, buffer_seconds=epoch_cfg.buffer_seconds
        )
        self.extractor = TriggeredEpochExtractor(self.buffer, epoch_cfg)
        preproc = OnlineEpochPreprocessor(n_channels=n_channels, cfg=preproc_cfg)
        self.decoder = OnlineP300Decoder(
            classifier=classifier, classes=classes, preprocessor=preproc, run_cfg=run_cfg
        )
        self.udp_receiver = UDPTriggerReceiver(udp_cfg)
        self.received_trigger_count = 0
        self.processed_epoch_count = 0

    def start(self) -> None:
        self.udp_receiver.start()

    def stop(self) -> None:
        self.udp_receiver.stop()

    def push_eeg_chunk(self, chunk_ch_time: np.ndarray) -> List[Dict]:
        self.buffer.append(chunk_ch_time)
        for trigger_id in self.udp_receiver.pop_all():
            self.extractor.add_trigger_now(trigger_id)
            self.received_trigger_count += 1
        return self._consume_ready_epochs()

    def _consume_ready_epochs(self) -> List[Dict]:
        outputs: List[Dict] = []
        ready = self.extractor.pop_ready_epochs()
        for trigger_id, epoch in ready:
            p_target = self.decoder.update(epoch, trigger_id)
            self.processed_epoch_count += 1
            outputs.append(
                {
                    "type": "epoch_update",
                    "trigger_id": int(trigger_id),
                    "p_target": float(p_target),
                    "epoch_count_in_trial": int(self.decoder.epoch_count),
                }
            )

        if (
            self.received_trigger_count >= self.run_cfg.expected_triggers_per_trial
            and self.processed_epoch_count >= self.run_cfg.expected_triggers_per_trial
        ):
            trial_result = self.decoder.finalize_trial()
            trial_result["type"] = "trial_result"
            trial_result["expected_triggers_per_trial"] = int(
                self.run_cfg.expected_triggers_per_trial
            )
            outputs.append(trial_result)
            self.received_trigger_count = 0
            self.processed_epoch_count = 0
        return outputs


def main():
    clf, classes, method, channels = train_offline_calibration_model(subject_id=1, downsample=2)
    print("Online UDP P300 system config:")
    print(f"- trained artifact removal method: {method}")
    print(f"- channels: {len(channels)}")
    print("- Unity trigger mapping: row/col IDs should be 1..6")
    print("- epoch window: [trigger-0.2s, trigger+0.8s]")
    print("- finalize one trial after 18 triggers (2 rows+4 cols, 3 rounds)")

    system = OnlineP300UDPSystem(
        classifier=clf,
        classes=classes,
        n_channels=len(channels),
        preproc_cfg=OnlinePreprocessConfig(fs=100.0),
        epoch_cfg=EpochWindowConfig(pre_seconds=0.2, post_seconds=0.8, buffer_seconds=12.0),
        udp_cfg=UDPTriggerConfig(bind_ip="0.0.0.0", bind_port=9999),
        run_cfg=OnlineRunConfig(
            downsample=2, num_stimuli=6, expected_triggers_per_trial=18, topk=2
        ),
    )

    print("\nSystem object is ready.")
    print("Integrate your real EEG stream by repeatedly calling:")
    print("  outputs = system.push_eeg_chunk(chunk_ch_time)")
    print("where chunk_ch_time has shape (n_channels, n_times).")
    print("Each output contains either epoch_update or trial_result.")
    print("\nExample runtime loop is omitted to avoid fake EEG input.")

    # Keep UDP listener lifecycle explicit for integration.
    system.start()
    time.sleep(0.2)
    system.stop()


if __name__ == "__main__":
    main()
