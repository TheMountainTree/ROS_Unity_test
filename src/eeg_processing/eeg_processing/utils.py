#!/usr/bin/env python3
"""Common utilities shared by SSVEP communication nodes."""

import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


class CircularEEGBuffer:
    """Ring buffer for continuous EEG samples with absolute sample indexing.

    Thread-safe version using threading.Lock for concurrent access protection.
    """

    def __init__(self, n_channels: int, fs: float, buffer_seconds: float):
        """
        Initialize the circular EEG buffer.

        Args:
            n_channels: Number of EEG channels
            fs: Sampling frequency in Hz
            buffer_seconds: Buffer capacity in seconds
        """
        self.n_channels = int(n_channels)
        self.fs = float(fs)
        self.capacity = int(round(self.fs * float(buffer_seconds)))

        if self.capacity <= 0:
            raise ValueError("buffer capacity must be > 0")

        self.data = np.zeros((self.n_channels, self.capacity), dtype=np.float32)
        self.write_ptr = 0
        self.total_written = 0
        self._lock = threading.Lock()

    @property
    def latest_abs_index(self) -> int:
        """Return the absolute index of the latest sample."""
        with self._lock:
            return int(self.total_written)

    @property
    def oldest_abs_index(self) -> int:
        """Return the absolute index of the oldest available sample."""
        with self._lock:
            return max(0, int(self.total_written) - int(self.capacity))

    def append(self, chunk_ch_time: np.ndarray) -> Tuple[int, int]:
        """
        Append a chunk of EEG data to the buffer.

        Args:
            chunk_ch_time: EEG data chunk with shape (n_channels, n_times)

        Returns:
            Tuple of (start_abs_index, end_abs_index) for the appended data

        Raises:
            ValueError: If chunk shape is invalid
        """
        x = np.asarray(chunk_ch_time, dtype=np.float32)

        if x.ndim != 2 or x.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk must have shape (n_channels, n_times), "
                f"expected ({self.n_channels}, T), got {x.shape}"
            )

        n = int(x.shape[1])
        if n == 0:
            with self._lock:
                return self.total_written, self.total_written

        with self._lock:
            if n >= self.capacity:
                x = x[:, -self.capacity:]
                n = self.capacity

            start_abs = int(self.total_written)
            first = min(n, self.capacity - self.write_ptr)
            second = n - first

            self.data[:, self.write_ptr:self.write_ptr + first] = x[:, :first]

            if second > 0:
                self.data[:, :second] = x[:, first:]

            self.write_ptr = (self.write_ptr + n) % self.capacity
            self.total_written += n

            return start_abs, int(self.total_written)

    def has_range(self, abs_start: int, abs_end: int) -> bool:
        """
        Check if the requested range is available in the buffer.

        Args:
            abs_start: Start absolute index
            abs_end: End absolute index (exclusive)

        Returns:
            True if range is available, False otherwise
        """
        if abs_end <= abs_start:
            return False

        with self._lock:
            oldest = max(0, int(self.total_written) - int(self.capacity))
            latest = int(self.total_written)
            return abs_start >= oldest and abs_end <= latest

    def get_range(self, abs_start: int, abs_end: int) -> np.ndarray:
        """
        Get a range of samples from the buffer.

        Args:
            abs_start: Start absolute index
            abs_end: End absolute index (exclusive)

        Returns:
            EEG data with shape (n_channels, n_samples)

        Raises:
            ValueError: If requested range is not available
        """
        with self._lock:
            oldest = max(0, int(self.total_written) - int(self.capacity))
            latest = int(self.total_written)
            if abs_end <= abs_start or abs_start < oldest or abs_end > latest:
                raise ValueError("requested range is not available in ring buffer")

            n = abs_end - abs_start
            rel_start = abs_start % self.capacity
            first = min(n, self.capacity - rel_start)
            second = n - first

            out = np.empty((self.n_channels, n), dtype=np.float32)
            out[:, :first] = self.data[:, rel_start:rel_start + first]

            if second > 0:
                out[:, first:] = self.data[:, :second]

            return out


class NodeState(Enum):
    """Enumeration of node states for the SSVEP controller."""

    # Initialization
    INIT_WAIT = auto()

    # Common states
    WAITING = auto()
    DONE = auto()

    # Decode mode states
    DECODE_PUBLISHING = auto()
    DECODE_HOLD = auto()
    DECODE_WAIT_START = auto()
    DECODE_STIMULATING = auto()
    DECODE_WAIT_CAPTURE = auto()

    # Reasoner mode states
    REASONER_WAIT_BATCH = auto()
    REASONER_WAIT_SELECTION = auto()

    # Pretrain mode states
    PRETRAIN_CUEING = auto()
    PRETRAIN_STIMULATING = auto()
    PRETRAIN_RESTING = auto()


@dataclass
class TrialState:
    """
    Dataclass for trial state variables.

    Contains all state variables that need to be reset between trials.
    """
    # Timing
    trial_start_mono: float = 0.0
    prepared_wall: str = ""
    start_wall: str = ""
    end_wall: str = ""
    decode_stop_mono: float = 0.0

    # Trial identification
    start_trial_id: int = -1
    start_status: str = "not_started"

    # Epoch mode
    epoch_mode: Optional[str] = None

    # Cue and stim timing (pretrain)
    cue_start_wall: str = ""
    stim_start_wall: str = ""
    stim_end_wall: str = ""

    # Trigger tracking
    trigger_received: bool = False
    trigger_wall: str = ""
    stim_start_trigger_sent: bool = False
    stim_end_trigger_sent: bool = False
    stim_start_trigger_wall: str = ""
    stim_end_trigger_wall: str = ""

    # EEG absolute indices
    stim_enter_abs: int = -1
    stim_exit_abs: int = -1
    stim_start_abs: int = -1
    stim_end_abs_inclusive: int = -1

    # Sample counts and flags
    raw_samples: int = 0
    epoch_complete: bool = False
    epoch_saved: bool = False
    trial_record_written: bool = False

    # Trigger processing
    last_trigger_value: int = 0
    epoch_start_pending: bool = False


def reset_trial_state_dict() -> TrialState:
    """Return a fresh per-trial state container."""
    return TrialState()

@dataclass
class NetworkConfig:
    """Network communication configuration."""

    decode_ack_ip: str
    decode_ack_port: int
    pretrain_ack_ip: str
    pretrain_ack_port: int
    trigger_local_ip: str
    trigger_local_port: int
    trigger_remote_ip: str
    trigger_remote_port: int
    eeg_server_ip: str
    eeg_server_port: int
    eeg_recv_buffer_size: int
    eeg_n_channels: int
    eeg_frame_floats: int
    eeg_fs: float


@dataclass
class DecodeConfig:
    """Decode mode configuration parameters."""

    image_publish_period: float
    inter_trial_interval: float
    trial_duration_s: float
    pre_stim_hold_s: float
    num_images: int
    max_trials: int
    start_wait_timeout_s: float
    capture_wait_timeout_s: float
    image_height: int
    image_width: int
    image_paths: List[str]
    image_dir: str


@dataclass
class PretrainConfig:
    """Pretrain mode configuration parameters."""

    repetitions_per_target: int
    cue_duration_s: float
    stim_duration_s: float
    rest_duration_s: float


@dataclass
class ReasonerConfig:
    """Reasoner mode configuration parameters."""

    enabled: bool
    input_topic: str
    output_topic: str
    mock_selected_index: int
    history_image_topic: str
    history_image_width: int
    history_image_height: int
    history_udp_ip: str
    history_udp_port: int
