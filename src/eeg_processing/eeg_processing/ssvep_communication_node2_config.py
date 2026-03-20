#!/usr/bin/env python3
"""Static configuration for SSVEP_Communication_Node2."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List


@dataclass
class GeneralConfig:
    use_reliable_qos: bool = True
    loop_period_s: float = 0.02
    startup_delay: float = 1.0
    num_targets: int = 8
    ssvep_frequencies_hz: List[float] = field(
        default_factory=lambda: [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0]
    )
    save_dir: str = "data/central_controller_ssvep3"


@dataclass
class UnityCommConfig:
    host_ip: str = "0.0.0.0"
    decode_start_port: int = 10000
    pretrain_start_port: int = 10001
    image_topic: str = "/image_seg"
    decode_command_topic: str = "/ssvep_decode_cmd"
    command_topic: str = "/ssvep_train_cmd"


@dataclass
class TriggerForwardConfig:
    local_ip: str = "192.168.56.103"
    local_port: int = 5006
    remote_ip: str = "192.168.56.3"
    remote_port: int = 8888


@dataclass
class EEGServerConfig:
    server_ip: str = "192.168.56.3"
    server_port: int = 8712
    recv_buffer_size: int = 4096
    n_channels: int = 8
    frame_floats: int = 9
    fs: float = 1000.0


@dataclass
class DecodeConfig:
    image_publish_period: float = 0.5
    inter_trial_interval: float = 0.0
    trial_duration_s: float = 4.0
    pre_stim_hold_s: float = 1.5
    num_images: int = 6
    max_trials: int = 1
    start_wait_timeout_s: float = 15.0
    capture_wait_timeout_s: float = 1.0
    image_height: int = 480
    image_width: int = 640
    image_paths: List[str] = field(default_factory=list)
    image_dir: str = (
        "~/workspace/eeg_robot/src/robot_ctr/graph/graph/results/"
        "segmentation_20260206_223629"
    )


@dataclass
class PretrainConfig:
    repetitions_per_target: int = 3
    cue_duration_s: float = 2.0
    stim_duration_s: float = 1.5
    rest_duration_s: float = 1.0


@dataclass
class ReasonerConfig:
    enabled: bool = False
    input_topic: str = "/reasoner/images"
    output_topic: str = "/reasoner/feedback"
    mock_selected_index: int = -1
    history_image_topic: str = "/history_image"
    history_image_width: int = 120
    history_image_height: int = 120
    history_udp_ip: str = "127.0.0.1"
    history_udp_port: int = 12001


@dataclass
class SSVEPCommunicationConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    unity: UnityCommConfig = field(default_factory=UnityCommConfig)
    trigger_forward: TriggerForwardConfig = field(default_factory=TriggerForwardConfig)
    eeg_server: EEGServerConfig = field(default_factory=EEGServerConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)


DEFAULT_SSVEP_COMMUNICATION_CONFIG = SSVEPCommunicationConfig()


def make_default_config() -> SSVEPCommunicationConfig:
    """Return a deep-copied mutable config object."""
    return deepcopy(DEFAULT_SSVEP_COMMUNICATION_CONFIG)
