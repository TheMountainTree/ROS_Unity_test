#!/usr/bin/env python3
import csv
import socket
import struct
import time
from pathlib import Path
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class EegUdpListenerNode(Node):
    """Listen EEG UDP packets and republish as Float32MultiArray.

    Default montage (SSVEP-8):
    1: POz, 2: PO3, 3: PO4, 4: PO5, 5: PO6, 6: Oz, 7: O1, 8: O2
    """

    CHANNEL_NAMES: List[str] = ["POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2"]

    def __init__(self) -> None:
        super().__init__("eeg_udp_listener_node")

        self.declare_parameter("bind_ip", "0.0.0.0")
        self.declare_parameter("bind_port", 8712)
        self.declare_parameter("source_ip", "192.168.56.3")
        self.declare_parameter("topic_name", "/eeg/ssvep8/raw")
        self.declare_parameter("sample_type", "float32_le")  # float32_le | int16_le
        self.declare_parameter("sample_scale", 1.0)  # used for int16 scaling if needed
        self.declare_parameter("recv_buffer_size", 65535)
        self.declare_parameter("save_csv", True)
        self.declare_parameter("csv_relative_path", "data/eeg_udp_stream.csv")

        bind_ip = str(self.get_parameter("bind_ip").value)
        bind_port = int(self.get_parameter("bind_port").value)
        self.source_ip = str(self.get_parameter("source_ip").value)
        topic_name = str(self.get_parameter("topic_name").value)
        self.sample_type = str(self.get_parameter("sample_type").value).strip().lower()
        self.sample_scale = float(self.get_parameter("sample_scale").value)
        self.recv_buffer_size = int(self.get_parameter("recv_buffer_size").value)
        self.save_csv = bool(self.get_parameter("save_csv").value)
        self.csv_relative_path = str(self.get_parameter("csv_relative_path").value).strip()

        if self.sample_type not in ("float32_le", "int16_le"):
            raise ValueError("sample_type must be one of: float32_le, int16_le")
        if self.recv_buffer_size <= 0:
            raise ValueError("recv_buffer_size must be > 0")
        if Path(self.csv_relative_path).is_absolute():
            raise ValueError("csv_relative_path must be a relative path")

        self.n_channels = len(self.CHANNEL_NAMES)
        self.publisher_ = self.create_publisher(Float32MultiArray, topic_name, 10)
        self.csv_file = None
        self.csv_writer = None

        if self.save_csv:
            package_root = Path(__file__).resolve().parents[1]
            csv_path = package_root / self.csv_relative_path
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = csv_path.exists() and csv_path.stat().st_size > 0
            self.csv_file = open(csv_path, mode="a", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            if not file_exists:
                self.csv_writer.writerow(["timestamp"] + self.CHANNEL_NAMES)
                self.csv_file.flush()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, bind_port))
        self.sock.settimeout(0.2)

        self.timer = self.create_timer(0.01, self.poll_udp)

        self.packet_count = 0
        self.sample_frame_count = 0

        self.get_logger().info(
            "EEG UDP listener started: "
            f"bind={bind_ip}:{bind_port}, source_ip={self.source_ip}, topic={topic_name}, "
            f"sample_type={self.sample_type}, channels={self.n_channels}, montage={','.join(self.CHANNEL_NAMES)}"
        )
        if self.save_csv:
            self.get_logger().info(f"CSV save enabled: relative_path={self.csv_relative_path}")

    def _decode_payload(self, payload: bytes) -> Tuple[List[float], int]:
        """Decode payload into flat float list and number of sample frames."""
        if self.sample_type == "float32_le":
            bytes_per_scalar = 4
            struct_fmt = "f"
        else:
            bytes_per_scalar = 2
            struct_fmt = "h"

        total_scalars = len(payload) // bytes_per_scalar
        if total_scalars == 0:
            return [], 0

        valid_scalars = total_scalars - (total_scalars % self.n_channels)
        if valid_scalars <= 0:
            return [], 0

        valid_bytes = valid_scalars * bytes_per_scalar
        raw = payload[:valid_bytes]

        unpack_fmt = f"<{valid_scalars}{struct_fmt}"
        vals = struct.unpack(unpack_fmt, raw)

        if self.sample_type == "int16_le":
            data = [float(v) * self.sample_scale for v in vals]
        else:
            data = [float(v) for v in vals]

        sample_frames = valid_scalars // self.n_channels
        return data, sample_frames

    def poll_udp(self) -> None:
        while True:
            try:
                payload, addr = self.sock.recvfrom(self.recv_buffer_size)
            except socket.timeout:
                break
            except OSError as exc:
                self.get_logger().error(f"UDP receive error: {exc}")
                break

            src_ip, src_port = addr
            if self.source_ip and src_ip != self.source_ip:
                self.get_logger().debug(
                    f"Ignored UDP from unexpected source {src_ip}:{src_port}, expected {self.source_ip}"
                )
                continue

            data, sample_frames = self._decode_payload(payload)
            if sample_frames <= 0:
                self.get_logger().warn(
                    f"Dropped packet from {src_ip}:{src_port}, payload_len={len(payload)} incompatible with {self.n_channels} channels"
                )
                continue

            msg = Float32MultiArray()
            msg.layout.dim = [
                MultiArrayDimension(label="samples", size=sample_frames, stride=sample_frames * self.n_channels),
                MultiArrayDimension(label="channels", size=self.n_channels, stride=self.n_channels),
            ]
            msg.layout.data_offset = 0
            msg.data = data
            self.publisher_.publish(msg)

            if self.csv_writer is not None and self.csv_file is not None:
                now = time.time()
                for i in range(sample_frames):
                    offset = i * self.n_channels
                    row = [f"{now:.6f}"] + data[offset:offset + self.n_channels]
                    self.csv_writer.writerow(row)
                self.csv_file.flush()

            self.packet_count += 1
            self.sample_frame_count += sample_frames
            self.get_logger().info(
                f"EEG packet#{self.packet_count}: {sample_frames}x{self.n_channels} samples "
                f"from {src_ip}:{src_port}, total_frames={self.sample_frame_count}"
            )

    def destroy_node(self) -> None:
        try:
            self.sock.close()
            if self.csv_file is not None:
                self.csv_file.close()
        finally:
            super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EegUdpListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
