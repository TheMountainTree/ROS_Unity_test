#!/usr/bin/env python3
import csv
import socket
import struct
import time
from pathlib import Path
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class EegTcpClientNode(Node):
    """Connect to EEG TCP Server and republish as Float32MultiArray.

    Default montage (SSVEP-8):
    1: POz, 2: PO3, 3: PO4, 4: PO5, 5: PO6, 6: Oz, 7: O1, 8: O2
    Plus 1 Trigger channel at the end of each frame.
    """

    # 只列出脑电通道，Trigger我们单独处理
    CHANNEL_NAMES: List[str] = ["POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2"]

    def __init__(self) -> None:
        super().__init__("eeg_tcp_client_node")

        self.declare_parameter("server_ip", "192.168.56.3")
        self.declare_parameter("server_port", 8712)
        self.declare_parameter("topic_name", "/eeg/ssvep8/raw")
        self.declare_parameter("recv_buffer_size", 4096)
        self.declare_parameter("save_csv", True)
        self.declare_parameter("csv_relative_path", "data/eeg_tcp_stream.csv")
        self.declare_parameter("eeg_sample_rate_hz", 1000.0)
        self.declare_parameter("backlog_interp_threshold_frames", 40)

        self.server_ip = str(self.get_parameter("server_ip").value)
        self.server_port = int(self.get_parameter("server_port").value)
        topic_name = str(self.get_parameter("topic_name").value)
        self.recv_buffer_size = int(self.get_parameter("recv_buffer_size").value)
        self.save_csv = bool(self.get_parameter("save_csv").value)
        self.csv_relative_path = str(self.get_parameter("csv_relative_path").value).strip()
        self.eeg_sample_rate_hz = float(self.get_parameter("eeg_sample_rate_hz").value)
        self.backlog_interp_threshold_frames = int(
            self.get_parameter("backlog_interp_threshold_frames").value
        )

        self.n_eeg_channels = len(self.CHANNEL_NAMES)
        self.frame_floats = self.n_eeg_channels + 1  # 8 EEG + 1 Trigger
        self.frame_bytes = self.frame_floats * 4     # 每个Float32占4字节，共36字节
        self.unpack_fmt = f"<{self.frame_floats}f"   # 小端序浮点数解包格式
        if self.eeg_sample_rate_hz <= 0:
            raise ValueError("eeg_sample_rate_hz must be > 0")
        if self.backlog_interp_threshold_frames <= 0:
            raise ValueError("backlog_interp_threshold_frames must be > 0")

        self.publisher_ = self.create_publisher(Float32MultiArray, topic_name, 100)
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
                # 表头加上 Trigger 列
                self.csv_writer.writerow(["timestamp"] + self.CHANNEL_NAMES + ["Trigger"])
                self.csv_file.flush()

        # 初始化 TCP Socket 客户端
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        
        self.get_logger().info(f"Connecting to EEG TCP Server at {self.server_ip}:{self.server_port}...")
        try:
            self.sock.connect((self.server_ip, self.server_port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.get_logger().info("Connected successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")
            raise

        self.sock.settimeout(0.01) # 设置为非阻塞模式的短超时，适配ROS的定时器轮询

        # 持久化接收缓存区（处理TCP粘包/半包的核心）
        self.buffer = bytearray()
        self.frame_synced = False
        self.backlog_warned = False
        
        self.timer = self.create_timer(0.01, self.poll_tcp)

        self.packet_count = 0
        self.sample_frame_count = 0

    def poll_tcp(self) -> None:
        try:
            # 尽量在一次轮询里读空内核接收缓冲，降低积压风险
            while True:
                chunk = self.sock.recv(self.recv_buffer_size)
                if not chunk:
                    self.get_logger().warn("TCP connection closed by server.")
                    rclpy.shutdown()
                    return
                self.buffer.extend(chunk)
        except socket.timeout:
            pass  # 无新数据，直接去处理缓存区里的遗留数据
        except Exception as e:
            self.get_logger().error(f"TCP recv error: {e}")
            return

        # 初始同步：滑动查找连续4个0x00，4字节末尾作为下一帧Byte0
        if not self.frame_synced:
            sync_pos = self.buffer.find(b"\x00\x00\x00\x00")
            if sync_pos < 0:
                # 保留末尾3字节，以便跨chunk匹配到4个0x00
                if len(self.buffer) > 3:
                    del self.buffer[:-3]
                return

            frame_start = sync_pos + 4
            if frame_start > 0:
                del self.buffer[:frame_start]
            self.frame_synced = True
            self.get_logger().info("TCP frame sync acquired (found 0x00000000 delimiter).")

        # 计算当前缓存区中包含多少个完整的帧
        num_complete_frames = len(self.buffer) // self.frame_bytes
        if num_complete_frames == 0:
            return

        eeg_data_flat = []
        now = time.time()
        use_ts_interp = (
            self.eeg_sample_rate_hz > 0
            and num_complete_frames > 1
        )

        if use_ts_interp and not self.backlog_warned:
            self.get_logger().warn(
                "TCP frame backlog detected; enabling per-frame timestamp interpolation "
                f"(frames_this_tick={num_complete_frames}, threshold={self.backlog_interp_threshold_frames})."
            )
            self.backlog_warned = True

        # 按精确的帧长度提取并解包
        for frame_idx in range(num_complete_frames):
            frame_raw = self.buffer[:self.frame_bytes]
            del self.buffer[:self.frame_bytes] # 从缓存中移除已读的字节
            
            # 解包：返回 9 个浮点数的元组 (POz, ..., O2, Trigger)
            vals = struct.unpack(self.unpack_fmt, frame_raw)
            
            eeg_data_flat.extend(vals[:self.n_eeg_channels]) # 发布给ROS时只发EEG通道
            
            if self.csv_writer is not None:
                if use_ts_interp:
                    # 以当前时刻作为最后一帧时间，按采样率反推本批次各帧时间
                    ts = now - ((num_complete_frames - 1 - frame_idx) / self.eeg_sample_rate_hz)
                else:
                    ts = now
                row = [f"{ts:.6f}"] + list(vals)
                self.csv_writer.writerow(row)

        if self.csv_file is not None:
            self.csv_file.flush()

        # 发布至 ROS Topic
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="samples", size=num_complete_frames, stride=num_complete_frames * self.n_eeg_channels),
            MultiArrayDimension(label="channels", size=self.n_eeg_channels, stride=self.n_eeg_channels),
        ]
        msg.layout.data_offset = 0
        msg.data = eeg_data_flat
        self.publisher_.publish(msg)

        self.packet_count += 1
        self.sample_frame_count += num_complete_frames
        
        # 为了避免刷屏，每收到一定数据量打印一次日志
        if self.packet_count % 10 == 0:
            self.get_logger().info(
                f"TCP Read: Extracted {num_complete_frames} frames. Total frames={self.sample_frame_count}. Buffer leftover={len(self.buffer)} bytes."
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
    try:
        node = EegTcpClientNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Node terminated: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
