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

        self.server_ip = str(self.get_parameter("server_ip").value)
        self.server_port = int(self.get_parameter("server_port").value)
        topic_name = str(self.get_parameter("topic_name").value)
        self.recv_buffer_size = int(self.get_parameter("recv_buffer_size").value)
        self.save_csv = bool(self.get_parameter("save_csv").value)
        self.csv_relative_path = str(self.get_parameter("csv_relative_path").value).strip()

        self.n_eeg_channels = len(self.CHANNEL_NAMES)
        self.frame_floats = self.n_eeg_channels + 1  # 8 EEG + 1 Trigger
        self.frame_bytes = self.frame_floats * 4     # 每个Float32占4字节，共36字节
        self.unpack_fmt = f"<{self.frame_floats}f"   # 小端序浮点数解包格式

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
            self.get_logger().info("Connected successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")
            raise

        self.sock.settimeout(0.01) # 设置为非阻塞模式的短超时，适配ROS的定时器轮询

        # 持久化接收缓存区（处理TCP粘包/半包的核心）
        self.buffer = bytearray()
        
        self.timer = self.create_timer(0.01, self.poll_tcp)

        self.packet_count = 0
        self.sample_frame_count = 0

    def poll_tcp(self) -> None:
        try:
            # 不断接收数据追加到缓存
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

        # 计算当前缓存区中包含多少个完整的帧
        num_complete_frames = len(self.buffer) // self.frame_bytes
        if num_complete_frames == 0:
            return

        eeg_data_flat = []
        
        now = time.time()

        # 按精确的帧长度提取并解包
        for _ in range(num_complete_frames):
            frame_raw = self.buffer[:self.frame_bytes]
            del self.buffer[:self.frame_bytes] # 从缓存中移除已读的字节
            
            # 解包：返回 9 个浮点数的元组 (POz, ..., O2, Trigger)
            vals = struct.unpack(self.unpack_fmt, frame_raw)
            
            eeg_data_flat.extend(vals[:self.n_eeg_channels]) # 发布给ROS时只发EEG通道
            
            if self.csv_writer is not None:
                row = [f"{now:.6f}"] + list(vals)
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