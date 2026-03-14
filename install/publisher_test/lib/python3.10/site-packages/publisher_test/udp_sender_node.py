#!/usr/bin/env python3
import socket

import rclpy
from rclpy.node import Node


class UdpSenderNode(Node):
    def __init__(self):
        super().__init__("udp_sender_node")

        self.declare_parameter("local_ip", "192.168.56.103")
        self.declare_parameter("local_port", 5006)
        self.declare_parameter("remote_ip", "192.168.56.3")
        self.declare_parameter("remote_port", 8888)
        self.declare_parameter("trigger_value", 1)
        self.declare_parameter("auto_increment", False)
        self.declare_parameter("publish_hz", 2.0)

        local_ip = self.get_parameter("local_ip").value
        local_port = int(self.get_parameter("local_port").value)
        self.remote_ip = self.get_parameter("remote_ip").value
        self.remote_port = int(self.get_parameter("remote_port").value)
        self.trigger_value = int(self.get_parameter("trigger_value").value)
        self.auto_increment = bool(self.get_parameter("auto_increment").value)
        publish_hz = float(self.get_parameter("publish_hz").value)

        if publish_hz <= 0:
            raise ValueError("publish_hz must be > 0")
        if not 0 <= self.trigger_value <= 255:
            raise ValueError("trigger_value must be in [0, 255]")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((local_ip, local_port))
        # UDP connect fixes the target endpoint so send() can be used directly.
        self.sock.connect((self.remote_ip, self.remote_port))

        timer_period = 1.0 / publish_hz
        self.timer = self.create_timer(timer_period, self.send_udp)

        self.get_logger().info(
            f"UDP trigger sender started: {local_ip}:{local_port} -> "
            f"{self.remote_ip}:{self.remote_port}, {publish_hz:.2f} Hz, "
            f"trigger={self.trigger_value}, auto_increment={self.auto_increment}"
        )

    def send_udp(self):
        payload = int(self.trigger_value).to_bytes(1, byteorder="little", signed=False)
        self.sock.send(payload)
        self.get_logger().info(f"Sent trigger: {self.trigger_value} (HEX: {self.trigger_value:02X})")
        if self.auto_increment:
            self.trigger_value = (self.trigger_value + 1) % 256

    def destroy_node(self):
        try:
            self.sock.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UdpSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
