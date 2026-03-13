#!/usr/bin/env python3
import csv
import os
import random
import socket
import time
from datetime import datetime
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image


class CentralControllerNode(Node):
    """
    Calibration-stage central controller.

    Behavior:
    - No cue topic.
    - For each trial, generate 6 images: 1 red target + 5 white non-target.
    - Target position (1..6) is random per trial.
    - Publish 6 images sequentially to /image_seg.

    Unity can keep its current logic: receive 6 images, then start row/column flashing.
    """

    def __init__(self):
        super().__init__("central_controller_node")

        self.declare_parameter("image_topic", "/image_seg")
        self.declare_parameter("use_reliable_qos", True)
        self.declare_parameter("image_publish_period", 0.50)
        # For stimRounds=5, flashDuration=0.1, interFlashInterval=0.075:
        # T_flash ~= 5 * 6 * (0.1 + 0.075) = 5.25s. Keep margin to avoid cross-trial.
        self.declare_parameter("inter_trial_interval", 0.0)
        self.declare_parameter("startup_delay", 1.5)
        self.declare_parameter("max_trials", 5)  # 0 means no limit
        self.declare_parameter("ack_bind_ip", "0.0.0.0")
        self.declare_parameter("ack_port", 10000)
        self.declare_parameter("trigger_bind_ip", "0.0.0.0")
        self.declare_parameter("trigger_port", 9999)
        self.declare_parameter(
            "save_dir",
            "/home/themountaintree/workspace/ROS_Unity_test/data/central_controller",
        )
        self.declare_parameter("num_images", 6)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_width", 640)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.use_reliable_qos = (
            self.get_parameter("use_reliable_qos").get_parameter_value().bool_value
        )
        self.image_period = (
            self.get_parameter("image_publish_period").get_parameter_value().double_value
        )
        self.inter_trial_interval = (
            self.get_parameter("inter_trial_interval").get_parameter_value().double_value
        )
        self.startup_delay = (
            self.get_parameter("startup_delay").get_parameter_value().double_value
        )
        self.max_trials = int(self.get_parameter("max_trials").get_parameter_value().integer_value)
        self.ack_bind_ip = self.get_parameter("ack_bind_ip").get_parameter_value().string_value
        self.ack_port = self.get_parameter("ack_port").get_parameter_value().integer_value
        self.trigger_bind_ip = (
            self.get_parameter("trigger_bind_ip").get_parameter_value().string_value
        )
        self.trigger_port = self.get_parameter("trigger_port").get_parameter_value().integer_value
        self.save_dir = self.get_parameter("save_dir").get_parameter_value().string_value
        self.num_images = int(self.get_parameter("num_images").get_parameter_value().integer_value)
        self.image_h = int(self.get_parameter("image_height").get_parameter_value().integer_value)
        self.image_w = int(self.get_parameter("image_width").get_parameter_value().integer_value)

        if self.num_images <= 1:
            raise ValueError("num_images must be >= 2")

        reliability = (
            QoSReliabilityPolicy.RELIABLE
            if self.use_reliable_qos
            else QoSReliabilityPolicy.BEST_EFFORT
        )
        image_qos = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.image_pub = self.create_publisher(Image, image_topic, image_qos)

        self.trial_idx = 0
        self.current_target_id = 1
        self.prev_target_id = None
        self.current_trial_start_ts = ""
        self.current_trial_trigger_count = 0
        self.current_trial_images: List[np.ndarray] = []
        self.publish_idx = 0
        self.wait_until = 0.0
        self.waiting_ack_trial_id = -1
        self.state = "init_wait"
        self.start_ready_time = time.monotonic() + self.startup_delay

        os.makedirs(self.save_dir, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trial_csv_path = os.path.join(self.save_dir, f"trials_{run_stamp}.csv")
        self.trigger_csv_path = os.path.join(self.save_dir, f"triggers_{run_stamp}.csv")
        self.trial_csv_file = open(self.trial_csv_path, "w", newline="", encoding="utf-8")
        self.trigger_csv_file = open(self.trigger_csv_path, "w", newline="", encoding="utf-8")
        self.trial_writer = csv.writer(self.trial_csv_file)
        self.trigger_writer = csv.writer(self.trigger_csv_file)
        self.trial_writer.writerow(
            [
                "trial_id",
                "target_id",
                "trial_start_wall_time",
                "trial_end_wall_time",
                "ack_trial_id",
                "trigger_count",
            ]
        )
        self.trigger_writer.writerow(
            [
                "wall_time",
                "mono_time",
                "trial_id",
                "target_id",
                "trigger_id",
                "src_ip",
                "src_port",
            ]
        )
        self.trial_csv_file.flush()
        self.trigger_csv_file.flush()

        self.ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ack_sock.bind((self.ack_bind_ip, int(self.ack_port)))
        self.ack_sock.setblocking(False)
        self.trigger_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.trigger_sock.bind((self.trigger_bind_ip, int(self.trigger_port)))
        self.trigger_sock.setblocking(False)

        self.timer = self.create_timer(self.image_period, self._on_timer)

        self.get_logger().info(
            "CentralControllerNode calibration mode started: "
            f"topic={image_topic}, image_period={self.image_period:.3f}s, "
            f"inter_trial_interval={self.inter_trial_interval:.3f}s, "
            f"startup_delay={self.startup_delay:.2f}s, "
            f"max_trials={self.max_trials}, "
            f"qos={'RELIABLE' if self.use_reliable_qos else 'BEST_EFFORT'}, "
            f"ack_udp={self.ack_bind_ip}:{self.ack_port}, "
            f"trigger_udp={self.trigger_bind_ip}:{self.trigger_port}, "
            f"save_dir={self.save_dir}"
        )
        self.get_logger().info(
            f"saving files: trial_csv={self.trial_csv_path}, trigger_csv={self.trigger_csv_path}"
        )

    def _white_image(self) -> np.ndarray:
        return np.full((self.image_h, self.image_w, 3), 255, dtype=np.uint8)

    def _red_image(self) -> np.ndarray:
        # bgr8: red is [0, 0, 255]
        img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]
        return img

    def _prepare_new_trial(self) -> None:
        if self.max_trials > 0 and self.trial_idx >= self.max_trials:
            self.state = "done"
            self.get_logger().info(
                f"max_trials={self.max_trials} reached. stop scheduling new trials."
            )
            return

        self.trial_idx += 1
        self.current_target_id = random.randint(1, self.num_images)
        if self.num_images > 1 and self.prev_target_id is not None:
            # Avoid consecutive same target to make calibration refresh obvious on UI.
            while self.current_target_id == self.prev_target_id:
                self.current_target_id = random.randint(1, self.num_images)
        self.prev_target_id = self.current_target_id
        self.current_trial_images = [self._white_image() for _ in range(self.num_images)]
        self.current_trial_images[self.current_target_id - 1] = self._red_image()
        self.publish_idx = 0
        self.state = "publishing"
        self.current_trial_start_ts = datetime.now().isoformat(timespec="milliseconds")
        self.current_trial_trigger_count = 0

        self.get_logger().info(
            f"[Trial {self.trial_idx}] target_id={self.current_target_id} "
            f"(1 red + {self.num_images - 1} white)"
        )

    def _to_ros_image(self, bgr: np.ndarray, img_idx_1based: int) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"trial={self.trial_idx};img={img_idx_1based};target={self.current_target_id}"
        )
        msg.height = int(bgr.shape[0])
        msg.width = int(bgr.shape[1])
        msg.encoding = "bgr8"
        msg.step = int(bgr.shape[1] * 3)
        msg.data = bgr.tobytes()
        return msg

    def _poll_trial_done_ack(self) -> int:
        # Return acked trial_id, or -1 if no valid ack available.
        while True:
            try:
                payload, addr = self.ack_sock.recvfrom(128)
            except BlockingIOError:
                return -1
            except OSError:
                return -1
            try:
                text = payload.decode("utf-8", errors="ignore").strip().lower()
            except Exception:
                continue
            self.get_logger().info(f"ack udp recv from {addr}: '{text}'")
            if not text.startswith("trial_done="):
                continue
            value = text.split("=", 1)[1].strip()
            try:
                return int(value)
            except ValueError:
                continue

    def _poll_trigger_ids(self) -> None:
        while True:
            try:
                payload, addr = self.trigger_sock.recvfrom(128)
            except BlockingIOError:
                return
            except OSError:
                return

            trigger_id = None
            if len(payload) == 1:
                trigger_id = int(payload[0])
            else:
                try:
                    trigger_id = int(payload.decode("utf-8", errors="ignore").strip())
                except ValueError:
                    trigger_id = None

            if trigger_id is None:
                continue

            wall_ts = datetime.now().isoformat(timespec="milliseconds")
            mono_ts = time.monotonic()
            self.trigger_writer.writerow(
                [
                    wall_ts,
                    f"{mono_ts:.6f}",
                    self.trial_idx,
                    self.current_target_id,
                    trigger_id,
                    addr[0],
                    addr[1],
                ]
            )
            self.trigger_csv_file.flush()
            if self.trial_idx > 0:
                self.current_trial_trigger_count += 1

    def _on_timer(self) -> None:
        now = time.monotonic()
        self._poll_trigger_ids()

        if self.state == "done":
            return

        if self.state == "init_wait":
            # Wait startup delay and at least one subscriber to reduce first-trial packet loss.
            if now < self.start_ready_time:
                return
            try:
                sub_count = self.image_pub.get_subscription_count()
            except Exception:
                sub_count = 1
            if sub_count < 1:
                return
            self._prepare_new_trial()
            return

        if self.state == "waiting":
            if now >= self.wait_until:
                self._prepare_new_trial()
            return

        if self.state == "wait_ack":
            ack_trial = self._poll_trial_done_ack()
            if ack_trial == self.waiting_ack_trial_id and now >= self.wait_until:
                self.get_logger().info(
                    f"[Trial {self.waiting_ack_trial_id}] ack received, start next trial"
                )
                self.trial_writer.writerow(
                    [
                        self.waiting_ack_trial_id,
                        self.current_target_id,
                        self.current_trial_start_ts,
                        datetime.now().isoformat(timespec="milliseconds"),
                        ack_trial,
                        self.current_trial_trigger_count,
                    ]
                )
                self.trial_csv_file.flush()
                self._prepare_new_trial()
                return
            return

        frame = self.current_trial_images[self.publish_idx]
        self.image_pub.publish(self._to_ros_image(frame, self.publish_idx + 1))

        self.get_logger().info(
            f"[Trial {self.trial_idx}] published {self.publish_idx + 1}/{self.num_images}"
        )
        self.publish_idx += 1

        if self.publish_idx >= self.num_images:
            self.state = "wait_ack"
            self.waiting_ack_trial_id = self.trial_idx
            self.wait_until = now + self.inter_trial_interval
            self.get_logger().info(
                f"[Trial {self.trial_idx}] batch complete, wait ack + {self.inter_trial_interval:.2f}s"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CentralControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.ack_sock.close()
        except Exception:
            pass
        try:
            node.trigger_sock.close()
        except Exception:
            pass
        try:
            node.trial_csv_file.close()
        except Exception:
            pass
        try:
            node.trigger_csv_file.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
