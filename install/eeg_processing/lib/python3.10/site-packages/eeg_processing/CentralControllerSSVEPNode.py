#!/usr/bin/env python3
import csv
import glob
import os
import random
import socket
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class CentralControllerSSVEPNode(Node):
    """
    SSVEP central controller (ROS-driven timing, Unity start handshake):
    1) Load local images (fallback: generated placeholders)
    2) Publish 6 images to /image_seg for each trial
    3) Wait Unity UDP: trial_started=<trial_id>
    4) Start trial timing in ROS only after start handshake
    5) Move to next trial by ROS timer, not Unity trial_done
    """

    def __init__(self):
        super().__init__("central_controller_ssvep_node")

        self.declare_parameter("image_topic", "/image_seg")
        self.declare_parameter("use_reliable_qos", True)
        self.declare_parameter("image_publish_period", 0.50)
        self.declare_parameter("inter_trial_interval", 0.0)
        self.declare_parameter("startup_delay", 1.5)
        self.declare_parameter("max_trials", 0)  # 0 means no limit
        self.declare_parameter("start_bind_ip", "0.0.0.0")
        self.declare_parameter("start_port", 10000)
        self.declare_parameter("start_wait_timeout_s", 15.0)
        self.declare_parameter("num_targets", 8)
        self.declare_parameter("num_images", 6)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("trial_duration_s", 2.0)
        self.declare_parameter("ssvep_frequencies_hz", [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0])
        self.declare_parameter("image_paths", [])
        self.declare_parameter(
            "image_dir",
            os.path.join(os.path.dirname(__file__), "assets", "stimuli"),
        )
        self.declare_parameter(
            "save_dir",
            "/home/themountaintree/workspace/ROS_Unity_test/data/central_controller_ssvep",
        )

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
        self.start_bind_ip = self.get_parameter("start_bind_ip").get_parameter_value().string_value
        self.start_port = int(self.get_parameter("start_port").get_parameter_value().integer_value)
        self.start_wait_timeout_s = (
            self.get_parameter("start_wait_timeout_s").get_parameter_value().double_value
        )
        self.num_targets = int(self.get_parameter("num_targets").get_parameter_value().integer_value)
        self.num_images = int(self.get_parameter("num_images").get_parameter_value().integer_value)
        self.image_h = int(self.get_parameter("image_height").get_parameter_value().integer_value)
        self.image_w = int(self.get_parameter("image_width").get_parameter_value().integer_value)
        self.trial_duration_s = (
            self.get_parameter("trial_duration_s").get_parameter_value().double_value
        )
        self.ssvep_frequencies = [
            float(v)
            for v in self.get_parameter("ssvep_frequencies_hz")
            .get_parameter_value()
            .double_array_value
        ]
        self.image_paths = list(
            self.get_parameter("image_paths").get_parameter_value().string_array_value
        )
        self.image_dir = self.get_parameter("image_dir").get_parameter_value().string_value
        self.save_dir = self.get_parameter("save_dir").get_parameter_value().string_value

        if self.num_images <= 1:
            raise ValueError("num_images must be >= 2")
        if len(self.ssvep_frequencies) < self.num_targets:
            raise ValueError("ssvep_frequencies_hz length must be >= num_targets")

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

        self.base_images = self._load_or_generate_images(self.num_images)
        self.base_image_ids = list(range(1, self.num_images + 1))

        self.state = "init_wait"
        self.start_ready_time = time.monotonic() + self.startup_delay
        self.trial_idx = 0
        self.current_trial_start_mono = 0.0
        self.current_trial_prepared_wall = ""
        self.current_trial_start_wall = ""
        self.current_start_trial_id = -1
        self.current_start_status = "not_started"
        self.current_target_id = -1
        self.current_target_freq_hz = 0.0
        self.current_trial_mapping: List[Tuple[int, int, float]] = []  # (slot, image_id, freq)
        self.publish_idx = 0
        self.wait_until = 0.0
        self.waiting_start_trial_id = -1
        self.waiting_start_since_mono = 0.0

        os.makedirs(self.save_dir, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mapping_csv_path = os.path.join(self.save_dir, f"ssvep_image_mapping_{run_stamp}.csv")
        self.trials_csv_path = os.path.join(self.save_dir, f"ssvep_trials_{run_stamp}.csv")
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

        self.start_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.start_sock.bind((self.start_bind_ip, self.start_port))
        self.start_sock.setblocking(False)

        self.timer = self.create_timer(self.image_period, self._on_timer)
        self.get_logger().info(
            "CentralControllerSSVEPNode started: "
            f"topic={image_topic}, qos={'RELIABLE' if self.use_reliable_qos else 'BEST_EFFORT'}, "
            f"trial_duration_s={self.trial_duration_s:.3f}, max_trials={self.max_trials}, "
            f"start_udp={self.start_bind_ip}:{self.start_port}, "
            f"start_wait_timeout_s={self.start_wait_timeout_s:.2f}, "
            f"save_dir={self.save_dir}"
        )
        self.get_logger().info(
            f"saving files: trials_csv={self.trials_csv_path}, mapping_csv={self.mapping_csv_path}"
        )

    def _read_image_bgr(self, path: str) -> Optional[np.ndarray]:
        if not os.path.isfile(path) or PILImage is None:
            return None
        try:
            img = PILImage.open(path).convert("RGB").resize((self.image_w, self.image_h))
            rgb = np.asarray(img, dtype=np.uint8)
            return rgb[:, :, ::-1].copy()
        except Exception:
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
            img[y0 : y0 + box, x0 : x0 + box, :] = 255
            out.append(img)
        return out

    def _load_or_generate_images(self, n: int) -> List[np.ndarray]:
        candidates = []
        if self.image_paths:
            candidates = [p for p in self.image_paths if p]
        else:
            for pat in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                candidates.extend(glob.glob(os.path.join(self.image_dir, pat)))
            candidates = sorted(candidates)

        frames = []
        for p in candidates:
            frame = self._read_image_bgr(p)
            if frame is not None:
                frames.append(frame)
            if len(frames) >= n:
                break
        if len(frames) < n:
            frames.extend(self._generate_placeholders(n - len(frames)))
        return frames[:n]

    def _to_ros_image(self, bgr: np.ndarray, img_idx_1based: int, image_id: int, freq: float) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"trial={self.trial_idx};img={img_idx_1based};image_id={image_id};target={self.current_target_id};"
            f"freq={self.current_target_freq_hz:.3f};dur={self.trial_duration_s:.3f}"
        )
        msg.height = int(bgr.shape[0])
        msg.width = int(bgr.shape[1])
        msg.encoding = "bgr8"
        msg.step = int(bgr.shape[1] * 3)
        msg.data = bgr.tobytes()
        return msg

    def _publish_stop_signal(self, trial_id: int) -> None:
        # Control message for Unity: stop current trial visuals immediately.
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"trial={trial_id};cmd=stop"
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.image_pub.publish(msg)

    def _prepare_new_trial(self) -> None:
        if self.max_trials > 0 and self.trial_idx >= self.max_trials:
            self.state = "done"
            self.get_logger().info(f"max_trials={self.max_trials} reached. stop new trials.")
            return

        self.trial_idx += 1
        self.publish_idx = 0
        self.state = "publishing"
        self.current_trial_start_mono = 0.0
        self.current_trial_prepared_wall = datetime.now().isoformat(timespec="milliseconds")
        self.current_trial_start_wall = ""
        self.current_start_trial_id = -1
        self.current_start_status = "not_started"
        self.current_target_id = random.randint(1, self.num_targets)
        self.current_target_freq_hz = self.ssvep_frequencies[self.current_target_id - 1]

        # Shuffle image->slot mapping each trial, keep slot->frequency fixed for dynamic image slots.
        order = list(range(self.num_images))
        random.shuffle(order)
        self.current_trial_mapping = []
        for slot in range(self.num_images):
            img_idx = order[slot]
            image_id = self.base_image_ids[img_idx]
            freq = self.ssvep_frequencies[slot]
            self.current_trial_mapping.append((slot + 1, image_id, freq))

        for slot_id, image_id, freq in self.current_trial_mapping:
            self.mapping_writer.writerow(
                [self.trial_idx, slot_id, image_id, f"{freq:.3f}", self.current_trial_prepared_wall]
            )
        self.mapping_csv_file.flush()
        self.get_logger().info(
            f"[Trial {self.trial_idx}] prepared target={self.current_target_id} "
            f"freq={self.current_target_freq_hz:.3f}Hz, mapping written, start publishing images"
        )

    def _poll_trial_started(self) -> int:
        # Return started trial_id, or -1 if no valid signal available.
        while True:
            try:
                payload, addr = self.start_sock.recvfrom(128)
            except BlockingIOError:
                return -1
            except OSError:
                return -1
            try:
                text = payload.decode("utf-8", errors="ignore").strip().lower()
            except Exception:
                continue
            self.get_logger().info(f"start udp recv from {addr}: '{text}'")
            if not text.startswith("trial_started="):
                continue
            value = text.split("=", 1)[1].strip()
            try:
                return int(value)
            except ValueError:
                continue

    def _finalize_trial_and_prepare_next(self) -> None:
        end_wall = datetime.now().isoformat(timespec="milliseconds")
        actual_duration_s = 0.0
        if self.current_trial_start_mono > 0.0:
            actual_duration_s = max(0.0, time.monotonic() - self.current_trial_start_mono)

        self.trials_writer.writerow(
            [
                self.trial_idx,
                self.current_trial_prepared_wall,
                self.current_trial_start_wall,
                end_wall,
                f"{self.trial_duration_s:.3f}",
                f"{actual_duration_s:.3f}",
                self.current_start_trial_id if self.current_start_trial_id > 0 else "",
                self.current_start_status,
            ]
        )
        self.trials_csv_file.flush()
        self.get_logger().info(
            f"[Trial {self.trial_idx}] completed, start_status={self.current_start_status}, "
            f"start_trial_id={self.current_start_trial_id}"
        )
        self.state = "waiting"
        self.wait_until = time.monotonic() + max(0.0, self.inter_trial_interval)

    def _on_timer(self) -> None:
        now = time.monotonic()

        if self.state == "done":
            return

        if self.state == "init_wait":
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

        if self.state == "publishing":
            slot_id, image_id, freq = self.current_trial_mapping[self.publish_idx]
            frame = self.base_images[image_id - 1]
            self.image_pub.publish(self._to_ros_image(frame, slot_id, image_id, freq))
            self.publish_idx += 1
            if self.publish_idx >= self.num_images:
                self.state = "wait_start"
                self.waiting_start_trial_id = self.trial_idx
                self.waiting_start_since_mono = now
                self.get_logger().info(
                    f"[Trial {self.trial_idx}] image batch complete, wait trial_started"
                )
            return

        if self.state == "wait_start":
            started_trial = self._poll_trial_started()
            if started_trial == self.waiting_start_trial_id:
                self.state = "stimulating"
                self.current_trial_start_mono = now
                self.current_trial_start_wall = datetime.now().isoformat(timespec="milliseconds")
                self.current_start_trial_id = started_trial
                self.current_start_status = "started"
                self.get_logger().info(
                    f"[Trial {self.trial_idx}] trial_started received, start timing "
                    f"{self.trial_duration_s:.2f}s"
                )
                return

            timed_out = (
                self.start_wait_timeout_s > 0.0
                and now - self.waiting_start_since_mono >= self.start_wait_timeout_s
            )
            if timed_out:
                self.get_logger().warning(
                    f"[Trial {self.waiting_start_trial_id}] start wait timeout "
                    f"after {self.start_wait_timeout_s:.2f}s, force start timing"
                )
                self.state = "stimulating"
                self.current_trial_start_mono = now
                self.current_trial_start_wall = datetime.now().isoformat(timespec="milliseconds")
                self.current_start_trial_id = -1
                self.current_start_status = "timeout_force_start"
                return

        if self.state == "stimulating":
            if now - self.current_trial_start_mono >= self.trial_duration_s:
                self._publish_stop_signal(self.trial_idx)
                self._finalize_trial_and_prepare_next()
                return


def main(args=None):
    rclpy.init(args=args)
    node = CentralControllerSSVEPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.start_sock.close()
        except Exception:
            pass
        for obj in [
            getattr(node, "mapping_csv_file", None),
            getattr(node, "trials_csv_file", None),
        ]:
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
