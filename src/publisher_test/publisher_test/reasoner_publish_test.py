#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class ReasonerPublishTestNode(Node):
    """Publish 24 local images in 4 batches and listen for SSVEP feedback."""

    def __init__(self):
        super().__init__("reasoner_publish_test")
        desc = lambda text: ParameterDescriptor(description=text)

        self.declare_parameter(
            "image_dir",
            os.path.expanduser("~/Pictures/截图"),
            descriptor=desc("图片目录，按自然排序取前 24 张。"),
        )
        self.declare_parameter(
            "input_topic",
            "/reasoner/images",
            descriptor=desc("发给 SSVEP 节点的图片批次话题。"),
        )
        self.declare_parameter(
            "output_topic",
            "/reasoner/feedback",
            descriptor=desc("接收 SSVEP 回传命令/历史图片的话题。"),
        )
        self.declare_parameter(
            "image_width",
            640,
            descriptor=desc("统一输出图片宽度。"),
        )
        self.declare_parameter(
            "image_height",
            480,
            descriptor=desc("统一输出图片高度。"),
        )

        self.image_dir = self.get_parameter("image_dir").value
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)

        self.publisher_ = self.create_publisher(Image, self.input_topic, 10)
        self.feedback_sub_ = self.create_subscription(
            Image, self.output_topic, self.feedback_callback, 10
        )

        self.groups = self._load_groups()
        self.current_group_idx = -1
        self.pending_history: List[Image] = []
        self.handshake_complete = False
        self.handshake_sent = False

        self.get_logger().info(
            f"reasoner_publish_test ready: groups={len(self.groups)}, "
            f"input_topic={self.input_topic}, output_topic={self.output_topic}"
        )

    @staticmethod
    def _natural_sort_key(path: str):
        basename = os.path.basename(path)
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", basename)]

    def _collect_image_paths(self, source_dir: str) -> List[str]:
        source_dir = os.path.expanduser(source_dir)
        if not os.path.isdir(source_dir):
            return []

        paths: List[str] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            paths.extend(str(p) for p in Path(source_dir).glob(pattern))
            paths.extend(str(p) for p in Path(source_dir).glob(pattern.upper()))
        return sorted(set(paths), key=self._natural_sort_key)

    def _read_image_bgr(self, path: str) -> Optional[np.ndarray]:
        if PILImage is None:
            self.get_logger().error("PIL is unavailable; cannot load local images.")
            return None
        try:
            img = (
                PILImage.open(path)
                .convert("RGB")
                .resize((self.image_width, self.image_height))
            )
            rgb = np.asarray(img, dtype=np.uint8)
            bgr = rgb[:, :, ::-1].copy()
            bgr = np.flipud(bgr).copy()
            return bgr
        except Exception as exc:
            self.get_logger().warning(f"failed to read image {path}: {exc}")
            return None

    def _load_groups(self) -> List[List[Dict[str, object]]]:
        image_paths = self._collect_image_paths(self.image_dir)[:24]
        if len(image_paths) < 24:
            raise RuntimeError(
                f"reasoner image source requires at least 24 images, found {len(image_paths)} in {self.image_dir}"
            )

        groups: List[List[Dict[str, object]]] = []
        for group_idx in range(4):
            group: List[Dict[str, object]] = []
            for local_idx, path in enumerate(image_paths[group_idx * 6 : (group_idx + 1) * 6]):
                bgr = self._read_image_bgr(path)
                if bgr is None:
                    raise RuntimeError(f"failed to load required image: {path}")
                group.append(
                    {
                        "group": group_idx,
                        "index": local_idx,
                        "path": path,
                        "image": bgr,
                    }
                )
            groups.append(group)
        return groups

    def _make_image_msg(
        self,
        image: np.ndarray,
        frame_id: str,
    ) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "bgr8"
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        return msg

    def _publish_cmd(self, cmd: str) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"cmd={cmd}"
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.publisher_.publish(msg)

    def publish_group(self, group_idx: int) -> None:
        if group_idx < 0 or group_idx >= len(self.groups):
            self.get_logger().warning(f"request out-of-range group={group_idx}")
            return
        group = self.groups[group_idx]
        self.current_group_idx = group_idx
        for item in group:
            frame_id = (
                f"source=reasoner;group={group_idx};index={item['index']};"
                f"image_path={os.path.basename(str(item['path']))};"
                f"end={1 if int(item['index']) == len(group) - 1 else 0}"
            )
            self.publisher_.publish(self._make_image_msg(item["image"], frame_id))
        self.get_logger().info(f"published group {group_idx + 1}/{len(self.groups)}")

    def _parse_frame_id(self, frame_id: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not frame_id:
            return out
        for part in frame_id.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
        return out

    def feedback_callback(self, msg: Image) -> None:
        meta = self._parse_frame_id(msg.header.frame_id)
        cmd = meta.get("cmd", "")
        if cmd == "ssvep_ready":
            if not self.handshake_sent:
                self._publish_cmd("reasoner_ready")
                self.handshake_sent = True
                self.handshake_complete = True
                self.get_logger().info("received ssvep_ready, sent reasoner_ready")
                self.publish_group(0)
            return
        if cmd == "request_next_group":
            if not self.handshake_complete:
                self.get_logger().warning("ignore request_next_group before handshake")
                return
            next_group = self.current_group_idx + 1
            if next_group >= len(self.groups):
                self.get_logger().info("received request_next_group but no more groups are available")
                return
            self.publish_group(next_group)
            return

        if meta.get("kind") != "history_return":
            return

        self.pending_history.append(msg)
        end_flag = meta.get("end", "0") == "1"
        if end_flag:
            self.get_logger().info(
                f"received history bundle with {len(self.pending_history)} images"
            )
            self.pending_history.clear()


def main(args=None):
    rclpy.init(args=args)
    node = ReasonerPublishTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
