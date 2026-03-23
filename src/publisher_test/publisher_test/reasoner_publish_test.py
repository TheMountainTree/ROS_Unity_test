#!/usr/bin/env python3
"""Reasoner node for SSVEP image selection workflow.

This node manages image groups and handles user selection feedback from SSVEP.

Protocol:
- Reasoner -> SSVEP: Image groups (6 images per group)
- Reasoner -> SSVEP: cmd=done (selection finished)
- SSVEP -> Reasoner: cmd=selection;slot=X (user selected an image)
- SSVEP -> Reasoner: cmd=rollback (user wants to redo previous selection)
- SSVEP -> Reasoner: cmd=confirm (user confirmed all selections)

Workflow:
1. Reasoner sends first group of 6 images
2. User selects an image (slot 0,1,2,4,5,6) -> Reasoner records and sends next group
3. User selects rollback (slot 7) -> Reasoner resends previous group
4. User selects confirm (slot 3) -> Reasoner prints all selections and sends cmd=done
"""

import os
import re
from dataclasses import dataclass, field
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


@dataclass
class SelectionRecord:
    """Record of a user selection."""
    group_idx: int
    image_index: int
    slot: int
    source_path: str
    history_id: int = -1


class ReasonerPublishTestNode(Node):
    """Manage image groups and handle user selection feedback from SSVEP."""

    def __init__(self):
        super().__init__("reasoner_publish_test")
        desc = lambda text: ParameterDescriptor(description=text)

        self.declare_parameter(
            "image_dir",
            os.path.expanduser("~/图片/截图"),
            descriptor=desc("Image directory, naturally sorted, first 24 images used."),
        )
        self.declare_parameter(
            "input_topic",
            "/reasoner/images",
            descriptor=desc("Topic to send image batches to SSVEP."),
        )
        self.declare_parameter(
            "output_topic",
            "/reasoner/feedback",
            descriptor=desc("Topic to receive SSVEP feedback commands."),
        )
        self.declare_parameter(
            "image_width",
            640,
            descriptor=desc("Output image width."),
        )
        self.declare_parameter(
            "image_height",
            480,
            descriptor=desc("Output image height."),
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

        # Load image groups
        self.groups = self._load_groups()
        
        # State management
        self.current_group_idx = -1  # -1 means no group sent yet
        self.selection_history: List[SelectionRecord] = []
        self.group_history: List[int] = []  # Track which groups were shown
        self.handshake_complete = False
        self.handshake_sent = False
        self.finished = False

        self.get_logger().info(
            f"Reasoner node ready: {len(self.groups)} groups, "
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
            bgr = np.flipud(bgr).copy()  # Flip for Unity coordinate system
            return bgr
        except Exception as exc:
            self.get_logger().warning(f"Failed to read image {path}: {exc}")
            return None

    def _load_groups(self) -> List[List[Dict[str, object]]]:
        """Load 24 images into 4 groups of 6 images each."""
        image_paths = self._collect_image_paths(self.image_dir)[:24]
        if len(image_paths) < 24:
            raise RuntimeError(
                f"Reasoner requires at least 24 images, found {len(image_paths)} in {self.image_dir}"
            )

        groups: List[List[Dict[str, object]]] = []
        for group_idx in range(4):
            group: List[Dict[str, object]] = []
            for local_idx, path in enumerate(image_paths[group_idx * 6 : (group_idx + 1) * 6]):
                bgr = self._read_image_bgr(path)
                if bgr is None:
                    raise RuntimeError(f"Failed to load required image: {path}")
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

    def _make_image_msg(self, image: np.ndarray, frame_id: str) -> Image:
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
        """Publish a command message to SSVEP."""
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"cmd={cmd}"
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.publisher_.publish(msg)
        self.get_logger().info(f"Sent command: {cmd}")

    def publish_group(self, group_idx: int) -> None:
        """Publish a group of 6 images to SSVEP."""
        if group_idx < 0 or group_idx >= len(self.groups):
            self.get_logger().warning(f"Requested out-of-range group={group_idx}")
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
        
        self.get_logger().info(f"Published group {group_idx + 1}/{len(self.groups)}")

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
        """Handle feedback from SSVEP node."""
        if self.finished:
            return

        meta = self._parse_frame_id(msg.header.frame_id)
        cmd = meta.get("cmd", "")

        # Handle handshake
        if cmd == "ssvep_ready":
            if not self.handshake_sent:
                self._publish_cmd("reasoner_ready")
                self.handshake_sent = True
                self.handshake_complete = True
                self.get_logger().info("Handshake complete, sending first group")
                self.publish_group(0)
                self.group_history.append(0)
            return

        if not self.handshake_complete:
            self.get_logger().warning("Ignore feedback before handshake")
            return

        # Handle selection
        if cmd == "selection":
            self._handle_selection(meta)
            return

        # Handle rollback
        if cmd == "rollback":
            self._handle_rollback()
            return

        # Handle confirm
        if cmd == "confirm":
            self._handle_confirm()
            return

    def _handle_selection(self, meta: Dict[str, str]) -> None:
        """Handle user image selection."""
        try:
            slot = int(meta.get("slot", "-1"))
            group_idx = int(meta.get("group", "-1"))
            image_index = int(meta.get("index", "-1"))
            history_id = int(meta.get("history_id", "-1"))
            source_path = meta.get("source_path", "")
        except ValueError:
            self.get_logger().warning(f"Invalid selection metadata: {meta}")
            return

        # Record the selection
        record = SelectionRecord(
            group_idx=group_idx,
            image_index=image_index,
            slot=slot,
            source_path=source_path,
            history_id=history_id,
        )
        self.selection_history.append(record)
        
        self.get_logger().info(
            f"Selection recorded: group={group_idx}, index={image_index}, "
            f"slot={slot}, path={source_path}, total_selections={len(self.selection_history)}"
        )

        # Send next group
        next_group = self.current_group_idx + 1
        if next_group >= len(self.groups):
            self.get_logger().info("All groups shown, waiting for user to confirm or rollback")
            return
        
        self.publish_group(next_group)
        self.group_history.append(next_group)

    def _handle_rollback(self) -> None:
        """Handle user rollback request - resend previous group."""
        if not self.selection_history:
            self.get_logger().warning("Rollback requested but no selection history")
            return
        
        # Remove last selection
        removed = self.selection_history.pop()
        self.get_logger().info(
            f"Rollback: removed selection from group={removed.group_idx}, "
            f"remaining_selections={len(self.selection_history)}"
        )

        # Resend the group that was shown when the removed selection was made
        # The user needs to re-select from that group
        rollback_group = removed.group_idx
        self.publish_group(rollback_group)
        
        # Update group history
        if self.group_history:
            self.group_history.pop()  # Remove current group
        self.group_history.append(rollback_group)

    def _handle_confirm(self) -> None:
        """Handle user confirmation - print all selections and finish."""
        self.finished = True
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("USER CONFIRMED SELECTIONS:")
        self.get_logger().info("=" * 60)
        
        if not self.selection_history:
            self.get_logger().info("  No images selected.")
        else:
            for i, record in enumerate(self.selection_history, 1):
                self.get_logger().info(
                    f"  {i}. Group {record.group_idx + 1}, Image {record.image_index + 1}: "
                    f"{record.source_path}"
                )
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Total selections: {len(self.selection_history)}")
        self.get_logger().info("=" * 60)

        # Notify SSVEP that we're done
        self._publish_cmd("done")


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