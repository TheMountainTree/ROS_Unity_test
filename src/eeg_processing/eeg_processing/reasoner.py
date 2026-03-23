#!/usr/bin/env python3
"""Reasoner-related behaviors for SSVEP communication node."""

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from .utils import NodeState

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class ReasonerModule:
    """Mix-in that encapsulates reasoner image/selection workflow."""

    @staticmethod
    def _parse_frame_id(frame_id: str) -> Dict[str, str]:
        meta: Dict[str, str] = {}
        if not frame_id:
            return meta
        for part in frame_id.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            meta[key.strip()] = value.strip()
        return meta

    def _on_reasoner_image(self, msg: Image) -> None:
        if not self.reasoner_mode_enabled or self.run_mode != "decode":
            return

        meta = self._parse_frame_id(msg.header.frame_id)
        cmd = meta.get("cmd", "")
        if cmd == "reasoner_ready":
            if not self.reasoner_handshake_complete:
                self.reasoner_handshake_complete = True
                self.get_logger().info("reasoner handshake complete")
            return

        if cmd == "done":
            self.get_logger().info(
                f"reasoner done, selection finished, history_size={len(self.history_stack)}"
            )
            self.state = NodeState.INIT_WAIT
            self.state_until = time.monotonic() + self.startup_delay
            self.current_reasoner_group_images = []
            self.ready_reasoner_batches = []
            self.reasoner_building_group_id = -1
            self.reasoner_building_images = {}
            self.reasoner_building_meta = {}
            return

        if msg.encoding != "bgr8" or msg.width <= 0 or msg.height <= 0:
            return

        try:
            group_id = int(meta.get("group", "-1"))
            image_index = int(meta.get("index", "-1"))
        except ValueError:
            self.get_logger().warning(f"invalid reasoner frame_id: {msg.header.frame_id}")
            return
        if image_index < 0 or image_index >= 6:
            self.get_logger().warning(f"invalid reasoner image index={image_index}")
            return

        expected_size = int(msg.width) * int(msg.height) * 3
        if len(msg.data) != expected_size:
            self.get_logger().warning(
                f"invalid reasoner image payload size={len(msg.data)}, expected={expected_size}"
            )
            return

        if self.reasoner_building_group_id != group_id:
            if self.reasoner_building_images:
                self.get_logger().warning(
                    f"dropping incomplete reasoner group={self.reasoner_building_group_id}, "
                    f"received={len(self.reasoner_building_images)}"
                )
            self.reasoner_building_group_id = group_id
            self.reasoner_building_images = {}
            self.reasoner_building_meta = {}

        bgr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
            (int(msg.height), int(msg.width), 3)
        ).copy()
        self.reasoner_building_images[image_index] = bgr
        self.reasoner_building_meta[image_index] = {
            "group": group_id,
            "index": image_index,
            "source_path": meta.get("image_path", ""),
        }

        end_flag = meta.get("end", "0") == "1"
        if len(self.reasoner_building_images) == 6 or end_flag:
            if len(self.reasoner_building_images) != 6:
                self.get_logger().warning(
                    f"reasoner group={group_id} ended before 6 images, received={len(self.reasoner_building_images)}"
                )
                return
            batch = []
            for idx in range(6):
                batch.append(
                    {
                        "group": group_id,
                        "index": idx,
                        "source_path": self.reasoner_building_meta[idx]["source_path"],
                        "image": self.reasoner_building_images[idx],
                    }
                )
            self.ready_reasoner_batches.append(batch)
            self.reasoner_building_group_id = -1
            self.reasoner_building_images = {}
            self.reasoner_building_meta = {}
            self.get_logger().info(
                f"received reasoner batch group={group_id}, queued_batches={len(self.ready_reasoner_batches)}"
            )

    def _publish_reasoner_cmd(self, cmd: str) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"cmd={cmd}"
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.reasoner_pub.publish(msg)

    def _publish_history_image_msg(self, image: np.ndarray, history_id: int) -> None:
        history_img = image
        if (
            image.shape[1] != self.history_image_width
            or image.shape[0] != self.history_image_height
        ):
            if PILImage is not None:
                rgb = image[:, :, ::-1]
                pil_img = PILImage.fromarray(rgb, mode="RGB")
                pil_img = pil_img.resize(
                    (self.history_image_width, self.history_image_height)
                )
                resized_rgb = np.asarray(pil_img, dtype=np.uint8)
                history_img = resized_rgb[:, :, ::-1].copy()
            else:
                self.get_logger().warning(
                    "PIL unavailable; history image will be published without resize"
                )
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"hist_id={history_id}"
        msg.height = int(history_img.shape[0])
        msg.width = int(history_img.shape[1])
        msg.encoding = "bgr8"
        msg.step = int(history_img.shape[1] * 3)
        msg.data = history_img.tobytes()
        self.history_pub.publish(msg)

    def _send_history_udp_command(self, payload: Dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        try:
            self.history_udp_sock.sendto(
                data, (self.history_udp_target_ip, self.history_udp_target_port)
            )
        except Exception as e:
            self.get_logger().warning(f"send history udp command failed: {e}")

    def _push_ready_reasoner_batch(self) -> bool:
        if not self.ready_reasoner_batches:
            return False
        batch = self.ready_reasoner_batches.pop(0)
        self.current_reasoner_group_images = batch
        self.base_images = [item["image"] for item in batch]
        self.base_image_ids = list(range(1, len(batch) + 1))
        self.get_logger().info(
            f"activate reasoner batch group={batch[0]['group']}, size={len(batch)}"
        )
        return True

    def _slot_to_group_image(self, slot_index: int) -> Optional[Dict[str, object]]:
        slot_map = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
        group_idx = slot_map.get(slot_index)
        if group_idx is None:
            return None
        if group_idx >= len(self.current_reasoner_group_images):
            return None
        return self.current_reasoner_group_images[group_idx]

    def _poll_mock_selected_index(self) -> None:
        value = self._param_int("mock_selected_index")
        if value == -1:
            return
        self.pending_mock_selection = value
        self.set_parameters([Parameter("mock_selected_index", value=-1)])
        self.get_logger().info(
            f"cached mock_selected_index={value}, current_state={self.state}"
        )

    def _consume_cached_mock_selection(self) -> int:
        value = self.pending_mock_selection
        self.pending_mock_selection = -1
        return value

    def _start_next_decode_trial_with_current_images(self) -> None:
        self.state = NodeState.WAITING
        self.state_until = time.monotonic() + max(0.0, self.decode_iti)

    def _handle_reasoner_selection(self, selection: int) -> None:
        """Handle user selection in reasoner mode."""
        if selection not in range(8):
            self.get_logger().warning(f"ignore invalid mock_selected_index={selection}")
            return

        if selection in (0, 1, 2, 4, 5, 6):
            selected = self._slot_to_group_image(selection)
            if selected is None:
                self.get_logger().warning(
                    f"no current group image for selection slot={selection}"
                )
                return
            self.next_history_id += 1
            history_item = {
                "history_id": self.next_history_id,
                "selection_slot": selection,
                "image": selected["image"],
                "source_path": selected.get("source_path", ""),
                "group": selected.get("group", -1),
                "index": selected.get("index", -1),
            }
            self.history_stack.append(history_item)
            self._publish_history_image_msg(
                history_item["image"], history_item["history_id"]
            )
            self._publish_reasoner_selection(selection, history_item)

            self.current_reasoner_group_images = []
            self.state = NodeState.REASONER_WAIT_BATCH
            self.get_logger().info(
                f"selection slot={selection} accepted, history_id={history_item['history_id']}, "
                f"history_size={len(self.history_stack)}, waiting next reasoner batch"
            )
            return

        if selection == 3:
            self._publish_reasoner_cmd("confirm")
            self.get_logger().info(
                f"selection slot=3 (confirm), history_size={len(self.history_stack)}, "
                "waiting for reasoner done"
            )
            self.state = NodeState.WAITING
            self.state_until = 0.0
            return

        if selection == 7:
            if not self.history_stack:
                self.get_logger().warning(
                    "selection slot=7 ignored because history is empty"
                )
                return
            deleted_item = self.history_stack.pop()
            self._send_history_udp_command({"cmd": "delete_last"})
            self._publish_reasoner_cmd("rollback")

            self.current_reasoner_group_images = []
            self.state = NodeState.REASONER_WAIT_BATCH
            self.get_logger().info(
                f"selection slot=7 (rollback), deleted history_id={deleted_item['history_id']}, "
                f"history_size={len(self.history_stack)}, waiting reasoner to resend previous group"
            )

    def _publish_reasoner_selection(self, slot: int, history_item: Dict[str, object]) -> None:
        """Publish selection result to reasoner."""
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"cmd=selection;slot={slot};"
            f"history_id={history_item['history_id']};"
            f"group={history_item.get('group', -1)};"
            f"index={history_item.get('index', -1)};"
            f"source_path={os.path.basename(str(history_item.get('source_path', '')))}"
        )
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.reasoner_pub.publish(msg)
