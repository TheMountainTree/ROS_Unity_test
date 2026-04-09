#!/usr/bin/env python3
"""Reasoner-related behaviors for SSVEP communication node (Node4_test).

Extended from reasoner_1.py for Node4_test integration.
Note: EEG decoding is handled in decode_2_test.py, this module handles
the selection logic and reasoner communication.
"""

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

    @staticmethod
    def _normalize_reasoner_batch_count(raw_count: str, default_count: int = 6) -> int:
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = default_count
        if count <= 0:
            count = default_count
        return max(1, min(6, count))

    def _reset_reasoner_building_state(self) -> None:
        self.reasoner_building_group_id = -1
        self.reasoner_building_images = {}
        self.reasoner_building_meta = {}
        self.reasoner_expected_group_count = 6
        self.reasoner_batch_end_received = False

    def _finalize_reasoner_group_if_ready(self, group_id: int) -> bool:
        expected = int(getattr(self, "reasoner_expected_group_count", 6))
        if len(self.reasoner_building_images) < expected:
            return False

        missing_indices = [idx for idx in range(expected) if idx not in self.reasoner_building_images]
        if missing_indices:
            self.get_logger().warning(
                f"reasoner group={group_id} missing indices={missing_indices}, "
                f"received={sorted(self.reasoner_building_images.keys())}"
            )
            return False

        batch = []
        for idx in range(expected):
            item_meta = self.reasoner_building_meta[idx]
            batch.append(
                {
                    "group": group_id,
                    "index": idx,
                    "source_path": item_meta["source_path"],
                    "stage": item_meta.get("stage", "object"),
                    "page": item_meta.get("page", "0"),
                    "item_uid": item_meta.get("item_uid", ""),
                    "item_label": item_meta.get("item_label", ""),
                    "image": self.reasoner_building_images[idx],
                }
            )
        self.ready_reasoner_batches.append(batch)
        self._reset_reasoner_building_state()
        self.get_logger().info(
            f"received reasoner batch group={group_id}, size={expected}, "
            f"queued_batches={len(self.ready_reasoner_batches)}"
        )
        return True

    def _on_reasoner_image(self, msg: Image) -> None:
        if not self.reasoner_mode_enabled or self.run_mode != "decode":
            return

        if not hasattr(self, "reasoner_expected_group_count"):
            self.reasoner_expected_group_count = 6
            self.reasoner_batch_end_received = False

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
            self._reset_reasoner_building_state()
            return

        if cmd == "reuse_page":
            if self.current_reasoner_group_images:
                stage = self._current_reasoner_stage()
                self.get_logger().info(
                    f"reasoner reuse_page received, stage={stage}, "
                    "restart prepare with current batch"
                )
                self._start_next_decode_trial_with_current_images()
            else:
                self.get_logger().warning(
                    "reasoner reuse_page received but no current batch to reuse"
                )
            return

        if cmd in ("batch_start", "batch_end"):
            try:
                group_id = int(meta.get("group", "-1"))
            except ValueError:
                self.get_logger().warning(f"invalid reasoner batch command frame_id: {msg.header.frame_id}")
                return
            if group_id < 0:
                self.get_logger().warning(f"invalid reasoner batch group={group_id}")
                return

            count = self._normalize_reasoner_batch_count(meta.get("count", "6"), default_count=6)
            if cmd == "batch_start":
                if (
                    self.reasoner_building_group_id != -1
                    and self.reasoner_building_group_id != group_id
                    and self.reasoner_building_images
                ):
                    self.get_logger().warning(
                        f"dropping incomplete reasoner group={self.reasoner_building_group_id}, "
                        f"received={len(self.reasoner_building_images)}"
                    )
                self.reasoner_building_group_id = group_id
                self.reasoner_building_images = {}
                self.reasoner_building_meta = {}
                self.reasoner_expected_group_count = count
                self.reasoner_batch_end_received = False
                return

            # cmd=batch_end
            if self.reasoner_building_group_id == -1:
                self.reasoner_building_group_id = group_id
            if self.reasoner_building_group_id != group_id:
                self.get_logger().warning(
                    f"ignore batch_end group={group_id}, "
                    f"current_building_group={self.reasoner_building_group_id}"
                )
                return
            self.reasoner_expected_group_count = count
            self.reasoner_batch_end_received = True
            if not self._finalize_reasoner_group_if_ready(group_id):
                self.get_logger().warning(
                    f"reasoner group={group_id} batch_end before enough images, "
                    f"received={len(self.reasoner_building_images)}, expected={self.reasoner_expected_group_count}"
                )
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
            self.reasoner_expected_group_count = self._normalize_reasoner_batch_count(
                meta.get("count", "6"),
                default_count=6,
            )
            self.reasoner_batch_end_received = False

        bgr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
            (int(msg.height), int(msg.width), 3)
        ).copy()
        self.reasoner_building_images[image_index] = bgr
        self.reasoner_building_meta[image_index] = {
            "group": group_id,
            "index": image_index,
            "source_path": meta.get("image_path", ""),
            "stage": meta.get("stage", "object"),
            "page": meta.get("page", "0"),
            "item_uid": meta.get("item_uid", ""),
            "item_label": meta.get("item_label", ""),
        }

        if self._finalize_reasoner_group_if_ready(group_id):
            return

        end_flag = meta.get("end", "0") == "1"
        if end_flag:
            if not self._finalize_reasoner_group_if_ready(group_id):
                self.get_logger().warning(
                    f"reasoner group={group_id} ended before expected count, "
                    f"received={len(self.reasoner_building_images)}, expected={self.reasoner_expected_group_count}"
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
            f"activate reasoner batch group={batch[0]['group']}, "
            f"stage={batch[0].get('stage', 'object')}, size={len(batch)}"
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
        """Poll mock_selected_index parameter (for fallback/debug mode)."""
        value = self._param_int("mock_selected_index")
        if value == -1:
            return
        self.pending_mock_selection = value
        self.set_parameters([Parameter("mock_selected_index", value=-1)])
        self.get_logger().info(
            f"cached mock_selected_index={value}, current_state={self.state}"
        )

    def _consume_cached_mock_selection(self) -> int:
        """Consume cached mock selection (for fallback/debug mode)."""
        value = self.pending_mock_selection
        self.pending_mock_selection = -1
        return value

    def _start_next_decode_trial_with_current_images(self) -> None:
        self.state = NodeState.WAITING
        self.state_until = time.monotonic() + max(0.0, self.decode_iti)

    def _pop_history_item_by_id(self, history_id: int) -> Optional[Dict[str, object]]:
        for idx in range(len(self.history_stack) - 1, -1, -1):
            item = self.history_stack[idx]
            if int(item.get("history_id", -1)) == int(history_id):
                return self.history_stack.pop(idx)
        return None

    def _current_reasoner_stage(self) -> str:
        if not self.current_reasoner_group_images:
            return "object"
        return str(self.current_reasoner_group_images[0].get("stage", "object"))

    @staticmethod
    def _is_same_stage_item(a: Optional[Dict[str, object]], b: Optional[Dict[str, object]]) -> bool:
        if a is None or b is None:
            return False
        return (
            str(a.get("stage", "")) == str(b.get("stage", ""))
            and str(a.get("item_uid", "")) == str(b.get("item_uid", ""))
        )

    def _handle_reasoner_selection(self, selection: int) -> None:
        """Handle user selection in reasoner mode.

        Args:
            selection: Slot index (0, 1, 2, 4, 5, 6 for images, 3 for confirm, 7 for undo)
        """
        if selection not in range(8):
            self.get_logger().warning(f"ignore invalid selection={selection}")
            return

        if selection in (0, 1, 2, 4, 5, 6):
            selected = self._slot_to_group_image(selection)
            if selected is None:
                self.get_logger().warning(
                    f"no current group image for selection slot={selection}"
                )
                return
            stage = str(selected.get("stage", "object"))
            last_history = self.history_stack[-1] if self.history_stack else None
            if self._is_same_stage_item(selected, last_history):
                self.get_logger().info(
                    f"duplicate selection skipped: stage={stage}, "
                    f"item_uid={selected.get('item_uid', '')}; restart prepare with current batch"
                )
                self._start_next_decode_trial_with_current_images()
                return
            self.next_history_id += 1
            history_item: Optional[Dict[str, object]] = {
                "history_id": self.next_history_id,
                "selection_slot": selection,
                "image": selected["image"],
                "source_path": selected.get("source_path", ""),
                "group": selected.get("group", -1),
                "index": selected.get("index", -1),
                "stage": stage,
                "page": selected.get("page", "0"),
                "item_uid": selected.get("item_uid", ""),
                "item_label": selected.get("item_label", ""),
            }
            self.history_stack.append(history_item)
            self.reasoner_action_stack.append(
                {
                    "type": "selection",
                    "history_id": history_item["history_id"],
                    "stage": stage,
                    "page": selected.get("page", "0"),
                    "item_uid": selected.get("item_uid", ""),
                    "item_label": selected.get("item_label", ""),
                    "slot": selection,
                }
            )
            self._publish_history_image_msg(
                history_item["image"], history_item["history_id"]
            )
            self._publish_reasoner_selection(selection, selected, history_item)

            if stage != "object":
                self.current_reasoner_group_images = []
            self.state = NodeState.REASONER_WAIT_BATCH
            self.get_logger().info(
                f"selection slot={selection} accepted, stage={stage}, "
                f"history_size={len(self.history_stack)}, waiting reasoner response"
            )
            return

        if selection == 3:
            self.reasoner_action_stack.append(
                {
                    "type": "confirm",
                    "stage": self._current_reasoner_stage(),
                    "page": (
                        self.current_reasoner_group_images[0].get("page", "0")
                        if self.current_reasoner_group_images
                        else "0"
                    ),
                }
            )
            self._publish_reasoner_cmd("confirm")
            self.get_logger().info(
                f"selection slot=3 (confirm), stage={self._current_reasoner_stage()}, "
                "waiting reasoner next batch"
            )
            self.state = NodeState.REASONER_WAIT_BATCH
            return

        if selection == 7:
            if not self.reasoner_action_stack:
                self.get_logger().info(
                    "selection slot=7 with empty action stack: restart flashing current page"
                )
                self._start_next_decode_trial_with_current_images()
                return

            current_stage = self._current_reasoner_stage()
            last_action = self.reasoner_action_stack.pop()
            action_type = str(last_action.get("type", "")).strip().lower()

            if action_type == "confirm":
                self._publish_reasoner_cmd("rollback")
                self.current_reasoner_group_images = []
                self.state = NodeState.REASONER_WAIT_BATCH
                self.get_logger().info(
                    f"selection slot=7 (undo confirm -> rollback), stage={current_stage}, "
                    "waiting reasoner previous batch"
                )
                return

            if action_type != "selection":
                self.get_logger().warning(
                    f"selection slot=7 ignored because unsupported action type='{action_type}'"
                )
                return

            history_id = int(last_action.get("history_id", -1))
            removed_item = self._pop_history_item_by_id(history_id)
            if removed_item is None and self.history_stack:
                removed_item = self.history_stack.pop()

            if removed_item is not None:
                self._send_history_udp_command({"cmd": "delete_last"})
            self._publish_reasoner_undo_selection(last_action)

            self.current_reasoner_group_images = []
            self.state = NodeState.REASONER_WAIT_BATCH
            if removed_item is not None:
                self.get_logger().info(
                    f"selection slot=7 (undo selection), stage={current_stage}, "
                    f"deleted history_id={removed_item['history_id']}, "
                    f"history_size={len(self.history_stack)}, waiting reasoner previous batch"
                )
            else:
                self.get_logger().info(
                    f"selection slot=7 (undo selection), stage={current_stage}, "
                    "history item missing, waiting reasoner previous batch"
                )
            return

    def _publish_reasoner_selection(
        self,
        slot: int,
        selected: Dict[str, object],
        history_item: Optional[Dict[str, object]] = None,
    ) -> None:
        """Publish selection result to reasoner."""
        history_id = -1 if history_item is None else int(history_item.get("history_id", -1))
        source_path = selected.get("source_path", "")
        stage = selected.get("stage", "object")
        page = selected.get("page", "0")
        item_uid = selected.get("item_uid", "")
        item_label = selected.get("item_label", "")
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            f"cmd=selection;slot={slot};"
            f"history_id={history_id};"
            f"group={selected.get('group', -1)};"
            f"index={selected.get('index', -1)};"
            f"source_path={os.path.basename(str(source_path))};"
            f"stage={stage};"
            f"page={page};"
            f"item_uid={item_uid};"
            f"item_label={item_label}"
        )
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.reasoner_pub.publish(msg)

    def _publish_reasoner_undo_selection(self, action: Dict[str, object]) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            "cmd=undo_selection;"
            f"slot={int(action.get('slot', -1))};"
            f"history_id={int(action.get('history_id', -1))};"
            f"stage={action.get('stage', 'object')};"
            f"page={action.get('page', '0')};"
            f"item_uid={action.get('item_uid', '')};"
            f"item_label={action.get('item_label', '')}"
        )
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.reasoner_pub.publish(msg)
