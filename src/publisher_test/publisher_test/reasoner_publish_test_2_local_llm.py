#!/usr/bin/env python3
"""Reasoner node for multi-stage SSVEP workflow (object -> category -> activity)."""

import json
import os
import re
import textwrap
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String as StringMsg

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except Exception:
    PILImage = None
    ImageDraw = None
    ImageFont = None


# Static code-level switches (non-ROS params by design).
PREFERRED_CAMERA = "camera2"  # Switch to "camera1" if needed.

# Local Qwen2.5-VL model config
QWEN_MODEL_PATH = "/home/frank/workspace/qwen25_vl"
QWEN_MAX_NEW_TOKENS = 512

MAX_BATCH_SIZE = 6
ACTIVITY_CANDIDATE_COUNT = 12
CATEGORY_WORDS = [
    "clean",
    "eat",
    "drink",
    "work",
    "cook",
    "repair",
    "store",
    "move",
    "organize",
    "dispose",
]
SLOT_TO_ITEM_INDEX = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
VALID_SELECTION_SLOTS = set(SLOT_TO_ITEM_INDEX.keys())

STAGE_OBJECT = "object"
STAGE_CATEGORY = "category"
STAGE_ACTIVITY = "activity"


class ReasonerPublishTest2Node(Node):
    """Three-stage reasoner publisher for Node3_1 reasoner loop."""

    def __init__(self):
        super().__init__("reasoner_publish_test_2")
        desc = lambda text: ParameterDescriptor(description=text)

        self.declare_parameter(
            "image_dir",
            str((Path.cwd() / "picture").resolve()),
            descriptor=desc("Picture directory containing camera1/camera2 object crops."),
        )
        self.declare_parameter(
            "input_topic",
            "/reasoner/images",
            descriptor=desc("Topic to send reasoner batches to SSVEP node."),
        )
        self.declare_parameter(
            "output_topic",
            "/reasoner/feedback",
            descriptor=desc("Topic to receive selection/confirm/rollback from SSVEP node."),
        )
        self.declare_parameter("image_width", 640, descriptor=desc("Published image width."))
        self.declare_parameter("image_height", 480, descriptor=desc("Published image height."))

        self.image_dir = os.path.expanduser(str(self.get_parameter("image_dir").value))
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)

        if PILImage is None:
            raise RuntimeError("PIL/Pillow is required for reasoner_publish_test_2")

        self.publisher_ = self.create_publisher(Image, self.input_topic, 10)
        self.llm_stream_pub = self.create_publisher(StringMsg, "/reasoner/llm_stream", 10)
        self.feedback_sub_ = self.create_subscription(
            Image, self.output_topic, self.feedback_callback, 10
        )

        self.text_image_cache: Dict[str, np.ndarray] = {}

        self.object_candidates = self._load_object_candidates(self.image_dir)
        if not self.object_candidates:
            raise RuntimeError(
                f"No valid object candidates loaded from directory: {self.image_dir}"
            )
        self.object_pages = self._paginate(self.object_candidates)
        self.object_uid_to_page: Dict[str, int] = {}
        for page_idx, page in enumerate(self.object_pages):
            for item in page:
                self.object_uid_to_page[str(item["uid"])] = page_idx

        self.category_candidates = self._build_category_candidates()
        self.category_pages = self._paginate(self.category_candidates)
        self.category_uid_to_page = {
            str(item["uid"]): page_idx
            for page_idx, page in enumerate(self.category_pages)
            for item in page
        }

        self.activity_candidates: List[Dict[str, object]] = []
        self.activity_pages: List[List[Dict[str, object]]] = []

        self.stage = STAGE_OBJECT
        self.current_page = 0
        self.current_group_id = -1
        self.next_group_id = 0
        self.current_page_items: List[Dict[str, object]] = []

        self.selected_objects: List[Dict[str, object]] = []
        self.selected_category: Optional[Dict[str, object]] = None
        self.selected_activity: Optional[Dict[str, object]] = None

        self.handshake_complete = False
        self.handshake_sent = False
        self.finished = False

        # Local Qwen model state (parallel loading)
        self._qwen_model = None
        self._qwen_processor = None
        self._model_loading = False
        self._model_loaded = False
        self._model_lock = threading.Lock()

        # Start model loading in background thread (non-blocking)
        self._start_model_loading()

        self.get_logger().info(
            "Reasoner v2 ready: "
            f"objects={len(self.object_candidates)} pages={len(self.object_pages)}, "
            f"categories={len(self.category_candidates)} pages={len(self.category_pages)}, "
            f"preferred_camera={PREFERRED_CAMERA}, "
            f"input_topic={self.input_topic}, output_topic={self.output_topic}"
        )
        self.get_logger().info("Qwen model loading in background...")

    def _start_model_loading(self):
        """Start loading the Qwen model in a background thread."""
        with self._model_lock:
            if self._model_loading or self._model_loaded:
                return
            self._model_loading = True

        def _load_model():
            try:
                import torch
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

                # Set environment to skip HuggingFace Hub verification
                os.environ["HF_HUB_OFFLINE"] = "1"

                self.get_logger().info(f"Loading Qwen2.5-VL model from {QWEN_MODEL_PATH}...")

                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    QWEN_MODEL_PATH,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    local_files_only=True,
                    trust_remote_code=True
                )
                model.eval()

                processor = AutoProcessor.from_pretrained(
                    QWEN_MODEL_PATH,
                    local_files_only=True,
                    trust_remote_code=True
                )

                with self._model_lock:
                    self._qwen_model = model
                    self._qwen_processor = processor
                    self._model_loaded = True
                    self._model_loading = False

                self.get_logger().info("Qwen2.5-VL model loaded successfully")

            except Exception as e:
                self.get_logger().error(f"Failed to load Qwen model: {e}")
                with self._model_lock:
                    self._model_loading = False

        thread = threading.Thread(target=_load_model, daemon=True)
        thread.start()

    @staticmethod
    def _natural_sort_key(path: str):
        basename = os.path.basename(path)
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", basename)]

    @staticmethod
    def _slugify(text: str) -> str:
        lowered = text.strip().lower()
        lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
        lowered = re.sub(r"_+", "_", lowered).strip("_")
        return lowered or "item"

    def _collect_image_paths(self, source_dir: str) -> List[str]:
        if not os.path.isdir(source_dir):
            return []
        paths: List[str] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            paths.extend(str(p) for p in Path(source_dir).glob(pattern))
            paths.extend(str(p) for p in Path(source_dir).glob(pattern.upper()))
        return sorted(set(paths), key=self._natural_sort_key)

    def _read_image_bgr(self, path: str) -> Optional[np.ndarray]:
        try:
            img = PILImage.open(path).convert("RGB").resize((self.image_width, self.image_height))
            rgb = np.asarray(img, dtype=np.uint8)
            bgr = rgb[:, :, ::-1].copy()
            return np.flipud(bgr).copy()
        except Exception as exc:
            self.get_logger().warning(f"Failed to read image {path}: {exc}")
            return None

    def _render_text_card_bgr(self, title: str, subtitle: str = "") -> np.ndarray:
        key = f"{title}||{subtitle}"
        cached = self.text_image_cache.get(key)
        if cached is not None:
            return cached

        img = PILImage.new("RGB", (self.image_width, self.image_height), color=(24, 28, 35))
        draw = ImageDraw.Draw(img)
        pad_x = max(20, int(self.image_width * 0.06))
        pad_y = max(20, int(self.image_height * 0.10))
        max_text_w = max(40, self.image_width - 2 * pad_x)
        max_text_h = max(40, self.image_height - 2 * pad_y)

        def _build_layout(font_obj):
            probe = draw.textbbox((0, 0), "MMMMMMMM", font=font_obj)
            probe_w = max(1, probe[2] - probe[0])
            avg_char_w = max(1, probe_w // 8)
            chars_per_line = max(4, max_text_w // avg_char_w)

            title_lines = textwrap.wrap(title, width=chars_per_line) or [" "]
            subtitle_lines = textwrap.wrap(subtitle, width=chars_per_line) if subtitle else []
            lines = title_lines + ([""] if subtitle_lines else []) + subtitle_lines

            line_heights = []
            line_widths = []
            spacing = max(6, int(getattr(font_obj, "size", 12) * 0.35))
            for line in lines:
                bbox = draw.textbbox((0, 0), line or " ", font=font_obj)
                line_widths.append(max(1, bbox[2] - bbox[0]))
                line_heights.append(max(1, bbox[3] - bbox[1]) + spacing)

            total_h = sum(line_heights)
            max_w = max(line_widths) if line_widths else 0
            return title_lines, subtitle_lines, lines, line_heights, max_w, total_h

        max_candidate = max(14, min(self.image_width, self.image_height))
        lo = 8
        hi = max_candidate
        best_font = ImageFont.load_default()
        best_layout = _build_layout(best_font)

        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                font = ImageFont.load_default(size=mid)
            except TypeError:
                font = ImageFont.load_default()
            layout = _build_layout(font)
            fits = layout[4] <= max_text_w and layout[5] <= max_text_h
            if fits:
                best_font = font
                best_layout = layout
                lo = mid + 1
            else:
                hi = mid - 1

        title_lines, _, lines, line_heights, _, total_h = best_layout
        y = max(10, (self.image_height - total_h) // 2)

        for idx, line in enumerate(lines):
            safe_line = line or " "
            bbox = draw.textbbox((0, 0), safe_line, font=best_font)
            line_w = max(1, bbox[2] - bbox[0])
            x = max(10, (self.image_width - line_w) // 2)
            color = (236, 239, 244) if idx < len(title_lines) else (145, 165, 189)
            draw.text((x, y), safe_line, fill=color, font=best_font)
            y += line_heights[idx]

        rgb = np.asarray(img, dtype=np.uint8)
        bgr = np.flipud(rgb[:, :, ::-1].copy()).copy()
        self.text_image_cache[key] = bgr
        return bgr

    def _parse_object_filename(self, basename: str) -> Optional[Dict[str, str]]:
        stem = Path(basename).stem
        if stem.lower().endswith("_total"):
            return None
        match = re.match(r"^(camera[12])_(\d+)_([^\.]+)$", stem, flags=re.IGNORECASE)
        if not match:
            return None
        camera = match.group(1).lower()
        obj_id = match.group(2)
        raw_name = match.group(3)
        label = raw_name.replace("_", " ").strip()
        uid = f"{obj_id}_{self._slugify(raw_name)}"
        return {
            "camera": camera,
            "obj_id": obj_id,
            "raw_name": raw_name,
            "label": label,
            "uid": uid,
        }

    def _load_object_candidates(self, source_dir: str) -> List[Dict[str, object]]:
        preferred = PREFERRED_CAMERA.lower().strip()
        if preferred not in ("camera1", "camera2"):
            preferred = "camera2"

        selected_by_uid: Dict[str, Dict[str, object]] = {}
        for path in self._collect_image_paths(source_dir):
            basename = os.path.basename(path)
            parsed = self._parse_object_filename(basename)
            if parsed is None:
                self.get_logger().info(f"Skip non-object file: {basename}")
                continue
            bgr = self._read_image_bgr(path)
            if bgr is None:
                continue
            candidate = {
                "uid": parsed["uid"],
                "label": parsed["label"],
                "camera": parsed["camera"],
                "obj_id": int(parsed["obj_id"]),
                "path": path,
                "image": bgr,
            }
            prev = selected_by_uid.get(str(candidate["uid"]))
            if prev is None:
                selected_by_uid[str(candidate["uid"])] = candidate
                continue
            prev_camera = str(prev["camera"])
            new_camera = str(candidate["camera"])
            if prev_camera != preferred and new_camera == preferred:
                selected_by_uid[str(candidate["uid"])] = candidate

        objects = list(selected_by_uid.values())
        objects.sort(key=lambda x: (int(x["obj_id"]), str(x["label"]).lower()))
        return objects

    def _build_category_candidates(self) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        for word in CATEGORY_WORDS:
            label = word.strip().lower()
            uid = f"category_{self._slugify(label)}"
            items.append(
                {
                    "uid": uid,
                    "label": label,
                    "path": f"text://{uid}",
                    "image": self._render_text_card_bgr(label),
                }
            )
        return items

    def _build_activity_candidates(self) -> List[Dict[str, object]]:
        activities = self._fetch_activity_candidates_from_llm()
        items: List[Dict[str, object]] = []
        for idx, entry in enumerate(activities):
            label = str(entry.get("activity", "")).strip()
            if not label:
                continue

            uid = f"activity_{idx:02d}_{self._slugify(label)}"
            items.append(
                {
                    "uid": uid,
                    "label": label,
                    "path": f"text://{uid}",
                    "image": self._render_text_card_bgr(label),
                }
            )
        return items[:ACTIVITY_CANDIDATE_COUNT]

    @staticmethod
    def _paginate(items: List[Dict[str, object]], size: int = MAX_BATCH_SIZE) -> List[List[Dict[str, object]]]:
        if not items:
            return []
        return [items[i: i + size] for i in range(0, len(items), size)]

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

    def _publish_cmd(self, cmd: str, **meta: object) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        frame_parts = [f"cmd={cmd}"]
        for key, value in meta.items():
            frame_parts.append(f"{key}={value}")
        msg.header.frame_id = ";".join(frame_parts)
        msg.height = 1
        msg.width = 1
        msg.encoding = "bgr8"
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self.publisher_.publish(msg)

    def _publish_llm_stream_event(
        self,
        event_type: str,
        text: str = "",
        stage: str = STAGE_ACTIVITY,
    ) -> None:
        payload = {"type": event_type, "stage": stage}
        if text:
            payload["text"] = text
        msg = StringMsg()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.llm_stream_pub.publish(msg)

    def _publish_pseudo_stream_text(self, text: str, chunk_size: int = 24) -> None:
        clean = str(text or "").strip()
        if not clean:
            return
        for i in range(0, len(clean), max(1, chunk_size)):
            self._publish_llm_stream_event("append", text=clean[i : i + chunk_size])

    def _current_pages(self) -> List[List[Dict[str, object]]]:
        if self.stage == STAGE_OBJECT:
            return self.object_pages
        if self.stage == STAGE_CATEGORY:
            return self.category_pages
        return self.activity_pages

    def _publish_current_page(self) -> None:
        pages = self._current_pages()
        if not pages:
            self.get_logger().warning(f"No pages available for stage={self.stage}")
            return
        if self.current_page < 0:
            self.current_page = 0
        if self.current_page >= len(pages):
            self.current_page = len(pages) - 1

        page_items = pages[self.current_page]
        count = len(page_items)
        if count <= 0:
            self.get_logger().warning(
                f"Skip empty page stage={self.stage}, page={self.current_page + 1}"
            )
            return

        group_id = self.next_group_id
        self.next_group_id += 1
        self.current_group_id = group_id
        self.current_page_items = page_items

        self._publish_cmd(
            "batch_start",
            group=group_id,
            count=count,
            stage=self.stage,
            page=self.current_page,
        )

        for index, item in enumerate(page_items):
            frame_id = (
                f"source=reasoner;group={group_id};index={index};"
                f"image_path={os.path.basename(str(item.get('path', '')))};"
                f"count={count};end={1 if index == count - 1 else 0};"
                f"stage={self.stage};page={self.current_page};"
                f"item_uid={item.get('uid', '')};"
                f"item_label={self._slugify(str(item.get('label', '')))}"
            )
            self.publisher_.publish(self._make_image_msg(item["image"], frame_id))

        self._publish_cmd(
            "batch_end",
            group=group_id,
            count=count,
            stage=self.stage,
            page=self.current_page,
        )

        self.get_logger().info(
            f"Published stage={self.stage}, page={self.current_page + 1}/{len(pages)}, "
            f"count={count}, group={group_id}"
        )

    @staticmethod
    def _parse_frame_id(frame_id: str) -> Dict[str, str]:
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
        if self.finished:
            return

        meta = self._parse_frame_id(msg.header.frame_id)
        cmd = meta.get("cmd", "")

        if cmd == "ssvep_ready":
            if not self.handshake_sent:
                self._publish_cmd("reasoner_ready")
                self.handshake_sent = True
                self.handshake_complete = True
                self.stage = STAGE_OBJECT
                self.current_page = 0
                self.get_logger().info("Handshake complete; entering StateA (object selection)")
                self._publish_current_page()
            return

        if not self.handshake_complete:
            self.get_logger().warning("Ignore feedback before handshake")
            return

        if cmd == "selection":
            self._handle_selection(meta)
            return

        if cmd == "confirm":
            self._handle_confirm()
            return

        if cmd == "rollback":
            self._handle_rollback()
            return

    def _item_from_selection_slot(self, slot: int) -> Optional[Dict[str, object]]:
        local_idx = SLOT_TO_ITEM_INDEX.get(slot)
        if local_idx is None:
            return None
        if local_idx < 0 or local_idx >= len(self.current_page_items):
            return None
        return self.current_page_items[local_idx]

    def _handle_selection(self, meta: Dict[str, str]) -> None:
        try:
            slot = int(meta.get("slot", "-1"))
        except ValueError:
            self.get_logger().warning(f"Invalid selection metadata: {meta}")
            return

        if slot not in VALID_SELECTION_SLOTS:
            self.get_logger().warning(f"Ignore selection slot={slot} for stage={self.stage}")
            return

        selected_item = self._item_from_selection_slot(slot)
        if selected_item is None:
            self.get_logger().warning(
                f"Selection slot={slot} out of range for stage={self.stage}, "
                f"page_count={len(self.current_page_items)}"
            )
            return

        if self.stage == STAGE_OBJECT:
            self.selected_objects.append(selected_item)
            self.get_logger().info(
                f"StateA select: uid={selected_item['uid']}, label={selected_item['label']}, "
                f"selected_total={len(self.selected_objects)}"
            )
            self._publish_current_page()
            return

        if self.stage == STAGE_CATEGORY:
            self.selected_category = selected_item
            self.get_logger().info(
                f"StateB select: category={selected_item['label']}, entering StateC"
            )
            self.stage = STAGE_ACTIVITY
            self.current_page = 0
            self.activity_candidates = self._build_activity_candidates()
            self.activity_pages = self._paginate(self.activity_candidates)
            if not self.activity_pages:
                self.get_logger().warning("No activity candidates, using fallback list")
                self.activity_candidates = self._fallback_activity_candidates()
                self.activity_pages = self._paginate(self.activity_candidates)
            self._publish_current_page()
            return

        if self.stage == STAGE_ACTIVITY:
            self.selected_activity = selected_item
            self.finished = True
            self._log_final_result()
            self._publish_cmd("done")
            return

    def _handle_confirm(self) -> None:
        pages = self._current_pages()
        if not pages:
            self.get_logger().warning(f"Confirm ignored: no pages in stage={self.stage}")
            return

        if self.stage == STAGE_OBJECT:
            if self.current_page < len(self.object_pages) - 1:
                self.current_page += 1
                self.get_logger().info(
                    f"StateA confirm: move to object page {self.current_page + 1}/{len(self.object_pages)}"
                )
                self._publish_current_page()
                return
            self.stage = STAGE_CATEGORY
            self.current_page = 0
            self.get_logger().info("StateA confirm on last page: entering StateB")
            self._publish_current_page()
            return

        if self.current_page < len(pages) - 1:
            self.current_page += 1
            self.get_logger().info(
                f"State {self.stage} confirm: next page {self.current_page + 1}/{len(pages)}"
            )
            self._publish_current_page()
            return

        self.get_logger().info(
            f"State {self.stage} confirm on last page: keep current page {self.current_page + 1}"
        )
        self._publish_current_page()

    def _handle_rollback(self) -> None:
        if self.stage == STAGE_OBJECT:
            if not self.selected_objects:
                self.get_logger().warning("StateA rollback ignored: no selected object history")
                return
            removed = self.selected_objects.pop()
            uid = str(removed.get("uid", ""))
            self.current_page = self.object_uid_to_page.get(uid, 0)
            self.get_logger().info(
                f"StateA rollback: removed uid={uid}, selected_total={len(self.selected_objects)}, "
                f"back_to_page={self.current_page + 1}/{len(self.object_pages)}"
            )
            self._publish_current_page()
            return

        if self.stage == STAGE_CATEGORY:
            self.stage = STAGE_OBJECT
            if self.selected_objects:
                last_uid = str(self.selected_objects[-1].get("uid", ""))
                self.current_page = self.object_uid_to_page.get(last_uid, self.current_page)
            self.get_logger().info("StateB rollback: return to StateA")
            self._publish_current_page()
            return

        if self.stage == STAGE_ACTIVITY:
            self.stage = STAGE_CATEGORY
            if self.selected_category is not None:
                uid = str(self.selected_category.get("uid", ""))
                self.current_page = self.category_uid_to_page.get(uid, 0)
            else:
                self.current_page = 0
            self.get_logger().info("StateC rollback: return to StateB")
            self._publish_current_page()
            return

    def _extract_json_array(self, text: str) -> List[object]:
        raw = text.strip()
        if not raw:
            return []
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
        return []

    def _fetch_activity_candidates_from_llm(self) -> List[Dict[str, object]]:
        object_labels = [str(obj.get("label", "")) for obj in self.selected_objects]
        category = ""
        if self.selected_category is not None:
            category = str(self.selected_category.get("label", ""))

        self._publish_llm_stream_event(
            "reset",
            text="Generating activity candidates...\n",
        )
        try:
            if not object_labels:
                self._publish_llm_stream_event(
                    "error",
                    text="No selected objects, fallback activity candidates used.\n",
                )
                return self._fallback_activity_candidates()

            # Check if model is loaded
            with self._model_lock:
                if not self._model_loaded or self._qwen_model is None:
                    if self._model_loading:
                        self.get_logger().warning("Qwen model still loading, using fallback")
                        self._publish_llm_stream_event(
                            "error",
                            text="Qwen model is still loading. Fallback activity candidates used.\n",
                        )
                    else:
                        self.get_logger().warning("Qwen model not available, using fallback")
                        self._publish_llm_stream_event(
                            "error",
                            text="Qwen model not available. Fallback activity candidates used.\n",
                        )
                    return self._fallback_activity_candidates()
                model = self._qwen_model
                processor = self._qwen_processor

            prompt_text = (
                "You are a planning assistant. Return ONLY a JSON array with exactly "
                f"{ACTIVITY_CANDIDATE_COUNT} entries sorted by descending plausibility. "
                "Each entry must be an object with fields: activity (string), score (number).\n\n"
                f"Selected objects: {object_labels}\n"
                f"Semantic category: {category}\n"
                "Generate candidate user activities grounded in these objects/category."
            )

            try:
                import torch
                from qwen_vl_utils import process_vision_info

                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt_text}],
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=QWEN_MAX_NEW_TOKENS,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                content = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                self._publish_pseudo_stream_text(content, chunk_size=24)
                self.get_logger().info(f"Qwen output: {content[:200]}...")

            except Exception as exc:
                self.get_logger().warning(f"Qwen inference failed, fallback enabled: {exc}")
                self._publish_llm_stream_event(
                    "error",
                    text=f"Qwen inference failed: {exc}\nFallback activity candidates used.\n",
                )
                return self._fallback_activity_candidates()

            raw_list = self._extract_json_array(content)
            if not raw_list:
                self.get_logger().warning("Qwen content does not contain valid JSON array, fallback")
                self._publish_llm_stream_event(
                    "error",
                    text="Qwen output is not a valid JSON array. Fallback candidates used.\n",
                )
                return self._fallback_activity_candidates()

            normalized: List[Dict[str, object]] = []
            seen_labels = set()
            for item in raw_list:
                if isinstance(item, dict):
                    label = str(item.get("activity", "")).strip()
                    score = item.get("score", "")
                else:
                    label = str(item).strip()
                    score = ""
                if not label:
                    continue
                lowered = label.lower()
                if lowered in seen_labels:
                    continue
                seen_labels.add(lowered)
                normalized.append({"activity": label, "score": score})
                if len(normalized) >= ACTIVITY_CANDIDATE_COUNT:
                    break

            if not normalized:
                self.get_logger().warning("Qwen output normalized to empty list, fallback")
                self._publish_llm_stream_event(
                    "error",
                    text="Qwen candidates normalized to empty list. Fallback candidates used.\n",
                )
                return self._fallback_activity_candidates()

            while len(normalized) < ACTIVITY_CANDIDATE_COUNT:
                fallback_tail = self._fallback_activity_candidates()
                for fallback_item in fallback_tail:
                    label = str(fallback_item.get("activity", "")).strip()
                    if not label or label.lower() in seen_labels:
                        continue
                    seen_labels.add(label.lower())
                    normalized.append(fallback_item)
                    if len(normalized) >= ACTIVITY_CANDIDATE_COUNT:
                        break

            return normalized[:ACTIVITY_CANDIDATE_COUNT]
        finally:
            self._publish_llm_stream_event("done")

    def _fallback_activity_candidates(self) -> List[Dict[str, object]]:
        object_labels = [str(obj.get("label", "")).strip() for obj in self.selected_objects]
        object_labels = [x for x in object_labels if x]
        category = "general"
        if self.selected_category is not None:
            category = str(self.selected_category.get("label", "general")).strip() or "general"

        generated: List[Dict[str, object]] = []
        for label in object_labels[:4]:
            generated.append({"activity": f"{category} with {label}", "score": ""})
            generated.append({"activity": f"prepare {label} for {category}", "score": ""})

        templates = [
            f"plan a {category} task",
            f"start {category} workflow",
            f"finish {category} routine",
            f"review tools for {category}",
            f"organize desk for {category}",
            f"clean up after {category}",
            f"document result of {category}",
            f"schedule next {category} step",
            f"optimize setup for {category}",
            f"final check before {category}",
        ]
        for text in templates:
            generated.append({"activity": text, "score": ""})

        unique: List[Dict[str, object]] = []
        seen = set()
        for item in generated:
            label = str(item.get("activity", "")).strip()
            if not label:
                continue
            lowered = label.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(item)
            if len(unique) >= ACTIVITY_CANDIDATE_COUNT:
                break

        return unique[:ACTIVITY_CANDIDATE_COUNT]

    def _log_final_result(self) -> None:
        self.get_logger().info("=" * 60)
        self.get_logger().info("StateC complete: final reasoning result")
        self.get_logger().info(f"Selected objects ({len(self.selected_objects)}):")
        for idx, item in enumerate(self.selected_objects, 1):
            self.get_logger().info(f"  {idx}. {item.get('label')} ({item.get('uid')})")

        category = self.selected_category.get("label") if self.selected_category else ""
        activity = self.selected_activity.get("label") if self.selected_activity else ""
        object_uids = [str(item.get("uid", "")).strip() for item in self.selected_objects]
        object_uids = [x for x in object_uids if x]
        objects_text = "、".join(object_uids) if object_uids else "（无）"
        self.get_logger().info(f"Selected category: {category}")
        self.get_logger().info(f"Selected activity: {activity}")
        self.get_logger().info(
            f"最终用户选择了物体：{objects_text}；最终活动：{activity or '（无）'}"
        )
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = ReasonerPublishTest2Node()
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
