import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
from glob import glob
import socket
import json

class HistorySenderNode(Node):
    def __init__(self):
        super().__init__('history_sender_node')
        # 发布话题，消息类型为 sensor_msgs/Image
        self.publisher_ = self.create_publisher(Image, '/history_image', 10)
        self.control_sub_ = self.create_subscription(
            String, '/history_control', self.control_callback, 10
        )
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.count = 0
        self.max_images = 10
        self.finished = False
        self.auto_shutdown_on_complete = False
        self.next_image_id = 1
        self.output_width = 100
        self.output_height = 100
        self.flip_vertical_for_unity = True
        self.udp_target_ip = "127.0.0.1"
        self.udp_target_port = 12001
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_source_path = os.path.expanduser(
            "~/workspace/eeg_robot/src/robot_ctr/graph/graph/results/segmentation_20260206_223629"
        )
        self.local_image_paths = self._collect_image_paths(self.image_source_path)

        if self.local_image_paths:
            self.get_logger().info(
                f'检测到本地图片源，共 {len(self.local_image_paths)} 张。将优先发送本地图片，最多发送 {self.max_images} 张。'
            )
        else:
            self.get_logger().info('未检测到可用本地图片，将发送随机色块图。')
        self.get_logger().info(
            f'历史删除控制 UDP 通道: {self.udp_target_ip}:{self.udp_target_port} (独立于 SSVEP 端口 9999/10000/10001)'
        )

    def _send_udp_command(self, payload):
        data = json.dumps(payload, ensure_ascii=True).encode('utf-8')
        self.udp_sock.sendto(data, (self.udp_target_ip, self.udp_target_port))

    def send_delete_last(self):
        self._send_udp_command({"cmd": "delete_last"})
        self.get_logger().info('已发送删除命令: delete_last')

    def send_clear_all(self):
        self._send_udp_command({"cmd": "clear"})
        self.get_logger().info('已发送删除命令: clear')

    def send_delete_id(self, image_id):
        self._send_udp_command({"cmd": "delete_id", "id": int(image_id)})
        self.get_logger().info(f'已发送删除命令: delete_id={image_id}')

    def control_callback(self, msg):
        command = msg.data.strip().lower()
        if not command:
            return

        if command == "delete_last":
            self.send_delete_last()
            return
        if command in ("clear", "delete_all"):
            self.send_clear_all()
            return
        if command.startswith("delete_id:"):
            try:
                image_id = int(command.split(":", 1)[1])
            except ValueError:
                self.get_logger().warn(f'无效 delete_id 指令: {msg.data}')
                return
            self.send_delete_id(image_id)
            return

        self.get_logger().warn(
            f'未知控制指令: {msg.data}，支持 delete_last / clear / delete_id:<int>'
        )

    def publish_history_image(self, img, desc):
        image_id = self.next_image_id
        self.next_image_id += 1

        ros_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        ros_msg.header.frame_id = f"hist_id={image_id}"
        self.publisher_.publish(ros_msg)
        self.get_logger().info(f'已发送第 {self.count + 1} 张图片(id={image_id}): {desc}')

    def _collect_image_paths(self, source_path):
        if not source_path:
            return []

        source_path = os.path.expanduser(source_path)
        if os.path.isfile(source_path):
            return [source_path]

        if not os.path.isdir(source_path):
            return []

        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
        image_paths = []
        for p in patterns:
            image_paths.extend(glob(os.path.join(source_path, p)))
            image_paths.extend(glob(os.path.join(source_path, p.upper())))
        return sorted(image_paths)

    def _normalize_for_unity(self, img):
        # 坐标系对齐：OpenCV/ROS 常见图像原点在左上，Unity 纹理显示常见表现需要上下翻转。
        if self.flip_vertical_for_unity:
            img = cv2.flip(img, 0)

        # 统一输出尺寸，避免 UI 布局因本地图分辨率差异导致 item 间距变化。
        if img.shape[1] != self.output_width or img.shape[0] != self.output_height:
            img = cv2.resize(img, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
        return img

    def _create_fallback_image(self):
        color = np.random.randint(0, 256, (3,)).tolist()
        img = np.zeros((self.output_height, self.output_width, 3), np.uint8)
        img[:] = color
        img = self._normalize_for_unity(img)
        return img, f'随机色块图 颜色={color}'

    def _create_local_image(self, index):
        if index >= len(self.local_image_paths):
            return self._create_fallback_image()

        img_path = self.local_image_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn(f'读取本地图片失败，回退随机图: {img_path}')
            return self._create_fallback_image()
        img = self._normalize_for_unity(img)
        return img, f'本地图片 {os.path.basename(img_path)}'

    def timer_callback(self):
        if self.finished:
            return

        if self.count < self.max_images:
            if self.local_image_paths:
                img, desc = self._create_local_image(self.count)
            else:
                img, desc = self._create_fallback_image()

            self.publish_history_image(img, desc)
            
            self.count += 1
        else:
            self.get_logger().info('已完成 10 张图片发送，流程结束。')
            self.timer.cancel()
            self.finished = True
            if self.auto_shutdown_on_complete:
                self.udp_sock.close()
                rclpy.shutdown()
            else:
                self.get_logger().info(
                    '自动发送已停止，节点保持在线。可继续通过 /history_control 发送删除命令。'
                )

    def destroy_node(self):
        try:
            self.udp_sock.close()
        except Exception:
            pass
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HistorySenderNode()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
