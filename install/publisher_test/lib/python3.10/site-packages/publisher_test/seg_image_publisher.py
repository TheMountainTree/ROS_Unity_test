#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class SegImagePublisher(Node):
    def __init__(self):
        super().__init__("seg_image_publisher")
        self.publisher_ = self.create_publisher(Image, "/image_seg", 10)
        self.timer = self.create_timer(0.3, self.publish_image)
        self.count = 0
        self.batch_count = 0
        self.get_logger().info("Seg Image Publisher started, publishing to /image_seg")
        self.get_logger().info("Will publish 6 images per batch for P300/SSVEP test")

    def publish_image(self):
        # P300/SSVEP需要6张图片才启动刺激
        # 每6张为一批
        height, width = 480, 640
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 为每张图片生成不同颜色
        colors = [
            [255, 0, 0],     # 蓝
            [0, 255, 0],     # 绿
            [0, 0, 255],     # 红
            [255, 255, 0],   # 青
            [255, 0, 255],   # 紫
            [0, 255, 255],   # 黄
        ]
        
        color_idx = self.count % 6
        img[:, :] = colors[color_idx]
        
        # 添加简单的数字标识 (用颜色块)
        block_size = 50
        start_x = width // 2 - block_size // 2
        start_y = height // 2 - block_size // 2
        img[start_y:start_y+block_size, start_x:start_x+block_size] = [255, 255, 255]
        
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = height
        msg.width = width
        msg.encoding = "bgr8"
        msg.step = width * 3
        msg.data = img.tobytes()
        
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published image {self.count + 1}/6 in batch {self.batch_count + 1} (color: {color_idx})")
        
        self.count += 1
        if self.count >= 6:
            self.get_logger().info(f"=== Batch {self.batch_count + 1} complete! Unity should start stimulation ===")
            self.count = 0
            self.batch_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = SegImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
