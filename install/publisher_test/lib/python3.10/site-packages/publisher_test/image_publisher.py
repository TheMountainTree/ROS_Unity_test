#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class ImagePublisher(Node):
    def __init__(self):
        super().__init__("image_publisher")
        self.publisher_ = self.create_publisher(Image, "/fetch_head/rgb/image_raw", 10)
        self.timer = self.create_timer(0.5, self.publish_image)
        self.count = 0
        self.get_logger().info("Image Publisher started, publishing to /fetch_head/rgb/image_raw")

    def publish_image(self):
        # 创建测试图像 (640x480 BGR格式)
        height, width = 480, 640
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 生成彩色渐变 (BGR格式)
        for y in range(height):
            for x in range(width):
                img[y, x] = [y % 256, x % 256, (y + x) % 256]
        
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = height
        msg.width = width
        msg.encoding = "bgr8"
        msg.step = width * 3
        msg.data = img.tobytes()
        
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published frame {self.count} (640x480 BGR8)")
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
