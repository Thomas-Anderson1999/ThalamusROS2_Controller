#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import os
import numpy as np

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge


class RecordNode(Node):

    def __init__(self):
        super().__init__('record_node')

        self.bridge = CvBridge()

        # ---------- RGB Publisher ----------
        self.rgb_pub = self.create_publisher(
            CompressedImage,
            '/camera/rgb/compressed',
            10
        )

        # ---------- Depth Publisher ----------
        self.depth_pub = self.create_publisher(
            Image,
            '/camera/depth',
            10
        )

        # ---------- RGB Video ----------
        self.video_path = os.path.expanduser('~/ros_workplace/data/img.mp4')
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.get_logger().error("RGB open failed")
            return

        # ---------- Depth Memmap ----------
        self.depth_path = os.path.expanduser('~/ros_workplace/data/DepthMap.bin')
        self.w = 256
        self.h = 256

        bytes_per_frame = self.w * self.h * 4
        file_bytes = os.path.getsize(self.depth_path)
        self.depth_frames = file_bytes // bytes_per_frame

        self.depth_mm = np.memmap(
            self.depth_path,
            dtype=np.float32,
            mode='r',
            shape=(self.depth_frames, self.h, self.w)
        )

        self.get_logger().info(f"Depth frames: {self.depth_frames}")

        # 30Hz
        self.timer = self.create_timer(1/30.0, self.timer_callback)

    def normalize_depth(self, depth):
        d = depth.copy()
        d[~np.isfinite(d)] = 0

        valid = d[d > 0]
        if valid.size < 10:
            return np.zeros_like(d, dtype=np.uint8)

        mn = np.percentile(valid, 1)
        mx = np.percentile(valid, 99)

        d = np.clip(d, mn, mx)
        d = (d - mn) / (mx - mn)

        return (d * 255).astype(np.uint8)

    def timer_callback(self):

        # ===== RGB =====
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # timestamp 공통 사용 (핵심: 동기화 기준)
        stamp = self.get_clock().now().to_msg()

        # 표시
        cv2.imshow("send_rgb", frame)

        # RGB publish (JPEG)
        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            msg = CompressedImage()
            msg.header = Header()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = encoded.tobytes()
            self.rgb_pub.publish(msg)

        # ===== Depth =====
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        idx %= self.depth_frames

        depth = self.depth_mm[idx]

        # depth publish (32FC1 그대로 전송)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header.stamp = stamp
        self.depth_pub.publish(depth_msg)

        # 표시용 normalize
        depth8 = self.normalize_depth(depth)
        color = cv2.applyColorMap(depth8, cv2.COLORMAP_TURBO)
        cv2.imshow("send_depth", color)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RecordNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
