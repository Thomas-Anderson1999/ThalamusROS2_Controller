#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.expanduser('~/ros_workplace/venv/lib/python3.12/site-packages'))

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge


class CameraNodeD455i(Node):

    def __init__(self):
        super().__init__('camera_node_d455i')

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

        # ---------- RealSense Pipeline ----------
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # depth를 color 프레임 좌표계에 맞춤
        self.align = rs.align(rs.stream.color)

        self.declare_parameter('show_preview', True)
        self.show_preview = self.get_parameter('show_preview').value

        self.get_logger().info("RealSense D455 started")

        # 30Hz
        self.timer = self.create_timer(1 / 30.0, self.timer_callback)

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

        # 프레임 미준비 시 skip (non-blocking)
        success, frames = self.pipeline.try_wait_for_frames(timeout_ms=0)
        if not success:
            return

        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # timestamp 공통 사용 (핵심: 동기화 기준)
        stamp = self.get_clock().now().to_msg()

        # ===== RGB =====
        frame = np.asanyarray(color_frame.get_data())  # BGR, uint8

        if self.show_preview:
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
        # z16 (uint16, 단위: get_units() 미터) → float32 미터
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth = depth_raw * depth_frame.get_units()  # 미터 단위 float32

        # 256x256 리사이즈
        depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_NEAREST)

        # depth publish (32FC1 그대로 전송)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header.stamp = stamp
        self.depth_pub.publish(depth_msg)

        if self.show_preview:
            depth8 = self.normalize_depth(depth)
            color_d = cv2.applyColorMap(depth8, cv2.COLORMAP_TURBO)
            cv2.imshow("send_depth", color_d)
            cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNodeD455i()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
