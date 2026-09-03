#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.expanduser('~/ros_workplace/venv/lib/python3.12/site-packages'))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import numpy as np
import pyrealsense2 as rs

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge

from camera_pub.kalman_ahrs import KalmanAHRS


class CameraNodeD455i(Node):

    def __init__(self):
        super().__init__('camera_node_d455i')

        self.bridge = CvBridge()

        # ---------- Publishers ----------
        # 라이브 스트림은 재전송 없이 최신 프레임만 — BEST_EFFORT + KEEP_LAST 1
        stream_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.rgb_pub = self.create_publisher(CompressedImage, '/camera/rgb/compressed', stream_qos)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', stream_qos)
        self.att_pub = self.create_publisher(Vector3, 'camera_attitude', 10)

        self.declare_parameter('show_preview', True)
        self.show_preview = self.get_parameter('show_preview').value

        # ---------- Kalman AHRS ----------
        self.ahrs = KalmanAHRS(dt=0.005, Q_val=0.01, R_val=10.0)

        # ---------- Video pipeline ----------
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)

        # ---------- IMU via sensor API (avoids pipeline conflict) ----------
        dev = profile.get_device()
        self._motion_sensor = None
        for sensor in dev.query_sensors():
            motion_profiles = [p for p in sensor.get_stream_profiles()
                               if p.stream_type() in (rs.stream.gyro, rs.stream.accel)]
            if motion_profiles:
                self._motion_sensor = sensor
                break

        if self._motion_sensor:
            imu_profiles = [p for p in self._motion_sensor.get_stream_profiles()
                            if (p.stream_type() == rs.stream.gyro  and p.fps() == 200)
                            or (p.stream_type() == rs.stream.accel and p.fps() == 63)]

            def _imu_cb(frame):
                motion = frame.as_motion_frame()
                if not motion:
                    return
                d  = motion.get_motion_data()
                st = motion.get_profile().stream_type()
                if st == rs.stream.gyro:
                    self.ahrs.push_gyro(d.x, d.y, d.z)
                elif st == rs.stream.accel:
                    self.ahrs.push_accel(d.x, d.y, d.z)

            for p in imu_profiles:
                self.get_logger().info(f"IMU profile: {p.stream_type()} @ {p.fps()} Hz")
            self._motion_sensor.open(imu_profiles)
            self._motion_sensor.start(_imu_cb)
            self.get_logger().info("IMU sensor started")
        else:
            self.get_logger().warn("No motion sensor found - attitude will not be published")

        self.get_logger().info("RealSense D455 started")

        # 30Hz publish timer
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
        success, frames = self.pipeline.try_wait_for_frames(timeout_ms=0)
        if not success:
            return

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        stamp = self.get_clock().now().to_msg()

        # ===== RGB =====
        frame = np.asanyarray(color_frame.get_data())

        if self.show_preview:
            cv2.imshow("send_rgb", frame)

        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            msg = CompressedImage()
            msg.header = Header()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = encoded.tobytes()
            self.rgb_pub.publish(msg)

        # ===== Depth =====
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth = depth_raw * depth_frame.get_units()
        depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_NEAREST)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header.stamp = stamp
        self.depth_pub.publish(depth_msg)

        if self.show_preview:
            depth8 = self.normalize_depth(depth)
            color_d = cv2.applyColorMap(depth8, cv2.COLORMAP_TURBO)
            cv2.imshow("send_depth", color_d)
            cv2.waitKey(1)

        # ===== Attitude =====
        if self.ahrs.is_ready:
            roll, pitch, yaw = self.ahrs.get_attitude_deg()
            self.att_pub.publish(Vector3(x=roll, y=pitch, z=yaw))

    def destroy_node(self):
        if self._motion_sensor:
            self._motion_sensor.stop()
            self._motion_sensor.close()
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
