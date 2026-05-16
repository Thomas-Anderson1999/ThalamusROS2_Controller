#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class SyncViewer(Node):

    def __init__(self):
        super().__init__('sync_viewer')

        self.bridge = CvBridge()

        # message_filters 사용 (핵심: 동기화)
        self.rgb_sub = Subscriber(self, CompressedImage, '/camera/rgb/compressed')
        self.depth_sub = Subscriber(self, Image, '/camera/depth')

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            10,
            0.03   # 30ms 허용
        )

        self.sync.registerCallback(self.callback)

        self.get_logger().info("Sync viewer started")

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

    def callback(self, rgb_msg, depth_msg):

        # RGB decode
        rgb = cv2.imdecode(np.frombuffer(rgb_msg.data, np.uint8), cv2.IMREAD_COLOR)

        # Depth decode
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        # 표시
        cv2.imshow("recv_rgb", rgb)

        depth8 = self.normalize_depth(depth)
        color = cv2.applyColorMap(depth8, cv2.COLORMAP_TURBO)
        cv2.imshow("recv_depth", color)


def main(args=None):
    rclpy.init(args=args)
    node = SyncViewer()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            cv2.waitKey(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
