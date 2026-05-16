#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import os
from sensor_msgs.msg import CompressedImage
import numpy as np


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ---------------- RGB Publisher ----------------
        self.pub = self.create_publisher(
            CompressedImage,
            '/image/compressed',
            10
        )

        self.video_path = os.path.expanduser('~/ros_workplace/data/img.mp4')
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f"Video open failed: {self.video_path}")
            return

        self.get_logger().info(f"Opened video: {self.video_path}")

        # ---------------- Depth (연속 프레임 bin) ----------------
        self.depth_path = os.path.expanduser('~/ros_workplace/data/DepthMap.bin')
        self.depth_w = 256
        self.depth_h = 256
        self.depth_dtype = np.float32

        self.depth_mm = None
        self.depth_frames = 0

        self._init_depth_memmap()

        # 같은 타이머에서 처리 (프레임 타이밍 동기)
        self.timer = self.create_timer(1/30.0, self.timer_callback)

    def _init_depth_memmap(self):
        if not os.path.isfile(self.depth_path):
            self.get_logger().error(f"Depth file not found: {self.depth_path}")
            return

        file_bytes = os.path.getsize(self.depth_path)
        bytes_per_frame = self.depth_w * self.depth_h * np.dtype(self.depth_dtype).itemsize

        if file_bytes % bytes_per_frame != 0:
            self.get_logger().error(
                f"Depth file size not multiple of one frame: "
                f"file_bytes={file_bytes}, bytes_per_frame={bytes_per_frame}"
            )
            return

        self.depth_frames = file_bytes // bytes_per_frame
        if self.depth_frames <= 0:
            self.get_logger().error("Depth frames computed as 0")
            return

        # (frames, H, W)로 memmap
        self.depth_mm = np.memmap(
            self.depth_path,
            dtype=self.depth_dtype,
            mode='r',
            shape=(self.depth_frames, self.depth_h, self.depth_w)
        )

        self.get_logger().info(
            f"Loaded depth memmap: {self.depth_path} "
            f"(frames={self.depth_frames}, size={self.depth_h}x{self.depth_w}, dtype=float32)"
        )

    def normalize_depth(self, depth_2d: np.ndarray) -> np.ndarray:
        """float32 depth → 보기용 uint8 (퍼센타일 기반)"""
        d = np.array(depth_2d, copy=True)

        # NaN/Inf 제거
        d[~np.isfinite(d)] = 0.0

        # 0은 무효로 보고 스케일 산정에서 제외
        valid = d[(d > 0.0) & np.isfinite(d)]
        if valid.size < 10:
            return np.zeros_like(d, dtype=np.uint8)

        mn = float(np.percentile(valid, 1))
        mx = float(np.percentile(valid, 99))
        if mx <= mn + 1e-12:
            mx = mn + 1.0

        d = np.clip(d, mn, mx)
        d = (d - mn) / (mx - mn)
        return (d * 255.0).astype(np.uint8)

    def timer_callback(self):
        # ===== RGB 처리 =====
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # 송신측 RGB 표시
        cv2.imshow("send", frame)

        # RGB → JPEG publish
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, encoded = cv2.imencode('.jpg', frame, encode_param)
        if success:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(encoded).tobytes()
            self.pub.publish(msg)

        # ===== Depth 처리 (publish 없음, 표시만) =====
        if self.depth_mm is not None and self.depth_frames > 0:
            # 현재 RGB 프레임 인덱스에 맞춰 depth 프레임 선택 (mod로 안전하게)
            # read() 후 POS_FRAMES는 "다음 프레임 번호"를 가리키는 경우가 많아서 -1
            rgb_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if rgb_idx < 0:
                rgb_idx = 0

            depth_idx = rgb_idx % self.depth_frames
            depth = self.depth_mm[depth_idx]

            depth8 = self.normalize_depth(depth)
            depth_color = cv2.applyColorMap(depth8, cv2.COLORMAP_TURBO)
            cv2.imshow("send_depth", depth_color)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
