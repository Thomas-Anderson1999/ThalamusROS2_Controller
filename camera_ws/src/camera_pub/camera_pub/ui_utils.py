import cv2
import numpy as np

from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class ClickableLabel(QLabel):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def mousePressEvent(self, event):
        # map click position to image pixel coordinates
        pm = self.pixmap()
        if pm is None:
            return
        lw, lh = self.width(), self.height()
        iw, ih = pm.width(), pm.height()
        # image is centered with KeepAspectRatio
        scale = min(lw / iw, lh / ih)
        ox = (lw - iw * scale) / 2
        oy = (lh - ih * scale) / 2
        x = int((event.x() - ox) / scale)
        y = int((event.y() - oy) / scale)
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        print(f"[{self._name}] click x={x}, y={y}")


class StreamingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Streaming")
        layout = QHBoxLayout()
        self.rgb_label   = ClickableLabel("camera")
        self.depth_label = ClickableLabel("depth")
        for lbl in (self.rgb_label, self.depth_label):
            lbl.setMinimumSize(320, 240)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: black;")
            layout.addWidget(lbl)
        self.setLayout(layout)
        self.resize(1280, 480)

    def update_rgb(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        self.rgb_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.rgb_label.width(), self.rgb_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_depth(self, depth_colored):
        img = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        self.depth_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.depth_label.width(), self.depth_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
