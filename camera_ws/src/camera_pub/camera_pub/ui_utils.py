import cv2
import numpy as np

from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class ClickableLabel(QLabel):
    def __init__(self, name):
        super().__init__()
        self._name = name
        self._click_cb = None
        self._dblclick_cb = None

    def set_click_cb(self, cb):
        self._click_cb = cb

    def set_dblclick_cb(self, cb):
        self._dblclick_cb = cb

    def _pixel_pos(self, event):
        pm = self.pixmap()
        if pm is None:
            return None, None
        lw, lh = self.width(), self.height()
        iw, ih = pm.width(), pm.height()
        scale = min(lw / iw, lh / ih)
        ox = (lw - iw * scale) / 2
        oy = (lh - ih * scale) / 2
        x = int((event.x() - ox) / scale)
        y = int((event.y() - oy) / scale)
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        return x, y

    def mousePressEvent(self, event):
        x, y = self._pixel_pos(event)
        if x is None:
            return
        print(f"[{self._name}] click x={x}, y={y}")
        if self._click_cb:
            self._click_cb(self._name, x, y)

    def mouseDoubleClickEvent(self, event):
        x, y = self._pixel_pos(event)
        if x is None:
            return
        print(f"[{self._name}] double-click x={x}, y={y}")
        if self._dblclick_cb:
            self._dblclick_cb()


class StreamingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Streaming")
        self._marker_pos  = None   # (x, y) image coords  — yellow
        self._marker_text = ""
        self._marker2_pos  = None  # (x, y) image coords  — green (optflow)
        self._marker2_text = ""
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

    def set_marker(self, x, y, text):
        self._marker_pos  = (x, y)
        self._marker_text = text
        print(f"[marker] {text}")

    def clear_marker(self):
        self._marker_pos  = None
        self._marker_text = ""

    def set_marker2(self, x, y, text):
        if text:
            self._marker2_pos  = (x, y)
            self._marker2_text = text
        else:
            self._marker2_pos  = None
            self._marker2_text = ""

    def set_click_cb(self, cb):
        self.rgb_label.set_click_cb(cb)
        self.depth_label.set_click_cb(cb)

    def set_dblclick_cb(self, cb):
        self.rgb_label.set_dblclick_cb(cb)
        self.depth_label.set_dblclick_cb(cb)

    def update_rgb(self, img_bgr):
        if self._marker_pos is not None or self._marker2_pos is not None:
            img_bgr = img_bgr.copy()
        if self._marker_pos is not None:
            mx, my = self._marker_pos
            r = 14
            cv2.circle(img_bgr, (mx, my), r, (0, 255, 255), 2)
            cv2.line(img_bgr, (mx - r, my), (mx + r, my), (0, 255, 255), 1)
            cv2.line(img_bgr, (mx, my - r), (mx, my + r), (0, 255, 255), 1)
            cv2.putText(img_bgr, self._marker_text, (mx + r + 4, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if self._marker2_pos is not None:
            mx2, my2 = self._marker2_pos
            r2 = 10
            cv2.circle(img_bgr, (mx2, my2), r2, (0, 255, 0), 2)
            cv2.line(img_bgr, (mx2 - r2, my2), (mx2 + r2, my2), (0, 255, 0), 1)
            cv2.line(img_bgr, (mx2, my2 - r2), (mx2, my2 + r2), (0, 255, 0), 1)
            cv2.putText(img_bgr, self._marker2_text, (mx2 + r2 + 4, my2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
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
