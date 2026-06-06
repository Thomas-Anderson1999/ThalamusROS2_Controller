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
        self._marker_3d   = None   # np.array [x,y,z] in metres — yellow
        self._marker2_pos  = None  # (x, y) image coords  — green (optflow)
        self._marker2_text = ""
        self._marker2_3d   = None  # np.array [x,y,z] in metres — green
        self._marker3_pos  = None  # (x, y) image coords  — red (alt+click)
        self._marker3_text = ""
        self._rover_px   = None  # projected rover pixel x — magenta
        self._rover_py   = None
        self._rover_dist = None  # float, 3D distance green marker → rover (m)
        self._esc_cb     = None
        self.setFocusPolicy(Qt.StrongFocus)
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

    def set_marker(self, x, y, text, pos_3d=None):
        self._marker_pos  = (x, y)
        self._marker_text = text
        if pos_3d is not None:
            self._marker_3d = pos_3d
        print(f"[marker] {text}")

    def clear_marker(self):
        self._marker_pos  = None
        self._marker_text = ""
        self._marker_3d   = None

    def set_marker2(self, x, y, text, pos_3d=None):
        if text:
            self._marker2_pos  = (x, y)
            self._marker2_text = text
            if pos_3d is not None:
                self._marker2_3d = pos_3d
        else:
            self._marker2_pos  = None
            self._marker2_text = ""
            self._marker2_3d   = None

    def set_marker3(self, x, y, text):
        self._marker3_pos  = (x, y)
        self._marker3_text = text

    def clear_marker3(self):
        self._marker3_pos  = None
        self._marker3_text = ""

    def set_rover_line(self, px, py, dist):
        self._rover_px   = px
        self._rover_py   = py
        self._rover_dist = dist

    def clear_rover_line(self):
        self._rover_px   = None
        self._rover_py   = None
        self._rover_dist = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.clear_marker3()
            if self._esc_cb:
                self._esc_cb()
        super().keyPressEvent(event)

    def set_esc_cb(self, cb):
        self._esc_cb = cb

    def set_click_cb(self, cb):
        self.rgb_label.set_click_cb(cb)
        self.depth_label.set_click_cb(cb)

    def set_dblclick_cb(self, cb):
        self.rgb_label.set_dblclick_cb(cb)
        self.depth_label.set_dblclick_cb(cb)

    def update_rgb(self, img_bgr):
        if (self._marker_pos is not None or self._marker2_pos is not None
                or self._marker3_pos is not None or self._rover_px is not None):
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
        if self._marker3_pos is not None:
            mx3, my3 = self._marker3_pos
            r3 = 12
            cv2.circle(img_bgr, (mx3, my3), r3, (0, 0, 255), 2)
            cv2.line(img_bgr, (mx3 - r3, my3), (mx3 + r3, my3), (0, 0, 255), 1)
            cv2.line(img_bgr, (mx3, my3 - r3), (mx3, my3 + r3), (0, 0, 255), 1)
            cv2.putText(img_bgr, self._marker3_text, (mx3 + r3 + 4, my3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if self._marker2_pos is not None and self._rover_px is not None:
            mx2, my2 = self._marker2_pos
            rvx, rvy = self._rover_px, self._rover_py
            h_img, w_img = img_bgr.shape[:2]
            # 라인이 화면을 벗어나는 경우 화면 경계 교점을 구해 텍스트 중점으로 사용
            dx, dy = rvx - mx2, rvy - my2
            t_clip = 1.0
            if dx > 0:
                t = (w_img - 1 - mx2) / dx
                if 0 < t < t_clip:
                    t_clip = t
            elif dx < 0:
                t = -mx2 / dx
                if 0 < t < t_clip:
                    t_clip = t
            if dy > 0:
                t = (h_img - 1 - my2) / dy
                if 0 < t < t_clip:
                    t_clip = t
            elif dy < 0:
                t = -my2 / dy
                if 0 < t < t_clip:
                    t_clip = t
            clip_x = int(mx2 + t_clip * dx)
            clip_y = int(my2 + t_clip * dy)
            cv2.line(img_bgr, (mx2, my2), (rvx, rvy), (255, 0, 255), 1)
            mid_x = (mx2 + clip_x) // 2
            mid_y = (my2 + clip_y) // 2
            if self._rover_dist is not None:
                cv2.putText(img_bgr, f"{self._rover_dist:.2f}m", (mid_x + 4, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        if self._marker_pos is not None and self._marker2_pos is not None:
            mx,  my  = self._marker_pos
            mx2, my2 = self._marker2_pos
            cv2.line(img_bgr, (mx, my), (mx2, my2), (0, 255, 0), 1)
            mid_x = (mx + mx2) // 2
            mid_y = (my + my2) // 2
            if self._marker_3d is not None and self._marker2_3d is not None:
                dist = float(np.linalg.norm(
                    np.asarray(self._marker_3d) - np.asarray(self._marker2_3d)))
                dist_text = f"{dist:.2f}m"
            else:
                dist = float(np.sqrt((mx2 - mx) ** 2 + (my2 - my) ** 2))
                dist_text = f"{dist:.2f}px"
            cv2.putText(img_bgr, dist_text, (mid_x + 4, mid_y),
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
