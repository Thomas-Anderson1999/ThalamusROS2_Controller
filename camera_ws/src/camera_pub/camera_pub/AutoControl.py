import os
import time
import sys
import numpy as np
import cv2

from ultralytics import YOLO

_PKG_DIR       = os.path.dirname(os.path.abspath(__file__))
_LIGHT_WEIGHT  = os.path.join(_PKG_DIR, "cnn_weight", "light_best.pt")
_PERSON_WEIGHT = os.path.join(_PKG_DIR, "cnn_weight", "person_best.pt")

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import Qt

from camera_pub.ui_base import UIBase, UIElementType, UIElement
from ThalamusEngine.Interface import LoadThalamusInterface, InitEngine, Pixelto3D


meter_per_enc = 11600
rotate_pi_per_enc = 4819

_ASM_FILE = "ScriptFreeModel.txt".encode("UTF-8")
_thalamus_ready = False


def _init_thalamus():
    global _thalamus_ready
    if _thalamus_ready:
        return True
    if LoadThalamusInterface():
        InitEngine(_ASM_FILE, 1280, 720, 0)
        _thalamus_ready = True
        print("[thalamus] InitEngine OK")
        return True
    print("[thalamus] LoadThalamusInterface FAILED")
    return False


def get_tilt_servo_angle(cam_angle):
    _CAM_ANGLES   = np.array([20, 10, 0, -13, -27, -40, -54, -90], dtype=float)
    _SERVO_ANGLES = np.array([ 0,  9, 19,  29,  39,  49,  59,  69], dtype=float)
    cam_angle = np.clip(cam_angle, _CAM_ANGLES.min(), _CAM_ANGLES.max())
    servo = np.interp(cam_angle, _CAM_ANGLES[::-1], _SERVO_ANGLES[::-1])
    return int(np.clip(servo, 0, 90))


class PresetCamWindow(QWidget, UIBase):
    def __init__(self, parent_window):
        super(PresetCamWindow, self).__init__()
        UIBase.__init__(self)

        self.parent_window = parent_window
        self.ui_dic_list   = parent_window.ui_dic_list  # share ui state

        tilt     = ["UP", "MD", "DN"]
        tilt_def = [0,    20,   65  ]
        pan      = ["LLL", "LL", "L", "N", "R", "RR", "RRR"]
        pan_def  = [-127,  -82, -37,   8,  53,   98,   135 ]

        # ── pan angle header ───────────────────────────────────
        ele_list = [
            UIElement(_type=UIElementType.BUTTON, _caption="Quit", _default_val=self.parent_window.quit_app),
            UIElement(_type=UIElementType.NEW_LINE),
        ]
        for p, d in zip(pan, pan_def):
            ele_list.append(UIElement(_type=UIElementType.EDIT, _caption=p, _default_val=str(d)))
        ele_list.append(UIElement(_type=UIElementType.NEW_LINE))

        # ── tilt rows × pan buttons ────────────────────────────
        for t, td in zip(tilt, tilt_def):
            ele_list.append(UIElement(_type=UIElementType.EDIT, _caption=t, _default_val=str(td)))
            for p in pan:
                colspan = 6 if p == "N" else 1
                ele_list.append(UIElement(
                    _type=UIElementType.BUTTON,
                    _caption=f"{p} {t}",
                    _default_val=lambda state, _t=t, _p=p: self.preset_control(_t, _p),
                    _defalt_culumn=colspan,
                ))
            ele_list.append(UIElement(_type=UIElementType.NEW_LINE))

        mainGrid = QGridLayout()
        subgrid, dic = self.createGroupBox("Preset Cam Control", ele_list)
        self.ui_dic_list.append(dic)
        mainGrid.addWidget(subgrid, 0, 0)

        # ── Auto-Control info ──────────────────────────────────
        info_ele_list = [
            UIElement(_type=UIElementType.EDIT,   _caption="ClickPosX",       _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="ClickPosY",       _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="TgtPan",          _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="TgtTilt",         _default_val="0"),
            UIElement(_type=UIElementType.BUTTON, _caption="set_clicked_pos", _default_val=self.set_clicked_pos),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_X",      _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_Y",      _default_val="0"),
            UIElement(_type=UIElementType.EDIT,   _caption="odometry_Yaw",    _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="local_action_yaw", _default_val="0"),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.COMBO, _caption="Depth_Src", _default_val=["Virtual_Plane", "Depth_Map"]),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosX", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosY", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtPosZ", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtBearing", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="TgtDist", _default_val="0"),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.COMBO,  _caption="CamTrackMode", _default_val=["None", "fix_cam_rotation", "fix_on_target"]),
            UIElement(_type=UIElementType.BUTTON, _caption="excute_first_action", _default_val=self.excute_first_action),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.LISTBOX, _caption="SeqCmd", _default_val=None, _defalt_row=3, _defalt_culumn=8),
        ]
        subgrid2, dic2 = self.createGroupBox("Semi-Auto-Control", info_ele_list)
        self.ui_dic_list.append(dic2)
        self.bind_listbox_changed(self.on_listbox_changed)
        mainGrid.addWidget(subgrid2, 1, 0)

        # ── Image Processing ──────────────────────────────────
        info_ele_list = [
            UIElement(_type=UIElementType.BUTTON, _caption="predict_light", _default_val=self.predict_light),
            UIElement(_type=UIElementType.BUTTON, _caption="predict_person", _default_val=self.predict_person),
        ]
        subgrid2, dic2 = self.createGroupBox("Image Process", info_ele_list)
        self.ui_dic_list.append(dic2)
        self.bind_listbox_changed(self.on_listbox_changed)
        mainGrid.addWidget(subgrid2, 2, 0)

        self.setLayout(mainGrid)
        self.setWindowTitle("Auto Control")
        self.resize(1200, 400)

    # ── Image Processing ──────────────────────────────────
    def _get_camera_frame(self):
        if not self.parent_window._streaming:
            print("[predict] 스트리밍이 열려있지 않습니다. open_streaming 먼저 실행하세요.")
            return None
        img = self.parent_window.ros_node.rgb_image
        if img is None:
            print("[predict] 카메라 이미지가 없습니다.")
            return None
        return img.copy()

    def _draw_and_show(self, img, results, names, window_title):
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{names[cls_id]} {conf:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow(window_title, img)
        cv2.waitKey(1)

    def predict_light(self):
        img = self._get_camera_frame()
        if img is None:
            return
        if not hasattr(self, '_model_light') or self._model_light is None:
            print(f"[predict_light] 모델 로딩: {_LIGHT_WEIGHT}")
            self._model_light = YOLO(_LIGHT_WEIGHT)
        results = self._model_light(img, conf=0.25, iou=0.5)[0]
        self._draw_and_show(img, results, self._model_light.names, "Light Detection")

    def predict_person(self):
        img = self._get_camera_frame()
        if img is None:
            return
        if not hasattr(self, '_model_person') or self._model_person is None:
            print(f"[predict_person] 모델 로딩: {_PERSON_WEIGHT}")
            self._model_person = YOLO(_PERSON_WEIGHT)
        results = self._model_person(img, conf=0.25, iou=0.5)[0]
        self._draw_and_show(img, results, self._model_person.names, "Person Detection")

    # ── preset cam control ─────────────────────────────────────
    def preset_control(self, tilt, pan):
        self.set_ui_val("TiltAngle",    self.get_ui_val(tilt))
        self.set_ui_val("PanningAngle", self.get_ui_val(pan))
        self.parent_window.SetTilt()
        time.sleep(0.2)
        self.parent_window.SetPanning()

    def set_clicked_pos(self):
        self.set_ui_val("TiltAngle", self.get_ui_val("TgtTilt"))
        self.parent_window.SetTilt()
        time.sleep(0.2)
        pan = (self.get_ui_val("PanningAngle", float) or 0.0) + (self.get_ui_val("TgtPan", float) or 0.0)
        self.set_ui_val("PanningAngle", str(int(pan)))
        self.parent_window.SetPanning()

    def excute_first_action(self):
        self.parent_window.MotCancel()
        time.sleep(0.25)
        if not self.parent_window.sequential_cmd_list:
            return
        first_cmd = self.parent_window.sequential_cmd_list[0]
        self.parent_window.update_odometry = True
        self.parent_window.set_ui_val("SendStr", first_cmd)
        self.parent_window.SendStr()
        self.parent_window.sequential_cmd_list = self.parent_window.sequential_cmd_list[1:]
        self.setUIVal("SeqCmd", self.parent_window.sequential_cmd_list)

    def on_listbox_changed(self, key):
        selected = self.getUIVal(key)
        if selected:
            self.set_ui_val("SendStr", selected[0])

    # ── image click handler ────────────────────────────────────
    def _on_image_click(self, name, x, y):
        self.set_ui_val("ClickPosX", str(x))
        self.set_ui_val("ClickPosY", str(y))
        print(f"[click] name={name} x={x} y={y}")

        # ── TgtPan / TgtTilt 계산 ──────────────────────────────
        rgb = self.parent_window.ros_node.rgb_image
        if rgb is None:
            print("[click] rgb is None, skip")
            return
        img_h, img_w = rgb.shape[:2]
        tx = int(1280 * x / img_w)
        ty = int(720  * y / img_h)
        print(f"[click] tx={tx} ty={ty}")

        if not _init_thalamus():
            print("[click] thalamus not ready")
            return
        try:
            norm_vec = np.array(Pixelto3D(tx, ty, 1))
            print(f"[click] norm_vec={norm_vec}")
        except Exception as e:
            print(f"[click] Pixelto3D error: {e}")
            return

        nx, ny, nz = tuple(norm_vec)

        pan_angle  = np.degrees(np.arctan2(nx, nz))
        tilt_deg   = np.degrees(np.arctan2(-ny, np.sqrt(nx*nx + nz*nz)))

        current_tilt = self.get_ui_val("camera_pitch", float) or 0.0
        tilt_servo   = get_tilt_servo_angle(current_tilt + tilt_deg)
        pan_servo    = int(pan_angle - (self.get_ui_val("N", int) or 0))
        print(f"[click] pan_servo={pan_servo} tilt_servo={tilt_servo}")

        self.set_ui_val("TgtPan",  str(pan_servo))
        self.set_ui_val("TgtTilt", str(tilt_servo))

        # ── TgtPos / TgtDist / TgtBearing 계산 ────────────────
        def rotate_vector(v, pitch, roll, yaw):
            pitch, roll, yaw = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw)
            Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
            Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
            Rz = np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
            return Rz @ Ry @ Rx @ v

        def get_crosspnt(v, pos=700.0):
            v = np.asarray(v, dtype=float)
            ny_ = v[1]
            if abs(ny_) < 1e-8 or pos / ny_ <= 0:
                return np.array([0., 0., 0.])
            return v * (pos / ny_)

        odom_x   = self.parent_window.odom_x[-1]   if self.parent_window.odom_x   else 0.0
        odom_y   = self.parent_window.odom_y[-1]   if self.parent_window.odom_y   else 0.0
        odom_yaw = self.parent_window.odom_yaw[-1] if self.parent_window.odom_yaw else 90.0

        cam_pitch = self.get_ui_val("camera_pitch", float) or 0.0
        cam_pan   = (self.get_ui_val("PanningAngle", float) or 0.0) - (self.get_ui_val("N", float) or 0.0)
        norm_rot  = rotate_vector(norm_vec, cam_pitch, 0, odom_yaw + cam_pan)

        depth_src = self.get_ui_val("Depth_Src") or "Virtual_Plane"
        if depth_src == "Depth_Map":
            depth = self.parent_window.ros_node.depth_image
            if depth is not None:
                dh, dw = depth.shape
                dx_ = int(dw * x / img_w)
                dy_ = int(dh * y / img_h)
                tgt_depth_m = float(depth[dy_, dx_])
                if np.isfinite(tgt_depth_m) and tgt_depth_m > 0:
                    pnt_3d = norm_rot * tgt_depth_m * 1000.0  # m → mm
                else:
                    pnt_3d = get_crosspnt(norm_rot)
            else:
                pnt_3d = get_crosspnt(norm_rot)
        else:
            pnt_3d = get_crosspnt(norm_rot)

        self.set_ui_val("TgtPosX", f"{odom_y + pnt_3d[2] / 1000:.3f}")
        self.set_ui_val("TgtPosY", f"{pnt_3d[1] / 1000:.3f}")
        self.set_ui_val("TgtPosZ", f"{odom_x + pnt_3d[0] / 1000:.3f}")

        veh_dist = float(min(np.sqrt(pnt_3d[0]**2 + pnt_3d[2]**2), 2500))
        self.set_ui_val("TgtDist", f"{int(veh_dist)}")

        bearing_deg = int(pan_angle - (self.get_ui_val("N", int) or 0) + (self.get_ui_val("PanningAngle", int) or 0))
        self.set_ui_val("TgtBearing", f"{bearing_deg}")

        # ── Ctrl+클릭: SeqCmd 자동 생성 ────────────────────────
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            self.parent_window.sequential_cmd_list = []
            if abs(bearing_deg) > 1:
                spd = self.getUIVal("RotSpd")[0]
                rotate_dist = int(bearing_deg / 180 * rotate_pi_per_enc)
                self.parent_window.sequential_cmd_list.append(f"RT 1 {spd} {rotate_dist}")
            if veh_dist > 100:
                spd = self.getUIVal("MVSpd")[0]
                fwd_dist = int(veh_dist / 1000 * meter_per_enc)
                self.parent_window.sequential_cmd_list.append(f"MV 1 {spd} -{fwd_dist}")
            self.setUIVal("SeqCmd", self.parent_window.sequential_cmd_list)
