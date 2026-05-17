import sys
import time
import copy
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import QTimer, Qt

from camera_pub.ui_base import UIBase, UIElementType, UIElement
from camera_pub.ui_utils import StreamingWindow
from camera_pub.ThalamusJoystick import ThalamusJoystick
from ThalamusEngine.Interface import LoadThalamusInterface, InitEngine, Pixelto3D
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import CompressedImage, Image
from serial_interfaces.srv import SerialBypass


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


class UIControllerNode(Node):
    def __init__(self):
        super().__init__('ui_controller')
        self.cli = self.create_client(SerialBypass, 'serial_bypass')
        self.create_subscription(String, 'serial_message', self._on_serial_msg, 10)
        self.create_subscription(Vector3, 'camera_attitude', self._on_attitude, 10)

        self.attitude    = None  # Vector3
        self.motion_data = None  # (enc1, enc2, spd1, spd2, stat1, stat2)
        self.rgb_image   = None  # numpy BGR
        self.depth_image = None  # numpy float32

        self.create_subscription(CompressedImage, '/camera/rgb/compressed', self._on_rgb, 10)
        self.create_subscription(Image, '/camera/depth', self._on_depth, 10)

    def _on_serial_msg(self, msg):
        print(f'[serial] {msg.data}')
        try:
            parts = msg.data.replace('\t', ' ').split()
            if len(parts) >= 8 and parts[0].startswith('PF>'):
                enc1  = -int(parts[0].replace('PF>', ''))
                enc2  =  int(parts[1])
                spd1  = -int(parts[2])
                spd2  =  int(parts[3])
                stat1 =  int(parts[6])
                stat2 =  int(parts[7])
                self.motion_data = (enc1, enc2, spd1, spd2, stat1, stat2)
        except Exception:
            pass

    def _on_attitude(self, msg):
        self.attitude = msg

    def _on_rgb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        self.rgb_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _on_depth(self, msg):
        self.depth_image = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)

    def send_serial(self, raw_str):
        if not self.cli.service_is_ready():
            self.get_logger().warn(f'serial_bypass not ready, dropped: {raw_str}')
            return
        req = SerialBypass.Request()
        req.data = '%-30sXX' % raw_str
        future = self.cli.call_async(req)
        future.add_done_callback(lambda f: self._on_done(f, raw_str))

    def _on_done(self, future, raw_str):
        try:
            res = future.result()
            if not res.success:
                self.get_logger().error(f'send failed [{raw_str}]: {res.message}')
        except Exception as e:
            self.get_logger().error(f'service call error: {e}')


class Window(QWidget, UIBase):
    def __init__(self, ros_node: UIControllerNode):
        super(Window, self).__init__()
        UIBase.__init__(self)

        self.ros_node = ros_node
        self.mainGrid = QGridLayout()
        self.ui_dic_list = []

        # ── basic control ──────────────────────────────────────
        start_ele_list = [
            UIElement(_type=UIElementType.BUTTON, _caption="SavePreset",    _default_val=self.SavePreset),
            UIElement(_type=UIElementType.BUTTON, _caption="LoadPreset",    _default_val=self.LoadPreset),
            UIElement(_type=UIElementType.BUTTON, _caption="Quit",          _default_val=self.quit_app),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="SendStr",       _default_val=""),
            UIElement(_type=UIElementType.BUTTON, _caption="SendStr",       _default_val=self.SendStr),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="TiltAngle",     _default_val="45"),
            UIElement(_type=UIElementType.BUTTON, _caption="SetTilt",       _default_val=self.SetTilt),
            UIElement(_type=UIElementType.EDIT,   _caption="PanningAngle",  _default_val="0"),
            UIElement(_type=UIElementType.BUTTON, _caption="SetPanning",    _default_val=self.SetPanning),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="MVSpd",         _default_val="1600"),
            UIElement(_type=UIElementType.EDIT,   _caption="Distance",      _default_val="500"),
            UIElement(_type=UIElementType.EDIT,   _caption="RotSpd",        _default_val="600"),
            UIElement(_type=UIElementType.EDIT,   _caption="RotAngle",      _default_val="90"),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.BUTTON, _caption="MotCancel",     _default_val=self.MotCancel),
            UIElement(_type=UIElementType.BUTTON, _caption="MoveFoward",    _default_val=self.MoveFoward),
            UIElement(_type=UIElementType.BUTTON, _caption="MoveBackward",  _default_val=self.MoveBackward),
            UIElement(_type=UIElementType.BUTTON, _caption="RotateLeft",    _default_val=self.RotateLeft),
            UIElement(_type=UIElementType.BUTTON, _caption="RotateRight",   _default_val=self.RotateRight),
            UIElement(_type=UIElementType.NEW_LINE),
        ]
        subgrid, start_dic = self.createGroupBox("basic control", start_ele_list)
        self.ui_dic_list.append(start_dic)
        self.mainGrid.addWidget(subgrid, 0, 0)

        # ── communication ──────────────────────────────────────
        start_ele_list = [
            UIElement(_type=UIElementType.BUTTON, _caption="open_streaming",  _default_val=self.open_streaming),
            UIElement(_type=UIElementType.BUTTON, _caption="reset_odometry",  _default_val=self.reset_odometry),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT, _caption="camera_roll",  _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="camera_pitch", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="camera_yaw",   _default_val="0"),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT, _caption="enc1",  _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="enc2",  _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="spd1",  _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="spd2",  _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="stat1", _default_val="0"),
            UIElement(_type=UIElementType.EDIT, _caption="stat2", _default_val="0"),
            UIElement(_type=UIElementType.NEW_LINE),
        ]
        subgrid, start_dic = self.createGroupBox("communication", start_ele_list)
        self.ui_dic_list.append(start_dic)
        self.mainGrid.addWidget(subgrid, 1, 0)

        # ── graph ──────────────────────────────────────────────
        start_ele_list = [
            UIElement(_type=UIElementType.GRAPH_1D, _caption="Graph1D_Enc",          _default_val=None, _defalt_row=1),
            UIElement(_type=UIElementType.GRAPH_1D, _caption="Graph1D_DistYaw",      _default_val=None, _defalt_row=1),
            UIElement(_type=UIElementType.GRAPH_2D_TRAJECTORY, _caption="Trajectory",_default_val=None, _defalt_row=1),
        ]
        subgrid, start_dic = self.createGroupBox("Graph Enc/Dist_Yaw/Trajectory", start_ele_list)
        self.ui_dic_list.append(start_dic)
        self.mainGrid.addWidget(subgrid, 2, 0)
        self.mainGrid.setRowMinimumHeight(2, 300)

        self.setLayout(self.mainGrid)
        self.setWindowTitle("Basic Control")
        self.resize(1000, 1000)
        self.LoadPreset()

        # odometry state
        self.enc_val_reset()
        self.reset_odometry()
        self.update_odometry = False
        self._streaming = False

        self.sequential_cmd_list = []

        self.measure_timer = QTimer(self)
        self.measure_timer.start(50)
        self.measure_timer.timeout.connect(self.measure_timer_slot)

        self.control_timer = QTimer(self)
        self.control_timer.start(80)
        self.control_timer.timeout.connect(self.control_timer_slot)

        self.joystick = ThalamusJoystick()
        self.old_hat = [0, 0]
        self.tilting_servo = -999999
        self.panning_servo = -999999
        self.joystick_timer = QTimer(self)
        self.joystick_timer.start(100)
        self.joystick_timer.timeout.connect(self.joystick_timer_slot)

    # ── odometry helpers ───────────────────────────────────────
    def enc_val_reset(self):
        self.enc_idx      = []
        self.enc_dist1    = []
        self.enc_dist2    = []
        self.enc_spd1     = []
        self.enc_spd2     = []
        self.cam_yaw_list = []
        self.cam_yaw_bias = None
        self.local_yaw    = 0

    def reset_odometry(self):
        self.enc_val_reset()
        self.odom_x   = [0]
        self.odom_y   = [0]
        self.odom_yaw = [90]
        self.setUIVal("Trajectory", [
            np.asarray([0.0]), np.asarray([0.0]), np.asarray([90.0]), 0
        ])

    # ── streaming ──────────────────────────────────────────────
    def open_streaming(self):
        self._streaming_window = StreamingWindow()
        self._streaming_window.set_click_cb(self._on_image_click)
        self._streaming_window.set_dblclick_cb(self.set_clicked_pos)
        self._streaming_window.show()
        self._streaming = True

    def _on_image_click(self, name, x, y):
        self.set_ui_val("ClickPosX", str(x))
        self.set_ui_val("ClickPosY", str(y))
        print(f"[click] name={name} x={x} y={y}")

        # ── TgtPan / TgtTilt 계산 ──────────────────────────────
        rgb = self.ros_node.rgb_image
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

        odom_x   = self.odom_x[-1]   if self.odom_x   else 0.0
        odom_y   = self.odom_y[-1]   if self.odom_y   else 0.0
        odom_yaw = self.odom_yaw[-1] if self.odom_yaw else 90.0

        cam_pitch = self.get_ui_val("camera_pitch", float) or 0.0
        cam_pan   = (self.get_ui_val("PanningAngle", float) or 0.0) - (self.get_ui_val("N", float) or 0.0)
        norm_rot  = rotate_vector(norm_vec, cam_pitch, 0, odom_yaw + cam_pan)

        depth_src = self.get_ui_val("Depth_Src") or "Virtual_Plane"
        if depth_src == "Depth_Map":
            depth = self.ros_node.depth_image
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
            self.sequential_cmd_list = []
            if abs(bearing_deg) > 1:
                spd = self.getUIVal("RotSpd")[0]
                rotate_dist = int(bearing_deg / 180 * rotate_pi_per_enc)
                self.sequential_cmd_list.append(f"RT 1 {spd} {rotate_dist}")
            if veh_dist > 100:
                spd = self.getUIVal("MVSpd")[0]
                fwd_dist = int(veh_dist / 1000 * meter_per_enc)
                self.sequential_cmd_list.append(f"MV 1 {spd} -{fwd_dist}")
            self.setUIVal("SeqCmd", self.sequential_cmd_list)

    # ── 50 ms update timer ─────────────────────────────────────
    def measure_timer_slot(self):
        if self._streaming:
            rgb = self.ros_node.rgb_image
            depth = self.ros_node.depth_image
            if rgb is not None:
                self._streaming_window.update_rgb(rgb)
            if depth is not None:
                d = depth.copy()
                d[~np.isfinite(d)] = 0
                valid = d[d > 0]
                if valid.size > 0:
                    mn, mx = np.percentile(valid, 1), np.percentile(valid, 99)
                    d = np.clip(d, mn, mx)
                    d = ((d - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    d = np.zeros_like(d, dtype=np.uint8)
                self._streaming_window.update_depth(cv2.applyColorMap(d, cv2.COLORMAP_TURBO))

        att = self.ros_node.attitude
        if att is not None:
            self.set_ui_val("camera_roll",  f"{att.x:.1f}")
            self.set_ui_val("camera_pitch", f"{att.y:.1f}")
            self.set_ui_val("camera_yaw",   f"{att.z:.1f}")

        md = self.ros_node.motion_data
        if md is None:
            return
        enc1, enc2, spd1, spd2, stat1, stat2 = md

        self.set_ui_val("enc1",  str(enc1))
        self.set_ui_val("enc2",  str(enc2))
        self.set_ui_val("spd1",  str(spd1))
        self.set_ui_val("spd2",  str(spd2))
        self.set_ui_val("stat1", str(stat1))
        self.set_ui_val("stat2", str(stat2))

        if stat1 == 0 or stat2 == 0:
            self.enc_val_reset()
        else:
            self.enc_idx.append(0 if not self.enc_idx else self.enc_idx[-1] + 1)
            self.enc_dist1.append(enc1)
            self.enc_dist2.append(enc2)
            self.enc_spd1.append(spd1)
            self.enc_spd2.append(spd2)
            att = self.ros_node.attitude
            cam_yaw = att.z if att is not None else (self.cam_yaw_list[-1] if self.cam_yaw_list else 0.0)
            if self.cam_yaw_bias is None:
                self.cam_yaw_bias = cam_yaw
            self.cam_yaw_list.append(cam_yaw)

            def return_rated_minmax(_min, _max, _rate=1.1):
                _ctr = (_min + _max) / 2
                _min = (_min - _ctr) * _rate + _ctr
                _max = (_max - _ctr) * _rate + _ctr
                return _min, _max

            x     = np.array(self.enc_idx)
            _enc1 = np.array(self.enc_dist1)
            _enc2 = np.array(self.enc_dist2)
            l_enc_min, l_enc_max = return_rated_minmax(min([_enc1.min(), _enc2.min()]),
                                                        max([_enc1.max(), _enc2.max()]))

            _spd1 = np.array(self.enc_spd1)
            _spd2 = np.array(self.enc_spd2)
            l_spd_min, l_spd_max = return_rated_minmax(min([_spd1.min(), _spd2.min()]),
                                                        max([_spd1.max(), _spd2.max()]))


            self.setUIVal("Graph1D_Enc", [
                (x, _enc1, ['l', l_enc_min, l_enc_max], "enc1(left)",  "r"),
                (x, _enc2, ['l', l_enc_min, l_enc_max], "enc2(right)",  "g"),
                (x, _spd1, ['r', l_spd_min, l_spd_max], "spd1(left)", "b"),
                (x, _spd2, ['r', l_spd_min, l_spd_max], "spd2(right)", "m"),
            ])

            _dist    = (_enc1 + _enc2) / 2 / meter_per_enc
            _yaw     = (_enc1 - _enc2) / rotate_pi_per_enc / 2 * 180
            _cam_yaw = np.array(self.cam_yaw_list) - self.cam_yaw_bias

            _neg_yaw = -_yaw
            l_yaw_min = min([_neg_yaw.min(), _cam_yaw.min()])
            l_yaw_max = max([_neg_yaw.max(), _cam_yaw.max()])

            l_yaw_min,l_yaw_max =  return_rated_minmax(l_yaw_min, l_yaw_max)

            self.setUIVal("Graph1D_DistYaw", [
                (x, _neg_yaw,  ['l', l_yaw_min, l_yaw_max], "enc_yaw", "r"),
                (x, _cam_yaw,  ['l', l_yaw_min, l_yaw_max], "cam_yaw", "g"),
            ])

            if self.update_odometry:
                ds    = (_dist[-1] - _dist[-2]) if len(_dist) >= 2 else 0.0
                d_yaw = (_yaw[-1]  - _yaw[-2])  if len(_yaw)  >= 2 else 0.0
                yaw_1 = self.odom_yaw[-1] + d_yaw
                self.local_yaw += d_yaw
                theta = np.deg2rad(yaw_1)
                self.odom_x.append(self.odom_x[-1]     + ds * np.cos(theta))
                self.odom_y.append(self.odom_y[-1]     + ds * np.sin(theta))
                self.odom_yaw.append(yaw_1)

                tx = np.asarray(self.odom_x)
                ty = np.asarray(self.odom_y)
                ty_ = np.asarray(self.odom_yaw)
                self.setUIVal("Trajectory", [tx, ty, ty_, len(tx) - 1])
                self.set_ui_val("odometry_X",      f"{self.odom_x[-1]:.3f}")
                self.set_ui_val("odometry_Y",      f"{self.odom_y[-1]:.3f}")
                self.set_ui_val("odometry_Yaw",    f"{self.odom_yaw[-1]:.3f}")
                self.set_ui_val("local_action_yaw", f"{int(self.local_yaw)}")
            else:
                self.enc_val_reset()

    # ── serial send helpers ────────────────────────────────────
    def _send(self, raw_str):
        self.ros_node.send_serial(raw_str)

    def SendStr(self):
        self._send(self.get_ui_val("SendStr"))

    def SetTilt(self):
        angle = int(min(self.get_ui_val("TiltAngle", float) / 90 * 1000, 1000))
        self.set_ui_val("SendStr", f"SV -1 {angle}")
        self.SendStr()

    def SetPanning(self):
        pan = 1500.0 - self.get_ui_val("PanningAngle", float) / 135.0 * 1000.0
        self.set_ui_val("SendStr", f"SV 1 {int(pan)}")
        self.SendStr()

    def MotCancel(self):
        self.set_ui_val("SendStr", "EN -1 1")
        self.SendStr()
        self.enc_val_reset()
        self.update_odometry = False

    def MoveFoward(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("MVSpd")[0]
        dist = int(self.get_ui_val("Distance", float) / 1000 * meter_per_enc)
        self.set_ui_val("SendStr", f"MV 1 {spd} -{dist}")
        self.SendStr()
        self.update_odometry = True

    def MoveBackward(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("MVSpd")[0]
        dist = int(self.get_ui_val("Distance", float) / 1000 * meter_per_enc)
        self.set_ui_val("SendStr", f"MV 1 {spd} {dist}")
        self.SendStr()
        self.update_odometry = True

    def RotateLeft(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("RotSpd")[0]
        dist = int(self.get_ui_val("RotAngle", float) / 180 * rotate_pi_per_enc)
        self.set_ui_val("SendStr", f"RT 1 {spd} -{dist}")
        self.SendStr()
        self.update_odometry = True

    def RotateRight(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("RotSpd")[0]
        dist = int(self.get_ui_val("RotAngle", float) / 180 * rotate_pi_per_enc)
        self.set_ui_val("SendStr", f"RT 1 {spd} {dist}")
        self.SendStr()
        self.update_odometry = True

    def preset_control(self, tilt, pan):
        self.set_ui_val("TiltAngle",    self.get_ui_val(tilt))
        self.set_ui_val("PanningAngle", self.get_ui_val(pan))
        self.SetTilt()
        time.sleep(0.2)
        self.SetPanning()

    def set_clicked_pos(self):
        self.set_ui_val("TiltAngle", self.get_ui_val("TgtTilt"))
        self.SetTilt()
        time.sleep(0.2)
        pan = (self.get_ui_val("PanningAngle", float) or 0.0) + (self.get_ui_val("TgtPan", float) or 0.0)
        self.set_ui_val("PanningAngle", str(int(pan)))
        self.SetPanning()

    # ── 80 ms control timer ────────────────────────────────────
    def control_timer_slot(self):
        mode = self.get_ui_val("CamTrackMode") or "None"

        if mode == "fix_cam_rotation":
            local_yaw  = self.get_ui_val("local_action_yaw", int) or 0
            n_offset   = self.get_ui_val("N", int) or 0
            target_pan = local_yaw + n_offset
            if target_pan != (self.get_ui_val("PanningAngle", int) or 0):
                self.set_ui_val("PanningAngle", f"{target_pan}")
                self.SetPanning()

        elif mode == "fix_on_target":
            odom_x   = self.odom_x[-1]   if self.odom_x   else None
            odom_y   = self.odom_y[-1]   if self.odom_y   else None
            odom_yaw = self.odom_yaw[-1] if self.odom_yaw else None
            if odom_x is None:
                return
            tgt_x = self.get_ui_val("TgtPosX", float) or 0.0
            tgt_y = self.get_ui_val("TgtPosY", float) or 0.0
            tgt_z = self.get_ui_val("TgtPosZ", float) or 0.0

            dx = tgt_x - odom_y
            dz = tgt_z - odom_x
            world_yaw     = -np.degrees(np.arctan2(dx, dz))
            panning_angle = int(world_yaw + (odom_yaw - 90))

            h_dist = np.sqrt(dx*dx + dz*dz)
            tilting_angle = 0.0 if h_dist < 1e-6 else -np.degrees(np.arctan2(tgt_y, h_dist))
            tilt_servo = get_tilt_servo_angle(tilting_angle)

            print(f"[fix_on_target] pan={panning_angle} tilt={tilt_servo}")
            if abs(panning_angle - (self.get_ui_val("PanningAngle", int) or 0)) > 1:
                self.set_ui_val("PanningAngle", f"{panning_angle}")
                self.SetPanning()
            elif abs(tilt_servo - (self.get_ui_val("TiltAngle", int) or 0)) > 1:
                self.set_ui_val("TiltAngle", f"{tilt_servo}")
                self.SetTilt()

    def excute_first_action(self):
        self.MotCancel()
        time.sleep(0.25)
        if not self.sequential_cmd_list:
            return
        first_cmd = self.sequential_cmd_list[0]
        self.update_odometry = True
        self.set_ui_val("SendStr", first_cmd)
        self.SendStr()
        self.sequential_cmd_list = self.sequential_cmd_list[1:]
        self.setUIVal("SeqCmd", self.sequential_cmd_list)

    def on_listbox_changed(self, key):
        selected = self.getUIVal(key)
        if selected:
            self.set_ui_val("SendStr", selected[0])

    # ── joystick ───────────────────────────────────────────────
    def KeepGoing(self):
        pass

    def Stopping(self):
        pass

    def joystick_timer_slot(self):
        hat, btn, axis = self.joystick.get_status()

        if len(btn) > 0:
            if btn[4] == 1 or btn[5] == 1:  # L1/R1: 모션 취소
                self.MotCancel()

            if btn[6] == 1:  # L2: 틸트 + 패닝
                if len(axis) > 4:
                    servo_tilt_angle = int(45 + axis[4] * 45)

                    #+하드웨어 특성에 따른 수정 : 내가 가진 조이스틱이 오른쪽 아날로그가 최대 0.41밖에 안나옴
                    raw_pan = axis[3]
                    if raw_pan > 0:
                        raw_pan = min(raw_pan / 0.41, 1.0)
                    servo_panning_angle = int(raw_pan * 135)
                    #-하드웨어 특성에 따른 수정

                    if int(self.get_ui_val("TiltAngle", float) or 0) != servo_tilt_angle:
                        self.set_ui_val("TiltAngle", str(servo_tilt_angle))
                        self.SetTilt()
                    elif int(self.get_ui_val("PanningAngle", float) or 0) != servo_panning_angle:
                        self.set_ui_val("PanningAngle", str(servo_panning_angle))
                        self.SetPanning()

            #print("axis:", " ".join(f"{v:6.2f}" for v in axis))

        # D-pad 전진 / 후진
        if hat[1] == 1:
            if self.old_hat[1] == hat[1]:
                self.KeepGoing()
            else:
                self.MoveFoward()
        elif hat[1] == -1:
            if self.old_hat[1] == hat[1]:
                self.KeepGoing()
            else:
                self.MoveBackward()
        elif hat[1] == 0:
            if self.old_hat[1] != hat[1]:
                self.Stopping()

        # D-pad 회전
        if hat[0] == 1:
            if self.old_hat[0] == hat[0]:
                self.KeepGoing()
            else:
                self.RotateRight()
        elif hat[0] == -1:
            if self.old_hat[0] == hat[0]:
                self.KeepGoing()
            else:
                self.RotateLeft()
        elif hat[0] == 0:
            if self.old_hat[0] != hat[0]:
                self.Stopping()

        # 버튼 3: 프리셋
        if len(btn) > 3:
            if btn[3] == 1:
                self.set_ui_val("SendStr", "SV 1 1425")
                self.SendStr()
                time.sleep(0.3)
                self.set_ui_val("SendStr", "SV -1 211")
                self.SendStr()

        self.old_hat = copy.deepcopy(hat)

    def quit_app(self):
        self.joystick_timer.stop()
        self.control_timer.stop()
        self.measure_timer.stop()
        QApplication.quit()


class PresetCamWindow(QWidget, UIBase):
    def __init__(self, parent_window: Window):
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
                    _default_val=lambda state, _t=t, _p=p: self.parent_window.preset_control(_t, _p),
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
            UIElement(_type=UIElementType.BUTTON, _caption="set_clicked_pos", _default_val=self.parent_window.set_clicked_pos),

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
            UIElement(_type=UIElementType.BUTTON, _caption="excute_first_action", _default_val=self.parent_window.excute_first_action),

            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.LISTBOX, _caption="SeqCmd", _default_val=None, _defalt_row=3, _defalt_culumn=8),
        ]
        subgrid2, dic2 = self.createGroupBox("Semi-Auto-Control", info_ele_list)
        self.ui_dic_list.append(dic2)
        self.bind_listbox_changed(self.parent_window.on_listbox_changed)
        mainGrid.addWidget(subgrid2, 1, 0)

        self.setLayout(mainGrid)
        self.setWindowTitle("Auto Control")
        self.resize(1200, 400)


def main(args=None):
    _init_thalamus()
    rclpy.init(args=args)
    ros_node = UIControllerNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    window = Window(ros_node)
    window.show()
    preset_cam_window = PresetCamWindow(window)
    preset_cam_window.show()
    ret = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)
