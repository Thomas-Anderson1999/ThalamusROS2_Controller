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
from camera_pub.AutoControl import (
    PresetCamWindow, _init_thalamus,
    meter_per_enc, rotate_pi_per_enc, get_tilt_servo_angle,
)
from camera_pub.img_proc_main import reset_img_proc_odometry, stop_vio, set_vio_cmd_type, get_last_vio_yaw
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import CompressedImage, Image
from serial_interfaces.srv import SerialBypass


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
            UIElement(_type=UIElementType.BUTTON, _caption="Auto-control", _default_val=self.focus_auto_window),
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
            UIElement(_type=UIElementType.EDIT, _caption="odom_yaw",     _default_val="0"),
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
        self._prev_moving = False
        self._streaming = False
        self._click_cb       = None
        self._dblclick_cb    = None
        self._preset_cam_win = None

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

    def register_auto_callbacks(self, click_cb, dblclick_cb):
        self._click_cb    = click_cb
        self._dblclick_cb = dblclick_cb

    def register_preset_window(self, w):
        self._preset_cam_win = w

    def focus_auto_window(self):
        if self._preset_cam_win is not None:
            self._preset_cam_win.hide()
            self._preset_cam_win.show()

    # ── odometry helpers ───────────────────────────────────────
    def enc_val_reset(self):
        self.enc_idx      = []
        self.enc_dist1    = []
        self.enc_dist2    = []
        self.enc_spd1     = []
        self.enc_spd2     = []
        self.cam_yaw_list = []
        self.cam_yaw_bias = None
        self.cam_yaw_prev = None
        self.cam_yaw_accum = 0.0
        self.local_yaw    = 0
        self.vio_yaw_list = []  # enc_idx와 길이 동기화

    def vio_yaw_reset(self):
        self.vio_yaw_list = []
        self.vio_yaw_bias = None

    def reset_odometry(self):
        self.enc_val_reset()
        self.vio_yaw_reset()
        self.odom_x   = [0]
        self.odom_y   = [0]
        self.odom_yaw = [90]
        self.setUIVal("Trajectory", [
            np.asarray([0.0]), np.asarray([0.0]), np.asarray([90.0]), 0
        ])

    # ── streaming ──────────────────────────────────────────────
    def open_streaming(self):
        self._streaming_window = StreamingWindow()
        if self._click_cb:
            self._streaming_window.set_click_cb(self._click_cb)
        if self._dblclick_cb:
            self._streaming_window.set_dblclick_cb(self._dblclick_cb)
        self._streaming_window.show()
        self._streaming = True

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

        is_moving = (stat1 != 0 and stat2 != 0)
        if not is_moving:
            self.enc_val_reset()
            if self._prev_moving:
                stop_vio()
        else:
            # enc 값이 갑자기 크게 감소하면 펌웨어 리셋으로 간주하고 초기화
            if self.enc_dist1 and enc1 < self.enc_dist1[-1] - 1000:
                print(f"[odom] enc 불연속 감지 ({self.enc_dist1[-1]} → {enc1}), enc_val_reset")
                self.enc_val_reset()
            self.enc_idx.append(0 if not self.enc_idx else self.enc_idx[-1] + 1)
            self.enc_dist1.append(enc1)
            self.enc_dist2.append(enc2)
            self.enc_spd1.append(spd1)
            self.enc_spd2.append(spd2)
            att = self.ros_node.attitude
            cam_yaw = att.z if att is not None else (self.cam_yaw_prev if self.cam_yaw_prev is not None else 0.0)
            if self.cam_yaw_prev is not None:
                delta_raw = cam_yaw - self.cam_yaw_prev
                delta = (delta_raw + 180) % 360 - 180  # normalize to [-180, 180]
                if abs(delta) <= 50:  # 50ms 내 50° 초과 = AHRS 글리치, 스킵
                    self.cam_yaw_accum += delta
            self.cam_yaw_prev = cam_yaw
            self.cam_yaw_list.append(self.cam_yaw_accum)
            vio_yaw_raw = get_last_vio_yaw()
            if self.vio_yaw_bias is None:
                self.vio_yaw_bias = vio_yaw_raw
            self.vio_yaw_list.append(vio_yaw_raw - self.vio_yaw_bias)

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
            _cam_yaw = np.array(self.cam_yaw_list)

            _neg_yaw = -_yaw
            _vio_yaw = np.array(self.vio_yaw_list)
            l_yaw_min = min([_neg_yaw.min(), _cam_yaw.min(), _vio_yaw.min()])
            l_yaw_max = max([_neg_yaw.max(), _cam_yaw.max(), _vio_yaw.max()])

            l_yaw_min, l_yaw_max = return_rated_minmax(l_yaw_min, l_yaw_max)

            self.setUIVal("Graph1D_DistYaw", [
                (x, _neg_yaw,  ['l', l_yaw_min, l_yaw_max], "odom_yaw", "r"),
                (x, _cam_yaw,  ['l', l_yaw_min, l_yaw_max], "imu_yaw", "g"),
                (x, _vio_yaw,  ['l', l_yaw_min, l_yaw_max], "vio_yaw", "b"),
            ])
            self.set_ui_val("odom_yaw", f"{self.odom_yaw[-1]:.1f}")

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
                if self._preset_cam_win is not None:
                    self._preset_cam_win.update_local(
                        self.odom_x[-1], self.odom_y[-1], self.odom_yaw[-1])
            else:
                self.enc_val_reset()

        self._prev_moving = is_moving

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
        stop_vio()

    def MoveFoward(self):
        self.MotCancel()
        reset_img_proc_odometry()
        self.vio_yaw_reset()
        set_vio_cmd_type("MOVE")
        time.sleep(0.25)
        spd  = self.getUIVal("MVSpd")[0]
        dist = int(self.get_ui_val("Distance", float) / 1000 * meter_per_enc)
        self.set_ui_val("SendStr", f"MV 1 {spd} -{dist}")
        self.SendStr()
        self.update_odometry = True

    def MoveBackward(self):
        self.MotCancel()
        reset_img_proc_odometry()
        self.vio_yaw_reset()
        set_vio_cmd_type("MOVE")
        time.sleep(0.25)
        spd  = self.getUIVal("MVSpd")[0]
        dist = int(self.get_ui_val("Distance", float) / 1000 * meter_per_enc)
        self.set_ui_val("SendStr", f"MV 1 {spd} {dist}")
        self.SendStr()
        self.update_odometry = True

    def RotateLeft(self):
        self.MotCancel()
        reset_img_proc_odometry()
        self.vio_yaw_reset()
        set_vio_cmd_type("ROT")
        time.sleep(0.25)
        spd  = self.getUIVal("RotSpd")[0]
        dist = int(self.get_ui_val("RotAngle", float) / 180 * rotate_pi_per_enc)
        self.set_ui_val("SendStr", f"RT 1 {spd} -{dist}")
        self.SendStr()
        self.update_odometry = True

    def RotateRight(self):
        self.MotCancel()
        reset_img_proc_odometry()
        self.vio_yaw_reset()
        set_vio_cmd_type("ROT")
        time.sleep(0.25)
        spd  = self.getUIVal("RotSpd")[0]
        dist = int(self.get_ui_val("RotAngle", float) / 180 * rotate_pi_per_enc)
        self.set_ui_val("SendStr", f"RT 1 {spd} {dist}")
        self.SendStr()
        self.update_odometry = True

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
        cv2.destroyAllWindows()
        QApplication.quit()


def main(args=None):
    _init_thalamus()
    rclpy.init(args=args)
    ros_node = UIControllerNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    window = Window(ros_node)
    preset_cam_window = PresetCamWindow(window)
    window.register_preset_window(preset_cam_window)
    window.register_auto_callbacks(
        preset_cam_window._on_image_click,
        preset_cam_window.set_clicked_pos,
    )
    window.show()
    preset_cam_window.show()
    ret = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)
