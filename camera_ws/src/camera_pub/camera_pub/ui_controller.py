import sys
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import QTimer

from camera_pub.ui_base import UIBase, UIElementType, UIElement
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from serial_interfaces.srv import SerialBypass


meter_per_enc = 11600
rotate_pi_per_enc = 4819


class UIControllerNode(Node):
    def __init__(self):
        super().__init__('ui_controller')
        self.cli = self.create_client(SerialBypass, 'serial_bypass')
        self.create_subscription(String, 'serial_message', self._on_serial_msg, 10)
        self.create_subscription(Vector3, 'camera_attitude', self._on_attitude, 10)

        self.attitude    = None  # Vector3
        self.motion_data = None  # (enc1, enc2, spd1, spd2, stat1, stat2)

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
            UIElement(_type=UIElementType.EDIT,   _caption="Distance",      _default_val="4646"),
            UIElement(_type=UIElementType.EDIT,   _caption="RotSpd",        _default_val="600"),
            UIElement(_type=UIElementType.EDIT,   _caption="RotAngle",      _default_val="4646"),
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
            UIElement(_type=UIElementType.BUTTON, _caption="reset_odometry", _default_val=self.reset_odometry),
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
        self.setWindowTitle("UI Controller")
        self.resize(1000, 1000)
        self.LoadPreset()

        # odometry state
        self.enc_val_reset()
        self.reset_odometry()
        self.update_odometry = False

        self.measure_timer = QTimer(self)
        self.measure_timer.start(50)
        self.measure_timer.timeout.connect(self.measure_timer_slot)

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

    # ── 50 ms update timer ─────────────────────────────────────
    def measure_timer_slot(self):
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

            x     = np.array(self.enc_idx)
            _enc1 = np.array(self.enc_dist1)
            _enc2 = np.array(self.enc_dist2)
            l_enc_min = min([_enc1.min(), _enc2.min()])
            l_enc_max = max([_enc1.max(), _enc2.max()])

            _spd1 = np.array(self.enc_spd1)
            _spd2 = np.array(self.enc_spd2)
            l_spd_min = min([_spd1.min(), _spd2.min()])
            l_spd_max = max([_spd1.max(), _spd2.max()])


            self.setUIVal("Graph1D_Enc", [
                (x, _enc1, ['l', l_enc_min, l_enc_max], "enc1(left)",  "r"),
                (x, _enc2, ['l', l_enc_min, l_enc_max], "enc2(right)",  "g"),
                (x, _spd1, ['r', l_spd_min, l_spd_max], "spd1(left)", "b"),
                (x, _spd2, ['r', l_spd_min, l_spd_max], "spd2(right)", "m"),
            ])

            _dist    = (_enc1 + _enc2) / 2 / meter_per_enc
            _yaw     = (_enc1 - _enc2) / rotate_pi_per_enc / 2 * 180
            _cam_yaw = np.array(self.cam_yaw_list) - self.cam_yaw_bias

            l_yaw_min = min([_yaw.min(), _cam_yaw.min()])
            l_yaw_max = max([_yaw.max(), _cam_yaw.max()])
            self.setUIVal("Graph1D_DistYaw", [
                (x, -_yaw,     ['l', l_yaw_min,    l_yaw_max],    "enc_yaw", "r"),
                (x, _cam_yaw, ['l', l_yaw_min,    l_yaw_max],    "cam_yaw", "g"),
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


def main(args=None):
    rclpy.init(args=args)
    ros_node = UIControllerNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    window = Window(ros_node)
    window.show()
    ret = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)
