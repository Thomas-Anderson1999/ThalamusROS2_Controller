import sys
import time
import threading

import rclpy
from rclpy.node import Node

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import QTimer

from camera_pub.ui_base import UIBase, UIElementType, UIElement
from std_msgs.msg import String
from serial_interfaces.srv import SerialBypass


meter_per_enc = 11600
rotate_pi_per_enc = 4819


class UIControllerNode(Node):
    def __init__(self):
        super().__init__('ui_controller')
        self.cli = self.create_client(SerialBypass, 'serial_bypass')
        self.create_subscription(String, 'serial_message', self._on_serial_msg, 10)

    def _on_serial_msg(self, msg):
        print(f'[serial] {msg.data}')

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

        start_ele_list = [
            UIElement(_type=UIElementType.BUTTON, _caption="SavePreset", _default_val=self.SavePreset),
            UIElement(_type=UIElementType.BUTTON, _caption="LoadPreset",  _default_val=self.LoadPreset),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="SendStr",     _default_val=""),
            UIElement(_type=UIElementType.BUTTON, _caption="SendStr",     _default_val=self.SendStr),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="TiltAngle",   _default_val="45"),
            UIElement(_type=UIElementType.BUTTON, _caption="SetTilt",     _default_val=self.SetTilt),
            UIElement(_type=UIElementType.EDIT,   _caption="PanningAngle",_default_val="0"),
            UIElement(_type=UIElementType.BUTTON, _caption="SetPanning",  _default_val=self.SetPanning),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.EDIT,   _caption="MVSpd",       _default_val="1600"),
            UIElement(_type=UIElementType.EDIT,   _caption="Distance",    _default_val="4646"),
            UIElement(_type=UIElementType.EDIT,   _caption="RotSpd",      _default_val="600"),
            UIElement(_type=UIElementType.EDIT,   _caption="RotAngle",    _default_val="4646"),
            UIElement(_type=UIElementType.NEW_LINE),
            UIElement(_type=UIElementType.BUTTON, _caption="MotCancel",   _default_val=self.MotCancel),
            UIElement(_type=UIElementType.BUTTON, _caption="MoveFoward",  _default_val=self.MoveFoward),
            UIElement(_type=UIElementType.BUTTON, _caption="MoveBackward",_default_val=self.MoveBackward),
            UIElement(_type=UIElementType.BUTTON, _caption="RotateLeft",  _default_val=self.RotateLeft),
            UIElement(_type=UIElementType.BUTTON, _caption="RotateRight", _default_val=self.RotateRight),
            UIElement(_type=UIElementType.NEW_LINE),
        ]
        subgrid, start_dic = self.createGroupBox("basic control", start_ele_list)
        self.ui_dic_list.append(start_dic)
        self.mainGrid.addWidget(subgrid, 0, 0)

        self.setLayout(self.mainGrid)
        self.setWindowTitle("UI Controller")
        self.resize(700, 200)
        self.LoadPreset()

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

    def MoveFoward(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("MVSpd")[0]
        dist = int(self.get_ui_val("Distance", float) / 1000 * meter_per_enc)
        self.set_ui_val("SendStr", f"MV 1 {spd} -{dist}")
        self.SendStr()

    def MoveBackward(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("MVSpd")[0]
        dist = int(self.get_ui_val("Distance", float) / 1000 * meter_per_enc)
        self.set_ui_val("SendStr", f"MV 1 {spd} {dist}")
        self.SendStr()

    def RotateLeft(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("RotSpd")[0]
        dist = int(self.get_ui_val("RotAngle", float) / 180 * rotate_pi_per_enc)
        self.set_ui_val("SendStr", f"RT 1 {spd} -{dist}")
        self.SendStr()

    def RotateRight(self):
        self.MotCancel()
        time.sleep(0.25)
        spd  = self.getUIVal("RotSpd")[0]
        dist = int(self.get_ui_val("RotAngle", float) / 180 * rotate_pi_per_enc)
        self.set_ui_val("SendStr", f"RT 1 {spd} {dist}")
        self.SendStr()


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
