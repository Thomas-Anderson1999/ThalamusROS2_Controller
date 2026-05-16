import rclpy
from rclpy.node import Node
import time

from serial_interfaces.srv import SerialBypass


def fmt(s):
    return "%-30sXX" % s


class SerialBypassTest(Node):
    def __init__(self):
        super().__init__('serial_bypass_test')
        self.cli = self.create_client(SerialBypass, 'serial_bypass')
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for serial_bypass service...')

    def send(self, raw_str):
        data = fmt(raw_str)
        req = SerialBypass.Request()
        req.data = data
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.get_logger().info(f'sent: {repr(data)} -> success={res.success}, msg={res.message}')


def main(args=None):
    rclpy.init(args=args)
    node = SerialBypassTest()

    cmds = ["SV -1 500", "SV -1 722"]

    for i in range(3):
        node.get_logger().info(f'--- round {i+1}/3 ---')
        for cmd in cmds:
            node.send(cmd)
            time.sleep(1.0)

    node.destroy_node()
    rclpy.shutdown()
