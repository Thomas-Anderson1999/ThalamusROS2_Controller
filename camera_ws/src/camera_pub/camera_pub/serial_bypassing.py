import threading

import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import String

from serial_interfaces.srv import SerialBypass


class SerialBypassingNode(Node):
    def __init__(self):
        super().__init__('serial_bypassing')

        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)

        port = self.get_parameter('port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value

        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.get_logger().info(f'Serial opened: {port} @ {baudrate}')

        self.srv = self.create_service(SerialBypass, 'serial_bypass', self.callback)
        self.get_logger().info('Service ready: serial_bypass')

        self.pub = self.create_publisher(String, 'serial_message', 10)

        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    def callback(self, request, response):
        try:
            data = request.data.encode('ascii')
            self.ser.write(data)
            self.ser.flush()
            response.success = True
            response.message = f'Sent {len(data)} bytes'
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f'Serial write failed: {e}')
        return response

    def _read_loop(self):
        while rclpy.ok():
            try:
                line = self.ser.readline()
                if line:
                    msg = String()
                    msg.data = line.decode(errors='replace').rstrip()
                    self.pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Serial read error: {e}')
                break

    def destroy_node(self):
        if self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBypassingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
