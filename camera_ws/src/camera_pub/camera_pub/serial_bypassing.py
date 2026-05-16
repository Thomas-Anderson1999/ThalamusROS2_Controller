import rclpy
from rclpy.node import Node
import serial

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

    def callback(self, request, response):
        try:
            self.ser.write(request.data.encode())
            response.success = True
            response.message = f'Sent {len(request.data)} bytes'
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f'Serial write failed: {e}')
        return response

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
