from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_pub',
            executable='serial_bypassing',
        ),
        Node(
            package='camera_pub',
            executable='camera_node_d455i',
            parameters=[{'show_preview': False}],
        ),
    ])
