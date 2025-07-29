import os
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_prefix('robotrainer_bayesian_optimization'),
        '..',
        '..',
        'src',
        'robotrainer_bayesian_optimization',
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='robotrainer_bayesian_optimization',
            executable='robotrainer_bayesian_optimization',
            name='robotrainer_bayesian_optimization',
            output='screen',
            prefix=['stdbuf -o L'],
            parameters=[config]
        )
    ])