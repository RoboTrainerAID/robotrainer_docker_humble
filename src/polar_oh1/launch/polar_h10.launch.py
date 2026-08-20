# Copyright 2026 Hochschule Karlsruhe IRAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Launch the Polar H10 driver.

Recording is deliberately not started here: the topics are bridged to ROS 1
and recorded on that side.
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [get_package_share_directory('polar_oh1'), 'config', 'polar_h10.yaml'])

    args = [
        DeclareLaunchArgument('config', default_value=default_config,
                              description='Parameter file for the driver.'),
        DeclareLaunchArgument(
            'address', default_value='',
            description='MAC address or name substring; overrides the config file when set.'),
    ]

    def make_driver(context, *_args, **_kwargs):
        # Only override the address when one was actually given, otherwise an
        # empty launch argument would wipe the value from the config file.
        parameters = [LaunchConfiguration('config').perform(context)]
        address = LaunchConfiguration('address').perform(context)
        if address:
            parameters.append({'address': address})
        return [Node(
            package='polar_oh1',
            executable='polar_h10_node',
            name='polar_h10_node',
            output='screen',
            parameters=parameters,
        )]

    return LaunchDescription(args + [OpaqueFunction(function=make_driver)])
