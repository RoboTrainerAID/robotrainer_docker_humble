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
"""Launch the Polar OH1+ driver."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    driver = Node(
        package='polar_oh1',
        executable='polar_oh1_node',
        name='polar_oh1_node',
        output='screen',
        parameters=[{
            'Device_Mac_Address': 'A0:9E:1A:E0:BC:97',
            'publish_ppg': True,
            # Disabling PPI keeps heart rate at 1 Hz instead of dropping it to
            # one update every 5 s (documented Polar OH1 behaviour).
            'publish_ppi': True,
            'ppg_decimation': 1,
        }],
    )
    return LaunchDescription([driver])
