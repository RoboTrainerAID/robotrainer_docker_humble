import os
import datetime
from ament_index_python.packages import get_package_share_directory , get_search_paths
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch import LaunchDescription, Action
import launch


def generate_launch_description():  
    # Subject Information
    Subject_Number = "P1"

    # create timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # path to save data with timestamp
    base_path = "/home/docker/ros_ws/src/data/"
    output_path = os.path.join(base_path, timestamp)

    ###### Physiological Sensor
    ros2_foxy_polar_oh1_node = Node(
            package='polar_oh1', 
            #namespace='Subject_Number',
            executable='polar_oh1_node',
            name='polar_oh1_node',
            #output='screen',
            parameters=[{'Sensor_Enable': True, 
            'Chunk_Enable': True,
            'Chunk_Length': 10,
            ### For sensor devices
            'Device_Mac_Address': 'A0:9E:1A:E0:BC:97',
            }] 
        )

    ros2_bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record','-o', output_path, '/biosensors/polar_oh1/hr', '/biosensors/polar_oh1/battery', 
             '/biosensors/polar_oh1/ppg_ch0', '/biosensors/polar_oh1/ppg_ch1', 
             '/biosensors/polar_oh1/ppg_ch2', '/biosensors/polar_oh1/ppg_ch3', 
             '/biosensors/polar_oh1/ppi', '/biosensors/polar_oh1/hrv'],
        output='both'
    )
  
    return LaunchDescription([
        ros2_foxy_polar_oh1_node,
        ros2_bag_record,        
    ])
