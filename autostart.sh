#!/bin/bash

./start_docker.sh "/bin/bash -c './src/polar_oh1/connect_polar.sh && sleep 10 && ros2 launch polar_oh1 ros2-polar_oh1.launch.py'"