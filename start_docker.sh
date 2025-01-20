#!/bin/sh

# Autostart command to run inside the container, default is bash
# Usage1: Modify ./autostart.sh file and add custom command there
# Usage2: Run from cli with ./start_docker "custom command"
COMMAND=${1:-bash}
CONTAINER_NAME=robotrainer_humble
ROS_DOMAIN_ID=36
PYTHONPATH=./:install/lib/python3.10/site-packages

# Ensure XAUTHORITY is set
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

docker run \
    --name ${CONTAINER_NAME} \
    --privileged \
    -it \
    --net host \
    --rm \
    -e DISPLAY=${DISPLAY} \
    -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID} \
    -e PYTHONPATH=${PYTHONPATH} \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=${XAUTHORITY} \
    -v $XAUTHORITY:$XAUTHORITY:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD/.vscode:/home/docker/ros2_ws/src/.vscode \
    -v $PWD/src/robotrainer3_description:/home/docker/ros2_ws/src/robotrainer3_description \
    -v $PWD/src/robotrainer3_bringup:/home/docker/ros2_ws/src/robotrainer3_bringup \
    -v /dev:/dev  \
    ${CONTAINER_NAME}:humble \
    ${COMMAND}

    # --env-file .env \
    # libEGL for Gazebo needs access to /dev/dri/renderD129
    # -v /dev:/dev \
    # -v /lib/modules:/lib/modules:ro \