#!/bin/sh

# Only detached mode to call from another bash script in the background
COMMAND=${1:-bash}
CONTAINER_NAME=robotrainer_polar_oh1
CONTAINER_TAG=humble
ROS_DOMAIN_ID=36
PYTHONPATH=./:install/lib/python3.10/site-packages

# Check if the container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running. Attaching to it..."
    docker exec -it ${CONTAINER_NAME} ${COMMAND}
    exit 0
fi

docker run \
    --name ${CONTAINER_NAME} \
    --privileged \
    --net host \
    --rm \
    -d \
    -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID} \
    -e PYTHONPATH=${PYTHONPATH} \
    -v $PWD/src:/home/docker/ros_ws/src \
    -v /dev:/dev  \
    -v /var/run/dbus:/var/run/dbus \
    ${CONTAINER_NAME}:${CONTAINER_TAG} \
    /bin/bash -c "ros2 launch polar_oh1 ros2-polar_oh1.launch.py"
    