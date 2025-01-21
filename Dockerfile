##############################################################################
##                                 Base Image                               ##
##############################################################################
ARG ROS_DISTRO=humble
FROM ros:$ROS_DISTRO-ros-base
ENV TZ=Europe/Berlin
ENV TERM=xterm-256color
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl htop nano net-tools iputils-ping sudo wget \
    python3-pip \
    python3-colcon-common-extensions \
    python3-argcomplete \
    python3-sphinx \
    gdb \
    ros-$ROS_DISTRO-behaviortree-cpp-v3 \
    ros-$ROS_DISTRO-nav2-msgs \
    qtbase5-dev \
    libqt5svg5-dev \
    libzmq3-dev \
    libdw-dev \
    libqt5opengl5-dev \
    qttools5-dev-tools \
    ros-$ROS_DISTRO-rviz* \
    ros-$ROS_DISTRO-rqt* \
    && apt-get clean && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install -U pip setuptools

# For ros2_control
RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-$ROS_DISTRO-ros2-control* \
    liburdfdom-tools \
    ros-$ROS_DISTRO-xacro \
    ros-$ROS_DISTRO-joint-state-publisher-gui \
    ros-$ROS_DISTRO-gazebo-ros* \
    && apt-get clean && rm -rf /var/lib/apt/lists/* 
# ros-$ROS_DISTRO-gazebo-ros2-control \
# ros-$ROS_DISTRO-ros2-controllers-test-nodes \
# ros-$ROS_DISTRO-ros2-controllers \

# For Gazebo/Ignition simulation
RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-$ROS_DISTRO-ign* \ 
    ros-$ROS_DISTRO-ros-ign* \ 
    ros-$ROS_DISTRO-ros-gz* \ 
    && apt-get clean && rm -rf /var/lib/apt/lists/* 
# ros-$ROS_DISTRO-ros-ign-gazebo \ 

ENV GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/docker/ros2_ws/src
ENV GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/docker/ros2_ws/src
ENV GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:/home/docker/ros2_ws/src
RUN echo "source /usr/share/gazebo/setup.bash" >> /home/$USER/.bashrc

# Jazzy
# ros-$ROS_DISTRO-ros-gz-sim
# ros-$ROS_DISTRO-gz-ros2-control

# VSCode server with pre-installed extensions
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server \
    --install-extension ms-python.autopep8 \
    --install-extension ms-python.python

##############################################################################
##                                 Create User                              ##
##############################################################################
ARG USER=docker
ARG PASSWORD=docker
ARG UID=1001
ARG GID=1001
ENV UID=$UID
ENV GID=$GID
ENV USER=$USER
RUN groupadd -g "$GID" "$USER"  && \
    useradd -m -u "$UID" -g "$GID" --shell $(which bash) "$USER" -G sudo && \
    echo "$USER:$PASSWORD" | chpasswd && \
    echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp && \
    chmod 0440 /etc/sudoers.d/sudogrp && \
    chown ${UID}:${GID} -R /home/${USER}
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /etc/bash.bashrc

# Set ROS2 DDS profile
COPY dds_profile.xml /home/$USER
RUN chown $USER:$USER /home/$USER/dds_profile.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/home/$USER/dds_profile.xml

USER $USER 
RUN mkdir -p /home/$USER/ros2_ws/src

##############################################################################
##                                 User Dependecies                         ##
##############################################################################
WORKDIR /home/$USER/ros2_ws/src
# ARG CACHE_BUST
RUN git clone https://github.com/BehaviorTree/Groot.git
RUN git clone -b humble https://github.com/AndreasZachariae/BehaviorTree.IRAS.git

RUN git clone -b humble https://github.com/ros-controls/ros2_control_demos
RUN git clone -b master https://github.com/StoglRobotics/ros_team_workspace.git
RUN echo "source /home/$USER/ros2_ws/src/ros_team_workspace/setup.bash" >> /home/$USER/.bashrc
ENV ROS_WS=/home/$USER/ros2_ws

COPY src/robotrainer3_description ./robotrainer3_description
COPY src/robotrainer3_bringup ./robotrainer3_bringup

USER $USER 

##############################################################################
##                                 Build ROS and run                        ##
##############################################################################
WORKDIR /home/$USER/ros2_ws
RUN rosdep update --rosdistro $ROS_DISTRO
RUN rosdep install --from-paths src --ignore-src -y
RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
RUN echo "source /home/$USER/ros2_ws/install/setup.bash" >> /home/$USER/.bashrc

RUN sudo sed --in-place --expression \
    '$i\source "/home/$USER/ros2_ws/install/setup.bash"' \
    /ros_entrypoint.sh

RUN sudo sed --in-place --expression \
    '$i\source "/usr/share/gazebo/setup.bash"' \
    /ros_entrypoint.sh

CMD ["bash"]
