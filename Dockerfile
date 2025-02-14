##############################################################################
##                                 Base Image                               ##
##############################################################################
ARG ROS_DISTRO=humble
# Ubuntu 22.04.
FROM osrf/ros:${ROS_DISTRO}-desktop
ENV TZ=Europe/Berlin
ENV TERM=xterm-256color
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

##############################################################################
##                                  User                                    ##
##############################################################################
ARG USER=docker
ARG PASSWORD=docker
ARG UID=1000
ARG GID=1000
ENV USER=${USER}
RUN groupadd -g ${GID} ${USER} && \
    useradd -m -u ${UID} -g ${GID} -p "$(openssl passwd -1 ${PASSWORD})" --shell $(which bash) ${USER} -G sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp
RUN usermod -a -G video ${USER}

# Set ROS2 DDS profile
COPY ./dds_profile.xml /home/${USER}
RUN chown ${USER}:${USER} /home/${USER}/dds_profile.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/home/${USER}/dds_profile.xml

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################
# Install default packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    iputils-ping nano htop git sudo wget curl gedit python3-pip gdb bluez bluetooth \
    && rm -rf /var/lib/apt/lists/*

# Install custom dependencies
# RUN apt-get update && apt-get install --no-install-recommends -y \
#     <YOUR_PACKAGE> \
#     && rm -rf /var/lib/apt/lists/*

RUN pip install \
    pexpect 
    


##############################################################################
##                                 dependencies_ws                          ##
##############################################################################
USER ${USER}
RUN mkdir -p /home/${USER}/dependencies_ws/src
WORKDIR /home/${USER}/dependencies_ws/src

# ARG CACHE_BUST
# RUN git clone --branch <BRANCH> <REPO_URL>
RUN git clone https://github.com/BehaviorTree/Groot.git
RUN git clone --branch humble https://github.com/AndreasZachariae/BehaviorTree.IRAS.git

# Build dependencies_ws
WORKDIR /home/${USER}/dependencies_ws
RUN rosdep update --rosdistro ${ROS_DISTRO}
USER root
RUN apt-get update 
RUN rosdep install --from-paths src --ignore-src -r -y
RUN rm -rf /var/lib/apt/lists/*
USER ${USER}
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build
RUN echo "source /home/${USER}/dependencies_ws/install/setup.bash" >> /home/$USER/.bashrc

##############################################################################
##                                 ros_ws                                   ##
##############################################################################
RUN mkdir -p /home/${USER}/ros_ws/src
WORKDIR /home/${USER}/ros_ws

# COPY <HOST_PATH> <CONTAINER_PATH>
COPY ./src ./src

# Build ros_ws
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    . /home/${USER}/dependencies_ws/install/setup.sh && \
    colcon build --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
RUN echo "source /home/${USER}/ros_ws/install/setup.bash" >> /home/${USER}/.bashrc

##############################################################################
##                                 Autostart                                ##
##############################################################################
RUN sudo sed --in-place --expression \
    '$isource "/home/${USER}/dependencies_ws/install/setup.bash"' \
    /ros_entrypoint.sh

RUN sudo sed --in-place --expression \
    '$isource "/home/${USER}/ros_ws/install/setup.bash"' \
    /ros_entrypoint.sh

CMD ["bash"]
