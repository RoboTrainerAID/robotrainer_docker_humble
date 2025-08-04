#!/bin/sh

CONTAINER_NAME=robotrainer_polar_oh1
CONTAINER_TAG=humble

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    -t ${CONTAINER_NAME}:${CONTAINER_TAG} \
    .

    # --no-cache \
    # --progress plain \
    # --build-arg CACHE_BUST="$(date +%s)" \