#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# USAGE: PYAERIAL_IMAGE=<image> $cuBB_SDK/pyaerial/container/start_daemon.sh
#
# Start PyAerial container in daemon mode for VS Code Dev Containers attach.
# The container will run in the background and restart automatically unless manually stopped.

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
host_cuBB_SDK=$(builtin cd $SCRIPT_DIR/../..;pwd)

echo "=== PyAerial Daemon Launcher ==="
echo "Starting from: $SCRIPT"
source $host_cuBB_SDK/cuPHY-CP/container/setup.sh

AERIAL_PLATFORM=${AERIAL_PLATFORM:-amd64}
TARGETARCH=$(basename $AERIAL_PLATFORM)

if [[ -z $PYAERIAL_IMAGE ]]; then
   PYAERIAL_IMAGE=pyaerial:${USER}-${AERIAL_VERSION_TAG}-${TARGETARCH}
fi

CONTAINER_NAME=pyaerial_$USER

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "✓ Container '$CONTAINER_NAME' is already running"
        echo ""
        echo "To attach in VS Code:"
        echo "  1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)"
        echo "  2. Select 'Dev Containers: Attach to Running Container...'"
        echo "  3. Choose '$CONTAINER_NAME'"
        echo ""
        echo "Or use Remote Explorer → Dev Containers → $CONTAINER_NAME"
        exit 0
    else
        echo "Container exists but stopped. Removing..."
        docker rm $CONTAINER_NAME
    fi
fi

echo "Starting container '$CONTAINER_NAME' in background..."
echo "Image: $PYAERIAL_IMAGE"

docker run --privileged \
            -d \
            --restart unless-stopped \
            $AERIAL_EXTRA_FLAGS \
            --gpus all \
            --name $CONTAINER_NAME \
            --add-host $CONTAINER_NAME:127.0.0.1 \
            --network host --shm-size=4096m \
            --device=/dev/gdrdrv:/dev/gdrdrv \
            -u $USER_ID:$GROUP_ID \
            -w /opt/nvidia/cuBB \
            -v $host_cuBB_SDK:/opt/nvidia/cuBB \
            -v $host_cuBB_SDK:/opt/nvidia/aerial_sdk \
            -v /dev/hugepages:/dev/hugepages \
            -v /lib/modules:/lib/modules \
            -v /var/log/aerial:/var/log/aerial \
            -e host_cuBB_SDK=$host_cuBB_SDK \
            --userns=host --ipc=host \
            $PYAERIAL_IMAGE tail -f /dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Container started successfully!"
    echo ""
    echo "Next steps:"
    echo "  • VS Code attach: Remote Explorer → Dev Containers → $CONTAINER_NAME"
    echo "  • View logs: docker logs -f $CONTAINER_NAME"
    echo "  • Stop container: $SCRIPT_DIR/stop_daemon.sh"
    echo ""
    echo "The container will automatically restart unless manually stopped."
else
    echo "✗ Failed to start container"
    exit 1
fi
