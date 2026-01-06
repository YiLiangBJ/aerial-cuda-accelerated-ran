#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

# USAGE:
#   PYAERIAL_IMAGE=<image> $cuBB_SDK/pyaerial/container/start_daemon.sh [cmd...]
#
# Start PyAerial container in daemon mode (background) for VS Code attach.
# The container restarts automatically unless manually stopped.
#
# Compatibility note vs run.sh:
# - run.sh starts an interactive container (or runs a command and exits).
# - start_daemon.sh keeps a single container running; if [cmd...] is provided,
#   it runs inside the daemon container via `docker exec`.

set -euo pipefail

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=${USER:-$(id -un)}

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
host_cuBB_SDK=$(builtin cd "$SCRIPT_DIR/../.."; pwd)

echo "$SCRIPT starting..."
source "$host_cuBB_SDK/cuPHY-CP/container/setup.sh"

EXEC_CMD=""
if [ $# -gt 0 ]; then
    EXEC_CMD="$*"
fi

AERIAL_PLATFORM=${AERIAL_PLATFORM:-amd64}
TARGETARCH=$(basename "$AERIAL_PLATFORM")

if [[ -z ${PYAERIAL_IMAGE:-} ]]; then
    PYAERIAL_IMAGE="pyaerial:${USER_NAME}-${AERIAL_VERSION_TAG}-${TARGETARCH}"
fi

CONTAINER_NAME="pyaerial_${USER_NAME}"

# Match run.sh behavior:
# - check host ~/.bashrc for the marker
# - if missing, append PS1 export into container ~/.bashrc before daemon cmd
DAEMON_CMD="tail -f /dev/null"
if [[ ! -f "$HOME/.bashrc" ]] || ! grep -qi 'export PS1="\[host:' "$HOME/.bashrc"; then
    DAEMON_CMD="echo 'export PS1=\"[host: $host_cuBB_SDK] \$PS1\"' >> ~/.bashrc && $DAEMON_CMD"
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' is already running"

        if [ -n "$EXEC_CMD" ]; then
            echo "Running command inside container: $EXEC_CMD"
            if [ -t 1 ]; then
                docker exec -it "$CONTAINER_NAME" /bin/bash -c "$EXEC_CMD"
            else
                docker exec -i "$CONTAINER_NAME" /bin/bash -c "$EXEC_CMD"
            fi
            exit $?
        fi

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
        docker rm "$CONTAINER_NAME"
    fi
fi

echo "Starting container '$CONTAINER_NAME' in background..."
echo "Image: $PYAERIAL_IMAGE"

docker run --privileged \
    -dt \
    --restart unless-stopped \
    ${AERIAL_EXTRA_FLAGS:-} \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --add-host "$CONTAINER_NAME":127.0.0.1 \
    --network host --shm-size=4096m \
    --device=/dev/gdrdrv:/dev/gdrdrv \
    -u "$USER_ID:$GROUP_ID" \
    -w /opt/nvidia/cuBB \
    -v "$host_cuBB_SDK":/opt/nvidia/cuBB \
    -v "$host_cuBB_SDK":/opt/nvidia/aerial_sdk \
    -v /dev/hugepages:/dev/hugepages \
    -v /lib/modules:/lib/modules \
    -v /var/log/aerial:/var/log/aerial \
    -e host_cuBB_SDK="$host_cuBB_SDK" \
    --userns=host --ipc=host \
    "$PYAERIAL_IMAGE" fixuid -q /bin/bash -c "$DAEMON_CMD"

echo ""
echo "Container started successfully."
echo "Next: attach via VS Code Dev Containers to '$CONTAINER_NAME'."

echo ""
echo "Tips:"
echo "  - Logs: docker logs -f $CONTAINER_NAME"
echo "  - Stop: $SCRIPT_DIR/stop_daemon.sh"

if [ -n "$EXEC_CMD" ]; then
    echo ""
    echo "Running command inside container: $EXEC_CMD"
    if [ -t 1 ]; then
        docker exec -it "$CONTAINER_NAME" /bin/bash -c "$EXEC_CMD"
    else
        docker exec -i "$CONTAINER_NAME" /bin/bash -c "$EXEC_CMD"
    fi
    exit $?
fi
