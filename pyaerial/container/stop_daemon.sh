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

# USAGE: $cuBB_SDK/pyaerial/container/stop_daemon.sh
#
# Stop and remove the PyAerial daemon container.

USER_ID=$(id -u)
CONTAINER_NAME=pyaerial_$USER

echo "=== PyAerial Daemon Stopper ==="

if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "✓ Container '$CONTAINER_NAME' does not exist (already removed)"
    exit 0
fi

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping container '$CONTAINER_NAME'..."
    docker stop $CONTAINER_NAME
    if [ $? -ne 0 ]; then
        echo "✗ Failed to stop container"
        exit 1
    fi
fi

echo "Removing container '$CONTAINER_NAME'..."
docker rm $CONTAINER_NAME

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Container stopped and removed successfully"
    echo ""
    echo "To restart: $(dirname $0)/start_daemon.sh"
else
    echo "✗ Failed to remove container"
    exit 1
fi
