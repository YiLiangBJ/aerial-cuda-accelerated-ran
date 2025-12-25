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

"""PyAerial release Docker image hpccm recipe using Aerial release image as the base.

Usage:
$ hpccm
    --recipe pyaerial_recipe.py
    --format docker
    --userarg AERIAL_BASE_IMAGE
    --userarg TENSORFLOW_IMAGE
"""
import os

AERIAL_BASE_IMAGE = USERARG.get("AERIAL_BASE_IMAGE")
if AERIAL_BASE_IMAGE is None:
    raise RuntimeError("User argument AERIAL_BASE_IMAGE must be set")


if cpu_target == 'x86_64':
    TARGETARCH='amd64'
    PyAerial = stages[0]
elif cpu_target == 'aarch64':
    TARGETARCH='arm64'
    TensorFlow = stages[0]
    PyAerial = stages[1]
else:
    raise RuntimeError("Unsupported platform")


if cpu_target == 'aarch64':
    TENSORFLOW_IMAGE = USERARG.get("TENSORFLOW_IMAGE")
    if TENSORFLOW_IMAGE is None:
        raise RuntimeError("User argument TENSORFLOW_IMAGE must be set")

    TensorFlow += baseimage(image=TENSORFLOW_IMAGE, _distro='ubuntu22', _arch=cpu_target, _as="tensorflow")


PyAerial += baseimage(image=AERIAL_BASE_IMAGE, _distro='ubuntu22', _arch=cpu_target)

PyAerial += user(user='root')

PyAerial += packages(ospackages=[
    'cudnn9-cuda-12',
    ])

if cpu_target == 'aarch64':

    PyAerial += copy(src='/usr/local/bin/deviceQuery', dest='/usr/local/bin/', _from="tensorflow")
    PyAerial += copy(src='/usr/local/bin/deviceQueryDrv', dest='/usr/local/bin/', _from="tensorflow")
    files = ['21-tensorflow-copyright.txt',
            '51-gpu-sm-version-check.sh',
            '52-gpu-driver-version-check.sh',
            '54-cpu-capabilities-check.sh',
            '56-network-driver-version-check.sh',
            '70-shm-check.sh',
            ]
    for f in files:
        PyAerial += copy(src=f'/opt/nvidia/entrypoint.d/{f}', dest='/opt/nvidia/entrypoint.d/', _from="tensorflow")


PyAerial += environment(variables={
    "VIRTUAL_ENV": "/opt/venv",
    "PATH": "/opt/venv/bin:$PATH",
    })

PyAerial += shell(commands=[
    'mkdir -p $VIRTUAL_ENV',
    'chown aerial:aerial $VIRTUAL_ENV',
    ])

PyAerial += user(user='aerial')

PyAerial += shell(commands=[
    'python3 -m venv $VIRTUAL_ENV --system-site-packages',
    ])

PyAerial += copy(src='.', dest='$cuBB_SDK', _chown="aerial")

# install PyAerial.
PyAerial += shell(commands=[
    '$cuBB_SDK/pyaerial/scripts/install_dev_pkg.sh'
    ])

PyAerial += shell(commands=[
    'pip install pip --upgrade',
    f'pip install -r $cuBB_SDK/pyaerial/container/requirements.txt -r $cuBB_SDK/pyaerial/container/requirements-{TARGETARCH}.txt',
])

if cpu_target == 'aarch64':
    PyAerial += copy(src='/tmp/pip', dest='/tmp/pip', _chown="aerial", _from="tensorflow")
    PyAerial += shell(commands=[
        'pip install --no-cache-dir /tmp/pip/tensorflow-*.whl',
        'pip check',
        'rm -rf /tmp/pip',
        ])

if cpu_target == 'x86_64':
    PyAerial += shell(commands=[
        'pip install torch==2.7.1',
        ])
elif cpu_target == 'aarch64':
    PyAerial += workdir(directory='/tmp')
    PyAerial += shell(commands=[
        'wget -q https://download.pytorch.org/whl/triton-3.3.1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl',
        'wget -q https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=aca3472608e3c92df5166537595687b53a6c997082478b372427b043dbed98d0',
        'pip install triton*.whl',
        'pip install torch*.whl',
        'rm *.whl',
        ])

PyAerial += shell(commands=[
    'rm -rf /home/aerial/.cache',
    ])
# read host uid/gid from build environment (fallback to build user's uid/gid)
_host_uid = int(os.environ.get('HOST_UID', os.getuid()))
_host_gid = int(os.environ.get('HOST_GID', os.getgid()))

# ensure virtualenv ownership matches host UID/GID after all installs (run as root)
PyAerial += user(user='root')
PyAerial += shell(commands=[
    f'chown -R {_host_uid}:{_host_gid} /opt/venv || true',
    ])
PyAerial += user(user='aerial')
PyAerial += workdir(directory='/home/aerial')
PyAerial += raw(docker='CMD ["/bin/bash"]')
