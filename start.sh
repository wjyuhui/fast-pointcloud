#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# do not use set -u: Humble setup.bash reads unbound vars
source /opt/ros/humble/setup.bash
source "${ROOT}/setup_local_deps.bash"
source "${ROOT}/install/setup.bash"

if [[ "${1:-}" == "bag" ]]; then
  exec ros2 launch rgbd_bringup perception.launch.py use_orbbec:=false
fi

exec ros2 launch rgbd_bringup perception.launch.py
