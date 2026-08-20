#!/bin/bash
# run.sh — Build and launch the DEFORM + f1tenth container
#
# Usage:
#   ./run.sh              # interactive shell
#   ./run.sh roslaunch f1tenth_deform_bridge deform_f1tenth_4agents.launch

set -e

IMAGE="deform_ros:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Allow the local Docker daemon to connect to the host X server
xhost +local:docker 2>/dev/null || true

docker run -it --rm \
    --env DISPLAY="$DISPLAY" \
    --env QT_X11_NO_MITSHM=1 \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network host \
    --device /dev/dri \
    --group-add video \
    --name deform_container \
    --volume /home/mrvik/f1tenth_gym/maps:/opt/f1tenth_gym/maps \
    --volume "${SCRIPT_DIR}/results":/tmp/deform_results:rw \
    --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/config/frm_shape_4robots.yaml":/root/DEFORM/src/plan_manager/config/frm_shape.yaml:ro \
    --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/config/bridge_params.yaml":/root/DEFORM/src/utility/f1tenth_deform_bridge/config/bridge_params.yaml:ro \
    "$IMAGE" \
    "$@"
