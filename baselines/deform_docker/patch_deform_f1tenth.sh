#!/bin/bash
# Rebuild DEFORM workspace with host-patched sources (mounted by deform_run_condition.sh).
set -e
source /opt/ros/noetic/setup.bash
cd /root/DEFORM
echo "[patch] MAX_LINEAR_VELOCITY=$(grep MAX_LINEAR_VELOCITY src/control/traj_opti/include/traj_opti/traj_opti.h)"
echo "[patch] empty-path guard=$(grep -c 'cur_path.empty()' src/plan_manager/src/plan_manager.cpp || echo 0)"
catkin_make -DCMAKE_BUILD_TYPE=Release
echo "[patch] catkin build done"
