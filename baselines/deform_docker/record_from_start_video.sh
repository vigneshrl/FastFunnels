#!/bin/bash
# Record from-track-start DEFORM demo to MP4 via rgb_array (no X11/ffmpeg).
# Usage: ./record_from_start_video.sh [output_basename]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASENAME="${1:-deform_4agents_from_start}"
OUT="${SCRIPT_DIR}/results/videos/${BASENAME}.mp4"
mkdir -p "${SCRIPT_DIR}/results/videos"

echo "[record] → ${OUT}"
docker run --rm \
  -e SDL_VIDEODRIVER=dummy \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  --network host \
  --volume /home/mrvik/f1tenth_gym/maps:/opt/f1tenth_gym/maps \
  --volume "${SCRIPT_DIR}/results":/tmp/deform_results:rw \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/config/bridge_params_full_narrow_4agents.yaml":/root/DEFORM/src/utility/f1tenth_deform_bridge/config/bridge_params.yaml:ro \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/config/frm_shape_4robots.yaml":/root/DEFORM/src/plan_manager/config/frm_shape.yaml:ro \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/launch":/root/DEFORM/src/utility/f1tenth_deform_bridge/launch:ro \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/scripts/f1tenth_bridge_node.py":/root/DEFORM/devel/lib/f1tenth_deform_bridge/f1tenth_bridge_node.py \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/scripts/eval_metrics.py":/root/DEFORM/devel/lib/f1tenth_deform_bridge/eval_metrics.py \
  deform_ros:f1tenth_patched \
  bash -lc "source /opt/ros/noetic/setup.bash && source /root/DEFORM/devel/setup.bash && \
    roslaunch f1tenth_deform_bridge deform_f1tenth_4agents_from_start_render.launch \
      target_episodes:=1 episode_timeout_s:=300 ignore_gym_done:=true \
      record_video_path:=/tmp/deform_results/videos/${BASENAME}.mp4"

ls -lh "${OUT}"
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1:nokey=0 "${OUT}" 2>/dev/null || true
