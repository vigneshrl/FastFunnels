#!/bin/bash
# Record one DEFORM demo to MP4 via ffmpeg x11grab.
# Usage: ./record_deform_video.sh <output_basename> <launch_file> [episode_timeout_s]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASENAME="${1:?basename required}"
LAUNCH="${2:?launch file required}"
TIMEOUT="${3:-180}"
IMAGE="deform_ros:f1tenth_patched"

mkdir -p "${SCRIPT_DIR}/results/videos"
OUT="${SCRIPT_DIR}/results/videos/${BASENAME}.mp4"
LOG="${SCRIPT_DIR}/results/videos/${BASENAME}.log"

SIZE=$(xdpyinfo -display "${DISPLAY:-:0}" 2>/dev/null | awk '/dimensions:/{print $2; exit}')
[ -z "$SIZE" ] && SIZE=1920x1080

echo "[record] launch=${LAUNCH} timeout=${TIMEOUT}s output=${OUT}"
ffmpeg -y -video_size "$SIZE" -framerate 20 -f x11grab -i "${DISPLAY:-:0}" \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$OUT" \
  >/tmp/ffmpeg_"${BASENAME}".log 2>&1 &
FFPID=$!
sleep 2

xhost +local:docker >/dev/null 2>&1 || true
docker run --rm \
  --env DISPLAY="${DISPLAY:-:0}" \
  --env QT_X11_NO_MITSHM=1 \
  --env LIBGL_ALWAYS_SOFTWARE=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --network host \
  --device /dev/dri \
  --group-add video \
  --name "deform_vid_${BASENAME}_$$" \
  --volume /home/mrvik/f1tenth_gym/maps:/opt/f1tenth_gym/maps \
  --volume "${SCRIPT_DIR}/results":/tmp/deform_results:rw \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/config/bridge_params_narrow_4agents.yaml":/root/DEFORM/src/utility/f1tenth_deform_bridge/config/bridge_params.yaml:ro \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/config/frm_shape_4robots.yaml":/root/DEFORM/src/plan_manager/config/frm_shape.yaml:ro \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/launch":/root/DEFORM/src/utility/f1tenth_deform_bridge/launch:ro \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/config/deform_rviz.rviz":/root/DEFORM/src/utility/f1tenth_deform_bridge/config/deform_rviz.rviz:ro \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/scripts/f1tenth_bridge_node.py":/root/DEFORM/devel/lib/f1tenth_deform_bridge/f1tenth_bridge_node.py \
  --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/scripts/eval_metrics.py":/root/DEFORM/devel/lib/f1tenth_deform_bridge/eval_metrics.py \
  "$IMAGE" \
  bash -lc "source /opt/ros/noetic/setup.bash && source /root/DEFORM/devel/setup.bash && roslaunch f1tenth_deform_bridge ${LAUNCH} target_episodes:=1 episode_timeout_s:=${TIMEOUT}" \
  >"$LOG" 2>&1 || true

kill -INT "$FFPID" >/dev/null 2>&1 || true
sleep 2
wait "$FFPID" >/dev/null 2>&1 || true

ls -lh "$OUT"
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1:nokey=0 "$OUT"
echo "[record] done: $OUT"
