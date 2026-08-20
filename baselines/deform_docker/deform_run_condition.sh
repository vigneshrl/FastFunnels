#!/bin/bash
# deform_run_condition.sh
# =======================
# Run one DEFORM condition (n_agents) on the open_narrow_obs map.
# Metrics are written automatically to deform_docker/results/ when each episode
# ends.  The bridge node auto-shuts the container down after the configured
# number of episodes — so you can launch a 20-trial sweep with a single call.
#
# Usage:
#   ./deform_run_condition.sh --agents 1
#   ./deform_run_condition.sh --agents 2 --trials 20
#   ./deform_run_condition.sh --agents 4 --trials 20 --timeout 240
#
# Narrow-zone table (spawn at zone entry, metrics scoped to narrow passage):
#   ./deform_run_condition.sh --agents 1 --narrow --trials 20
#   ./deform_run_condition.sh --agents 2 --narrow --trials 20
#   ./deform_run_condition.sh --agents 4 --narrow --trials 20
#
# To run the full paper sweep (3 conditions × 20 trials):
#   ./deform_run_condition.sh --agents 1 --trials 20
#   ./deform_run_condition.sh --agents 2 --trials 20
#   ./deform_run_condition.sh --agents 4 --trials 20
#
# Then collect the table:
#   python ../deform_sweep.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="deform_ros:latest"
CONFIG_DIR="${SCRIPT_DIR}/f1tenth_deform_bridge/config"
LAUNCH_DIR="${SCRIPT_DIR}/f1tenth_deform_bridge/launch"

# ── Parse args ──────────────────────────────────────────────────────────────
N_AGENTS=4    # default
TRIALS=1      # default — one interactive run; set ≥20 for paper-table sweep
TIMEOUT=180   # per-episode timeout in seconds
NARROW=0      # 1 = spawn at narrow-zone entry, narrow-scoped metrics

while [[ $# -gt 0 ]]; do
    case $1 in
        --agents)  N_AGENTS="$2"; shift 2 ;;
        --trials)  TRIALS="$2";   shift 2 ;;
        --timeout) TIMEOUT="$2";  shift 2 ;;
        --narrow)  NARROW=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Select per-condition files ───────────────────────────────────────────────
if [[ "$NARROW" -eq 1 ]]; then
    case $N_AGENTS in
        1) BRIDGE_PARAMS="bridge_params_narrow_1agent.yaml"
           LAUNCH="deform_f1tenth_1agent_narrow.launch" ;;
        2) BRIDGE_PARAMS="bridge_params_narrow_2agents.yaml"
           LAUNCH="deform_f1tenth_2agents_narrow.launch" ;;
        4) BRIDGE_PARAMS="bridge_params_narrow_4agents.yaml"
           LAUNCH="deform_f1tenth_4agents_narrow.launch" ;;
        *) echo "Unsupported --agents value: $N_AGENTS (choose 1, 2, or 4)"; exit 1 ;;
    esac
    # Default shorter timeout for narrow-only segment (override with --timeout N)
    if [[ "$TIMEOUT" -eq 180 ]] && [[ "$TRIALS" -gt 1 ]]; then
        TIMEOUT=240
        [[ "$N_AGENTS" -eq 4 ]] && TIMEOUT=300
    fi
else
case $N_AGENTS in
    1)
        BRIDGE_PARAMS="bridge_params_full_narrow_1agent.yaml"
        LAUNCH="deform_f1tenth_1agent.launch"
        ;;
    2)
        BRIDGE_PARAMS="bridge_params_full_narrow_2agents.yaml"
        LAUNCH="deform_f1tenth_2agents.launch"
        ;;
    4)
        BRIDGE_PARAMS="bridge_params_full_narrow_4agents.yaml"
        LAUNCH="deform_f1tenth_4agents.launch"
        ;;
    *)
        echo "Unsupported --agents value: $N_AGENTS (choose 1, 2, or 4)"
        exit 1
        ;;
esac
fi

case $N_AGENTS in
    1) FRM_SHAPE="frm_shape_1robot.yaml" ;;
    2) FRM_SHAPE="frm_shape_2robots.yaml" ;;
    4) FRM_SHAPE="frm_shape_4robots.yaml" ;;
esac

mkdir -p "${SCRIPT_DIR}/results"
echo "────────────────────────────────────────────────"
MODE="full-track"
[[ "$NARROW" -eq 1 ]] && MODE="narrow-zone-entry"
echo "[deform_run] mode=${MODE}  n_agents=${N_AGENTS}  trials=${TRIALS}  timeout=${TIMEOUT}s  map=open_narrow_obs"
echo "[deform_run] bridge params : ${BRIDGE_PARAMS}"
echo "[deform_run] frm_shape     : ${FRM_SHAPE}"
echo "[deform_run] launch file   : ${LAUNCH}"
echo "[deform_run] results dir   : ${SCRIPT_DIR}/results"
echo "────────────────────────────────────────────────"

xhost +local:docker 2>/dev/null || true

# One-time: patch TurtleBot3 NMPC caps (0.22 m/s) → F1Tenth limits, commit image.
PATCHED_IMAGE="${IMAGE%%:*}:f1tenth_patched"
if ! docker image inspect "$PATCHED_IMAGE" &>/dev/null; then
    echo "[deform_run] Building ${PATCHED_IMAGE} (NMPC speed patch + catkin_make)…"
    CID=$(docker run -d \
        --volume "${SCRIPT_DIR}/patch_deform_f1tenth.sh":/patch_deform_f1tenth.sh:ro \
        --volume /home/mrvik/DEFORM/src/control/traj_opti/include/traj_opti/traj_opti.h:/root/DEFORM/src/control/traj_opti/include/traj_opti/traj_opti.h:ro \
        --volume /home/mrvik/DEFORM/src/plan_manager/src/plan_manager.cpp:/root/DEFORM/src/plan_manager/src/plan_manager.cpp:ro \
        --volume /home/mrvik/DEFORM/src/planning/path_searching2d/src/dyn_a_star.cpp:/root/DEFORM/src/planning/path_searching2d/src/dyn_a_star.cpp:ro \
        "$IMAGE" bash /patch_deform_f1tenth.sh)
    docker logs -f "$CID" 2>&1 | tail -20
    docker commit "$CID" "$PATCHED_IMAGE" >/dev/null
    docker rm -f "$CID" >/dev/null
    echo "[deform_run] Created ${PATCHED_IMAGE}"
fi
IMAGE="$PATCHED_IMAGE"

# Run TRIALS docker sessions sequentially.  Each container runs exactly one
# episode and exits cleanly after the bridge node auto-shuts down (driven by
# target_episodes=1).  Keeping it 1-episode-per-container guarantees DEFORM's
# plan_manager starts from a clean state every trial.
for trial in $(seq 1 "${TRIALS}"); do
    echo ""
    echo "════════════════════════════════════════════════"
    echo "[deform_run] Trial ${trial}/${TRIALS}  (n_agents=${N_AGENTS})"
    echo "════════════════════════════════════════════════"

    # Use a unique container name per trial so leftover state from a previous
    # killed container doesn't collide with --name.
    CONTAINER="deform_container_${N_AGENTS}_${trial}_$$"

    docker run --rm \
        --env DISPLAY="${DISPLAY}" \
        --env QT_X11_NO_MITSHM=1 \
        --env LIBGL_ALWAYS_SOFTWARE=1 \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --network host \
        --device /dev/dri \
        --group-add video \
        --name "${CONTAINER}" \
        --volume /home/mrvik/f1tenth_gym/maps:/opt/f1tenth_gym/maps \
        --volume "${SCRIPT_DIR}/results":/tmp/deform_results:rw \
        --volume "${CONFIG_DIR}/${BRIDGE_PARAMS}":/root/DEFORM/src/utility/f1tenth_deform_bridge/config/bridge_params.yaml:ro \
        --volume "${CONFIG_DIR}/${FRM_SHAPE}":/root/DEFORM/src/plan_manager/config/frm_shape.yaml:ro \
        --volume "${LAUNCH_DIR}":/root/DEFORM/src/utility/f1tenth_deform_bridge/launch:ro \
        --volume "${CONFIG_DIR}/deform_rviz.rviz":/root/DEFORM/src/utility/f1tenth_deform_bridge/config/deform_rviz.rviz:ro \
        --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/scripts/f1tenth_bridge_node.py":/root/DEFORM/devel/lib/f1tenth_deform_bridge/f1tenth_bridge_node.py \
        --volume "${SCRIPT_DIR}/f1tenth_deform_bridge/scripts/eval_metrics.py":/root/DEFORM/devel/lib/f1tenth_deform_bridge/eval_metrics.py \
        "$IMAGE" \
        bash -c "
            source /opt/ros/noetic/setup.bash && \
            source /root/DEFORM/devel/setup.bash && \
            roslaunch f1tenth_deform_bridge ${LAUNCH} \
                target_episodes:=1 \
                episode_timeout_s:=${TIMEOUT}
        " \
    || echo "[deform_run] Trial ${trial} container exited non-zero (continuing)"
done

echo ""
echo "[deform_run] Finished ${TRIALS} trial(s) for n_agents=${N_AGENTS}"
echo "[deform_run] Results in ${SCRIPT_DIR}/results/"
