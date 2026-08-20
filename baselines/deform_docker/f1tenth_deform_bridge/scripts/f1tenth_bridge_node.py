#!/usr/bin/env python3
"""
f1tenth_bridge_node.py
======================
ROS bridge that wraps the f1tenth_gym Gymnasium environment and exposes
each of the 4 agents as standard ROS topics that DEFORM's plan_manager
can subscribe to / command.

Topic mapping (per agent i = 0..3):
  Published:
    /tb_{i}/odom                          nav_msgs/Odometry     — agent pose & velocity
    /tb_{i}/camera/depth/image_filter     sensor_msgs/Image     — synthetic depth (32FC1)
    /tb_{i}/camera/depth/camera_info      sensor_msgs/CameraInfo
    /map                                  nav_msgs/OccupancyGrid — static map (latched)

  Subscribed:
    /tb_{i}/cmd_vel       geometry_msgs/Twist    — velocity command from DEFORM

  Static TF:
    tb_{i}/base_link → tb_{i}/camera_depth_optical_frame

Depth images are synthesised by raycasting into the known occupancy map plus
treating every other robot as a cylindrical obstacle (radius ROBOT_R).

Coordinate conversion (Ackermann ↔ differential-drive):
    speed    = twist.linear.x
    steering = atan(twist.angular.z * WHEELBASE / speed)   [bicycle model]
"""

import sys
import os
import math
import threading
import yaml
import pygame
import csv
import datetime

import rospy
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from sensor_msgs.msg import Image as RosImage, CameraInfo
from std_msgs.msg import Header
import tf2_ros
import tf.transformations as tft
from PIL import Image
from visualization_msgs.msg import Marker, MarkerArray

# ── F1Tenth vehicle constants ───────────────────────────────────────────────
LF        = 0.15875   # distance CoG → front axle (m)
LR        = 0.17145   # distance CoG → rear axle  (m)
WHEELBASE = LF + LR   # = 0.3302 m

STEER_MIN = -0.4189   # rad  (≈ -24°)
STEER_MAX =  0.4189   # rad  (≈ +24°)
SPEED_MIN = -5.0      # m/s
SPEED_MAX =  20.0     # m/s
# Below this |v|, bicycle steer = atan(omega*L/v) is ill-conditioned (Ackermann).
STEER_SPEED_THRESHOLD = 0.01
# Minimum forward speed when DEFORM commands turn with ~zero v (diff-drive habit).
STEER_NUDGE_SPEED     = 0.20

DEPTH_W   = 320               # image columns
DEPTH_H   = 240               # image rows (tiled from single horizontal ray-cast)
DEPTH_FOV = math.radians(87)  # horizontal FOV (≈ RealSense D435)
DEPTH_MAX = 10.0              # m — maximum depth range
ROBOT_R   = 0.20              # m — robot body radius for inter-agent occlusion


class F1TenthBridge:
    def __init__(self):
        # Must match <node name="f1tenth_bridge"> in launch files so ~params resolve.
        rospy.init_node("f1tenth_bridge")

        # ── Parameters ────────────────────────────────────────────────────
        self.num_agents  = rospy.get_param("~num_agents",  4)
        self.map_name    = rospy.get_param("~map_name",    "open_narrow_obs")
        self.gym_root    = rospy.get_param("~gym_root",    "/opt/f1tenth_gym")
        self.rate_hz     = rospy.get_param("~rate_hz",     50)
        self.cmd_timeout = rospy.get_param("~cmd_timeout", 0.5)  # s
        self.render       = rospy.get_param("~render", True)
        self._record_video_path = str(rospy.get_param("~record_video_path", ""))
        self._record_fps        = int(rospy.get_param("~record_fps", 20))
        self._record_skip       = max(1, int(rospy.get_param("~record_frame_skip", 2)))
        self._video_writer      = None
        self._record_frame_idx  = 0
        if self._record_video_path:
            self.render = True
            os.makedirs(os.path.dirname(self._record_video_path) or ".", exist_ok=True)
            rospy.on_shutdown(self._close_video)

        # Initial poses [x, y, theta] — staggered approaching toll plaza
        # custom_map3 toll plaza barriers are at wp181-185 (~-39, -11.5)
        # spawn at wp173-179 just before the barriers; track heading ~-0.36 rad
        default_poses = [
            [25.585, 11.624, 2.79],   # WP55 +0.3m right
            [25.379, 11.060, 2.79],   # WP55 -0.3m left
            [28.277, 10.637, 2.79],   # WP52 +0.3m right
            [28.071, 10.073, 2.79],   # WP52 -0.3m left
        ]
        self.init_poses = rospy.get_param("~initial_poses", default_poses)

        # ── Add f1tenth_gym to Python path & import ───────────────────────
        sys.path.insert(0, self.gym_root)
        # eval_metrics.py lives alongside this script
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import gymnasium as gymnasium_mod
            import f1tenth_gym  # triggers gym registration
            self._gym = gymnasium_mod
        except ImportError as e:
            rospy.logfatal("Cannot import f1tenth_gym from %s: %s", self.gym_root, e)
            raise
            # 1.2583960096163425, 20.228066212248873
        # ── Goal publishing (auto thread or manual RViz click) ────────────
        self._auto_goal    = bool(rospy.get_param("~auto_publish_goal", False))
        self._goal_x       = rospy.get_param("~goal_x",       -10.5)
        self._goal_y       = rospy.get_param("~goal_y",         27.0)
        self._goal_yaw     = rospy.get_param("~goal_yaw",        1.53)
        self._goal_delay_s = rospy.get_param("~goal_delay_s",    5.0)
        self._goal_republish_period_s = float(
            rospy.get_param("~goal_republish_period_s", 2.0)
        )
        self._goal_tb_num  = rospy.get_param("~tb_num",           self.num_agents)

        # ── Full-run metrics (loaded once; reset per episode) ───────────
        self._zone      = None   # FullRunMetrics instance, reset each episode
        self._zone_cls  = None
        self._cl_xs     = None
        self._cl_ys     = None
        self._episode_n = 0
        self._results_dir = rospy.get_param("~results_dir", "/tmp/deform_results")
        os.makedirs(self._results_dir, exist_ok=True)
        self._load_centerline()

        # ── Multi-trial control (for automated sweeps) ─────────────────────
        # `target_episodes` = how many episodes to run before auto-shutting
        # down the ROS node (the launch file has required=true on this node,
        # so this also exits the docker container cleanly).  Default is 1
        # so the existing interactive workflow is unchanged.
        self._target_episodes  = int(rospy.get_param("~target_episodes",  1))
        self._episode_timeout  = float(rospy.get_param("~episode_timeout_s", 180.0))
        # Keep sim running after gym collision/lap (useful for long demo videos).
        self._ignore_gym_done  = bool(rospy.get_param("~ignore_gym_done", False))
        # map_goal: end when all agents reach the map goal waypoint
        # narrow_clear: end when all agents exit the narrow zone (spawn-at-entry runs)
        self._episode_end_mode = str(
            rospy.get_param("~episode_end_mode", "map_goal")
        )
        self._episode_start_ts = None   # set when each episode begins
        self._step_count = 0

        # ── State ─────────────────────────────────────────────────────────
        self.cmd_vels       = [Twist()] * self.num_agents
        self.cmd_stamps     = [rospy.Time(0)] * self.num_agents
        self.action         = np.zeros((self.num_agents, 2), dtype=np.float64)

        # ── ROS infrastructure ────────────────────────────────────────────
        self.tf_br         = tf2_ros.TransformBroadcaster()
        self._static_tf_br = tf2_ros.StaticTransformBroadcaster()
        self.odom_pubs     = []
        self.depth_pubs    = []
        self.cinfo_pubs    = []

        for i in range(self.num_agents):
            self.odom_pubs.append(
                rospy.Publisher(f"/tb_{i}/odom", Odometry, queue_size=10)
            )
            self.depth_pubs.append(
                rospy.Publisher(
                    f"/tb_{i}/camera/depth/image_filter", RosImage, queue_size=10
                )
            )
            self.cinfo_pubs.append(
                rospy.Publisher(
                    f"/tb_{i}/camera/depth/camera_info", CameraInfo, queue_size=10
                )
            )
            rospy.Subscriber(
                f"/tb_{i}/cmd_vel", Twist,
                self._make_cmd_cb(i)
            )

        self.map_pub = rospy.Publisher(
            "/map", OccupancyGrid, queue_size=1, latch=True
        )
        self.obstacle_pub = rospy.Publisher(
            "/obstacle_features", MarkerArray, queue_size=1, latch=True
        )
        self.goal_pub = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=1, latch=True
        )

        # ── Init gym ──────────────────────────────────────────────────────
        self._init_gym()

        # ── Publish static map + obstacle markers ─────────────────────────
        self._publish_map()
        self._load_map_occ()
        self._publish_camera_tfs()
        self._load_and_publish_obstacles()

        rospy.loginfo(
            "[Bridge] Ready — %d agents, map='%s', rate=%d Hz, "
            "auto_goal=%s, episode_end=%s",
            self.num_agents, self.map_name, self.rate_hz,
            self._auto_goal, self._episode_end_mode,
        )

        # ── Optionally auto-publish goal; otherwise user clicks in RViz ──
        if self._auto_goal:
            self._start_goal_thread()
        else:
            rospy.loginfo("[Bridge] auto_publish_goal=false — use RViz 2D Nav Goal")

    # ──────────────────────────────────────────────────────────────────────
    # Goal publishing (background thread — no separate ROS node needed)
    # ──────────────────────────────────────────────────────────────────────
    def _start_goal_thread(self):
        def _worker():
            rospy.loginfo("[Bridge] GoalThread: waiting for %d plan_manager(s)…",
                          self._goal_tb_num)
            for i in range(self._goal_tb_num):
                topic = f"/tb_{i}/cmd_vel"
                while not rospy.is_shutdown():
                    if any(topic in t for t, _ in rospy.get_published_topics()):
                        break
                    rospy.sleep(0.5)
                rospy.loginfo("[Bridge] GoalThread: %s ready", topic)

            rospy.loginfo("[Bridge] GoalThread: all planners up — publishing in %.1fs",
                          self._goal_delay_s)
            rospy.sleep(self._goal_delay_s)

            if rospy.is_shutdown():
                return

            msg = PoseStamped()
            msg.header.stamp    = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.pose.position.x = self._goal_x
            msg.pose.position.y = self._goal_y
            msg.pose.position.z = 0.0
            q = tft.quaternion_from_euler(0.0, 0.0, self._goal_yaw)
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]

            period = max(0.5, self._goal_republish_period_s)
            rospy.loginfo(
                "[Bridge] GoalThread: republishing goal every %.1fs → (%.2f, %.2f)",
                period, self._goal_x, self._goal_y,
            )
            while not rospy.is_shutdown():
                msg.header.stamp = rospy.Time.now()
                self.goal_pub.publish(msg)
                rospy.sleep(period)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ──────────────────────────────────────────────────────────────────────
    # Narrow-zone metrics helpers
    # ──────────────────────────────────────────────────────────────────────
    def _load_centerline(self):
        """Load centerline CSV + narrow region for the current map and import FullRunMetrics."""
        csv_path = os.path.join(
            self.gym_root, "maps", self.map_name,
            f"{self.map_name}_centerline.csv"
        )
        try:
            from eval_metrics import FullRunMetrics
            self._zone_cls = FullRunMetrics
            wps = np.loadtxt(csv_path, delimiter=",", comments="#")
            self._cl_xs = wps[:, 0].astype(float)
            self._cl_ys = wps[:, 1].astype(float)
            rospy.loginfo("[Bridge] Centerline loaded: %s (%d wp)", csv_path, len(self._cl_xs))
        except FileNotFoundError:
            rospy.logwarn("[Bridge] No centerline CSV at %s — full-run metrics disabled", csv_path)
        except ImportError:
            rospy.logwarn("[Bridge] eval_metrics not found — full-run metrics disabled")

        # ── Narrow region from obs_pos.yaml (for deformability) ──────────
        self._narrow_xy = None
        self._gap_w     = None
        obs_yaml_path = os.path.join(
            self.gym_root, "maps", self.map_name,
            f"{self.map_name}_obs_pos.yaml"
        )
        try:
            with open(obs_yaml_path) as f:
                obs_data = yaml.safe_load(f)
            nr = (obs_data.get("narrow_regions") or [None])[0]
            if nr is not None:
                self._narrow_xy = tuple(nr["center_m"])
                self._gap_w     = float(nr["gap_width_m"])
                rospy.loginfo(
                    "[Bridge] Narrow region: center=%s, gap=%.3f m",
                    self._narrow_xy, self._gap_w
                )
        except FileNotFoundError:
            rospy.logwarn("[Bridge] No obs_pos.yaml at %s — deformability disabled", obs_yaml_path)

    def _reset_zone(self):
        """Create a fresh FullRunMetrics collector for a new episode."""
        if self._zone_cls is not None and self._cl_xs is not None:
            self._zone = self._zone_cls(
                centerline_xs=self._cl_xs,
                centerline_ys=self._cl_ys,
                dt=1.0 / float(self.rate_hz),
                narrow_center_xy=self._narrow_xy,
                gap_width_m=self._gap_w,
                goal_xy=(self._goal_x, self._goal_y),
            )
        self._episode_start_ts = rospy.Time.now()
        self._step_count = 0

    def _flush_zone(self):
        """Print + save full-run metrics when an episode ends, then reset."""
        if self._zone is None:
            return
        zm = self._zone.summary()
        self._episode_n += 1
        rospy.loginfo(
            "[Bridge] ── Episode %d narrow-zone metrics ──\n"
            "  avg_speed_mps   : %.3f\n"
            "  success         : %.1f\n"
            "  time_to_goal_s  : %.3f\n"
            "  flow_rate       : %.4f agents/(m·s)\n"
            "  deformability   : %s\n"
            "  n_cleared       : %s / %s",
            self._episode_n,
            zm["avg_speed_mps"], zm["success"],
            zm["time_to_goal_s"], zm["flow_rate"],
            zm["deformability"],
            zm.get("n_completed", "?"), zm.get("n_agents", "?"),
        )
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self._results_dir, f"zone_ep{self._episode_n:03d}_{ts}.csv")
        row = {"n_agents": self.num_agents, "map_name": self.map_name, **zm}
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        rospy.loginfo("[Bridge] Metrics saved → %s", out)
        self._zone = None

    # ──────────────────────────────────────────────────────────────────────
    # Gym init
    # ──────────────────────────────────────────────────────────────────────
    def _init_gym(self):
        if self._record_video_path:
            render_mode = "rgb_array"
        elif self.render:
            render_mode = "human"
        else:
            render_mode = None

        self.env = self._gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "num_agents":         self.num_agents,
                "map":                self.map_name,
                "timestep":           0.01,
                "integrator":         "rk4",
                "model":              "st",
                "control_input":      ["speed", "steering_angle"],
                "observation_config": {"type": "original"},
            },
            render_mode=render_mode,
        )

        init_np = np.array(self.init_poses[:self.num_agents], dtype=np.float64)
        self.obs, self.info = self.env.reset(
            options={"poses": init_np}
        )
        self._warn_spawn_collisions()
        self._record_frame()

    def _warn_spawn_collisions(self):
        """Log if any agent starts in collision (common cause of invisible / instant-done runs)."""
        col = getattr(self.env.unwrapped, "collisions", None)
        if col is not None and np.any(col):
            rospy.logerr(
                "[Bridge] Spawn collision for agent(s) %s — fix initial_poses "
                "(map cells can look free while the car footprint overlaps walls).",
                np.where(col)[0].tolist(),
            )

    def _episode_end_reason(self, done, episode_complete, timed_out):
        if episode_complete:
            return (
                "narrow zone cleared"
                if self._episode_end_mode == "narrow_clear"
                else "map goal reached"
            )
        if timed_out:
            return f"timeout ({self._episode_timeout:.0f}s)"
        if done:
            col = getattr(self.env.unwrapped, "collisions", None)
            ck = (self.info or {}).get("checkpoint_done")
            if col is not None and np.any(col):
                return "collision (gym)"
            if ck is not None and np.all(ck):
                return "lap complete (all checkpoints)"
            return "gym signalled done"
        return "unknown"

    def _record_frame(self):
        """Write rgb_array frame to MP4 (works headless; avoids black X11 capture)."""
        if not self._record_video_path:
            return
        self._record_frame_idx += 1
        if (self._record_frame_idx - 1) % self._record_skip != 0:
            return
        try:
            frame = self.env.render()
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "[Bridge] Video frame render failed: %s", exc)
            return
        if frame is None:
            return
        try:
            import cv2
        except ImportError:
            rospy.logerr("[Bridge] cv2 required for record_video_path")
            return
        if self._video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                self._record_video_path, fourcc, self._record_fps, (w, h)
            )
            rospy.loginfo(
                "[Bridge] Recording video → %s (%dx%d @ %d fps)",
                self._record_video_path, w, h, self._record_fps,
            )
        self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _close_video(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            rospy.loginfo("[Bridge] Video saved → %s", self._record_video_path)

    # ──────────────────────────────────────────────────────────────────────
    # cmd_vel callback factory
    # ──────────────────────────────────────────────────────────────────────
    def _make_cmd_cb(self, idx):
        def cb(msg):
            self.cmd_vels[idx]   = msg
            self.cmd_stamps[idx] = rospy.Time.now()
        return cb

    # ──────────────────────────────────────────────────────────────────────
    # Twist → [steering, speed]   (bicycle model)
    # ──────────────────────────────────────────────────────────────────────
    def _twist_to_action(self, twist):
        speed = float(twist.linear.x)
        omega = float(twist.angular.z)

        # DEFORM (unicycle NMPC) may command omega with |v|≈0; F1Tenth needs v·tan(δ)/L.
        if abs(speed) < STEER_SPEED_THRESHOLD and abs(omega) > 0.02:
            speed = STEER_NUDGE_SPEED if speed >= 0.0 else -STEER_NUDGE_SPEED

        if abs(speed) > STEER_SPEED_THRESHOLD:
            steer = math.atan2(omega * WHEELBASE, speed)
        else:
            steer = 0.0

        steer = float(np.clip(steer, STEER_MIN, STEER_MAX))
        speed = float(np.clip(speed, SPEED_MIN, SPEED_MAX))
        return np.array([speed, steer], dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────
    # Publish nav_msgs/Odometry + TF for one agent
    # ──────────────────────────────────────────────────────────────────────
    def _publish_odom(self, idx, stamp):
        obs = self.obs
        x     = float(obs["poses_x"][idx])
        y     = float(obs["poses_y"][idx])
        theta = float(obs["poses_theta"][idx])
        vx    = float(obs["linear_vels_x"][idx])
        vy    = float(obs["linear_vels_y"][idx])
        wz    = float(obs["ang_vels_z"][idx])

        q = tft.quaternion_from_euler(0.0, 0.0, theta)

        # — Odometry message —
        odom                         = Odometry()
        odom.header.stamp            = stamp
        odom.header.frame_id         = "world"
        odom.child_frame_id          = f"tb_{idx}/base_link"
        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.position.z    = 0.0
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x    = vx
        odom.twist.twist.linear.y    = vy
        odom.twist.twist.angular.z   = wz
        self.odom_pubs[idx].publish(odom)

        # — TF: world → tb_i/odom  (identity) —
        t1                          = TransformStamped()
        t1.header.stamp             = stamp
        t1.header.frame_id          = "world"
        t1.child_frame_id           = f"tb_{idx}/odom"
        t1.transform.rotation.w     = 1.0
        self.tf_br.sendTransform(t1)

        # — TF: tb_i/odom → tb_i/base_link —
        t2                            = TransformStamped()
        t2.header.stamp               = stamp
        t2.header.frame_id            = f"tb_{idx}/odom"
        t2.child_frame_id             = f"tb_{idx}/base_link"
        t2.transform.translation.x    = x
        t2.transform.translation.y    = y
        t2.transform.translation.z    = 0.0
        t2.transform.rotation.x       = q[0]
        t2.transform.rotation.y       = q[1]
        t2.transform.rotation.z       = q[2]
        t2.transform.rotation.w       = q[3]
        self.tf_br.sendTransform(t2)

    # ──────────────────────────────────────────────────────────────────────
    # Load occupancy map for depth raycasting
    # ──────────────────────────────────────────────────────────────────────
    def _load_map_occ(self):
        map_dir   = os.path.join(self.gym_root, "maps", self.map_name)
        yaml_path = os.path.join(map_dir, f"{self.map_name}_map.yaml")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        pgm_path = os.path.join(map_dir, cfg["image"])
        img = np.array(Image.open(pgm_path))
        img = np.flipud(img)
        free_thresh = int(cfg.get("free_thresh", 0.196) * 255)
        self._map_occ      = img < free_thresh          # True = occupied
        self._map_res      = float(cfg["resolution"])
        self._map_origin_x = float(cfg["origin"][0])
        self._map_origin_y = float(cfg["origin"][1])
        rospy.loginfo("[Bridge] Occupancy map loaded for raycasting (%dx%d @ %.4fm)",
                      self._map_occ.shape[1], self._map_occ.shape[0], self._map_res)

    # ──────────────────────────────────────────────────────────────────────
    # Publish static TF: tb_i/base_link → tb_i/camera_depth_optical_frame
    # ──────────────────────────────────────────────────────────────────────
    def _publish_camera_tfs(self):
        tfs = []
        # optical frame convention: Z forward, X right, Y down
        # rotation from base_link (X forward) = R_z(-90°) * R_x(-90°)
        q = tft.quaternion_from_euler(-math.pi / 2, 0.0, -math.pi / 2)
        for i in range(self.num_agents):
            t = TransformStamped()
            t.header.stamp    = rospy.Time.now()
            t.header.frame_id = f"tb_{i}/base_link"
            t.child_frame_id  = f"tb_{i}/camera_depth_optical_frame"
            t.transform.translation.x = 0.10   # 10 cm forward
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.05   # 5 cm above base
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            tfs.append(t)
        self._static_tf_br.sendTransform(tfs)

    # ──────────────────────────────────────────────────────────────────────
    # Raycast into occupancy map → float32 depth row (walls only)
    # Other robots are intentionally excluded: DEFORM tracks them via
    # explicit /tb_i/odom subscriptions and CasADi inter-robot constraints.
    # Including them in the depth image triggers drone_detect's avoidance
    # mode which conflicts with the formation optimizer.
    # ──────────────────────────────────────────────────────────────────────
    def _raycast(self, x, y, theta):
        half   = DEPTH_FOV / 2.0
        angles = theta + np.linspace(-half, half, DEPTH_W)
        cos_a  = np.cos(angles)
        sin_a  = np.sin(angles)

        n_steps = int(DEPTH_MAX / self._map_res) + 1
        t_vals  = np.arange(n_steps, dtype=np.float32) * self._map_res

        # World coords along every ray: shapes (DEPTH_W, n_steps)
        xs = x + np.outer(cos_a, t_vals)
        ys = y + np.outer(sin_a, t_vals)

        map_xi = ((xs - self._map_origin_x) / self._map_res).astype(np.int32)
        map_yi = ((ys - self._map_origin_y) / self._map_res).astype(np.int32)

        H, W = self._map_occ.shape
        valid = (map_xi >= 0) & (map_xi < W) & (map_yi >= 0) & (map_yi < H)

        wall_hit = np.zeros((DEPTH_W, n_steps), dtype=bool)
        wall_hit[valid] = self._map_occ[map_yi[valid], map_xi[valid]]

        first_idx = np.argmax(wall_hit, axis=1)
        any_wall  = wall_hit.any(axis=1)
        return np.where(any_wall, t_vals[first_idx], DEPTH_MAX).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Publish synthetic depth image + camera_info for one agent
    # ──────────────────────────────────────────────────────────────────────
    def _publish_depth_image(self, idx, stamp):
        obs   = self.obs
        x     = float(obs["poses_x"][idx])
        y     = float(obs["poses_y"][idx])
        theta = float(obs["poses_theta"][idx])

        depths = self._raycast(x, y, theta)                  # (DEPTH_W,)
        img_np = np.tile(depths, (DEPTH_H, 1)).astype(np.float32)

        img_msg              = RosImage()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = f"tb_{idx}/camera_depth_optical_frame"
        img_msg.height       = DEPTH_H
        img_msg.width        = DEPTH_W
        img_msg.encoding     = "32FC1"
        img_msg.is_bigendian = 0
        img_msg.step         = DEPTH_W * 4
        img_msg.data         = img_np.tobytes()
        self.depth_pubs[idx].publish(img_msg)

        fx = DEPTH_W / (2.0 * math.tan(DEPTH_FOV / 2.0))
        cx = DEPTH_W / 2.0
        cy = DEPTH_H / 2.0

        cinfo                    = CameraInfo()
        cinfo.header             = img_msg.header
        cinfo.height             = DEPTH_H
        cinfo.width              = DEPTH_W
        cinfo.distortion_model   = "plumb_bob"
        cinfo.D                  = [0.0, 0.0, 0.0, 0.0, 0.0]
        cinfo.K                  = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
        cinfo.R                  = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        cinfo.P                  = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.cinfo_pubs[idx].publish(cinfo)

    # ──────────────────────────────────────────────────────────────────────
    # Convert PGM map → nav_msgs/OccupancyGrid and publish (latched)
    # ──────────────────────────────────────────────────────────────────────
    def _publish_map(self):
        map_dir  = os.path.join(self.gym_root, "maps", self.map_name)
        yaml_path = os.path.join(map_dir, f"{self.map_name}_map.yaml")

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        pgm_path = os.path.join(map_dir, cfg["image"])
        img = np.array(Image.open(pgm_path))     # shape (H, W), uint8
        img = np.flipud(img)                      # ROS origin = bottom-left

        occ_thresh  = int(cfg.get("occupied_thresh", 0.45) * 255)
        free_thresh = int(cfg.get("free_thresh",     0.196) * 255)

        # PGM: 0 = black = obstacle, 255 = white = free
        data = np.full(img.shape, -1, dtype=np.int8)
        data[img > occ_thresh]  = 0    # free
        data[img < free_thresh] = 100  # occupied

        grid                         = OccupancyGrid()
        grid.header.stamp            = rospy.Time.now()
        grid.header.frame_id         = "world"
        grid.info.resolution         = cfg["resolution"]
        grid.info.width              = img.shape[1]
        grid.info.height             = img.shape[0]
        grid.info.origin.position.x  = cfg["origin"][0]
        grid.info.origin.position.y  = cfg["origin"][1]
        grid.info.origin.orientation.w = 1.0
        grid.data                    = data.flatten().tolist()

        self.map_pub.publish(grid)
        rospy.loginfo(
            "[Bridge] Published map '%s' (%dx%d @ %.4f m/px)",
            self.map_name, grid.info.width, grid.info.height, grid.info.resolution
        )

    # ──────────────────────────────────────────────────────────────────────
    # Load obs_pos.yaml and publish obstacle markers (latched)
    # ──────────────────────────────────────────────────────────────────────
    def _load_and_publish_obstacles(self):
        map_dir   = os.path.join(self.gym_root, "maps", self.map_name)
        yaml_path = os.path.join(map_dir, f"{self.map_name}_obs_pos.yaml")

        if not os.path.exists(yaml_path):
            rospy.logwarn("[Bridge] No obs_pos.yaml found at %s — obstacle markers skipped", yaml_path)
            return

        with open(yaml_path) as f:
            obs_data = yaml.safe_load(f)

        marker_array = MarkerArray()
        mid = 0
        now = rospy.Time.now()

        def _base_marker(ns, shape, r, g, b):
            m               = Marker()
            m.header.frame_id = "world"
            m.header.stamp  = now
            m.ns            = ns
            m.id            = mid
            m.type          = shape
            m.action        = Marker.ADD
            m.color.r       = r
            m.color.g       = g
            m.color.b       = b
            m.color.a       = 0.8
            m.pose.orientation.w = 1.0
            return m

        # Toll pillars — orange cubes
        for pillar in obs_data.get("toll_pillars") or []:
            m = _base_marker("toll_pillars", Marker.CUBE, 1.0, 0.5, 0.0)
            m.id = mid; mid += 1
            cx, cy = pillar["center_m"]
            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = 0.5
            ang = math.radians(pillar["angle_deg"])
            q   = tft.quaternion_from_euler(0.0, 0.0, ang)
            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]
            m.scale.x = pillar["width_m"]
            m.scale.y = pillar["depth_m"]
            m.scale.z = 1.0
            marker_array.markers.append(m)

        # Narrow regions — blue flat boxes marking the bottleneck centre
        for nr in obs_data.get("narrow_regions") or []:
            m = _base_marker("narrow_regions", Marker.CUBE, 0.0, 0.4, 1.0)
            m.id = mid; mid += 1
            cx, cy = nr["center_m"]
            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = 0.1
            ang = math.radians(nr.get("angle_deg", 0.0))
            q   = tft.quaternion_from_euler(0.0, 0.0, ang)
            m.pose.orientation.x = q[0]; m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]; m.pose.orientation.w = q[3]
            m.scale.x = nr.get("block_length_m", nr["gap_width_m"])
            m.scale.y = nr["gap_width_m"]
            m.scale.z = 0.2
            marker_array.markers.append(m)

        # Obstacles — red cylinder (circle) or red cube (rect)
        for obs in obs_data.get("obstacles") or []:
            shape = Marker.CYLINDER if obs["type"] == "circle" else Marker.CUBE
            m = _base_marker("obstacles", shape, 1.0, 0.0, 0.0)
            m.id = mid; mid += 1
            cx, cy = obs["center_m"]
            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = 0.5
            if obs["type"] == "circle":
                r = obs["radius_m"]
                m.scale.x = r * 2
                m.scale.y = r * 2
                m.scale.z = 1.0
            else:
                ang = math.radians(obs.get("angle_deg", 0.0))
                q   = tft.quaternion_from_euler(0.0, 0.0, ang)
                m.pose.orientation.x = q[0]; m.pose.orientation.y = q[1]
                m.pose.orientation.z = q[2]; m.pose.orientation.w = q[3]
                m.scale.x = obs["width_m"]
                m.scale.y = obs["height_m"]
                m.scale.z = 1.0
            marker_array.markers.append(m)

        self.obstacle_pub.publish(marker_array)
        rospy.loginfo(
            "[Bridge] Published %d obstacle marker(s) on /obstacle_features",
            len(marker_array.markers),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("[Bridge] Spinning…")
        self._reset_zone()   # start collector for episode 1

        while not rospy.is_shutdown():
            now = rospy.Time.now()

            # — Build action array from latest cmd_vel —
            for i in range(self.num_agents):
                age = (now - self.cmd_stamps[i]).to_sec()
                if age < self.cmd_timeout:
                    self.action[i] = self._twist_to_action(self.cmd_vels[i])
                else:
                    # Stale command → coast to stop
                    self.action[i] = np.array([0.0, 0.0])

            # — Pump Pygame event queue (human display only) —
            if self.render and not self._record_video_path and pygame.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        rospy.signal_shutdown("Pygame window closed")

            # — Step the gym —
            try:
                self.obs, _reward, done, _trunc, _info = self.env.step(self.action)
                if self._record_video_path:
                    self._record_frame()
                elif self.render:
                    self.env.render()
            except Exception as exc:
                rospy.logerr("[Bridge] gym.step() error: %s", exc)
                rate.sleep()
                continue

            # — Collect full-run metrics —
            if self._zone is not None:
                self._zone.step(self.obs)

            # — Publish state for each agent —
            for i in range(self.num_agents):
                self._publish_odom(i, now)
                self._publish_depth_image(i, now)

            # — Detect episode end (zone cleared / map goal / collision / timeout) —
            if self._zone is not None:
                if self._episode_end_mode == "narrow_clear":
                    episode_complete = self._zone.all_zone_cleared
                else:
                    episode_complete = self._zone.all_done
            else:
                episode_complete = False
            timed_out = (
                self._episode_start_ts is not None
                and (now - self._episode_start_ts).to_sec() > self._episode_timeout
            )
            gym_done = done and not self._ignore_gym_done
            ep_ended = gym_done or episode_complete or timed_out

            if done and self._ignore_gym_done and not episode_complete and not timed_out:
                rospy.logwarn_throttle(
                    5.0,
                    "[Bridge] Gym done (collision/lap) — continuing for video (ignore_gym_done)",
                )

            if ep_ended:
                reason = self._episode_end_reason(
                    gym_done, episode_complete, timed_out
                )
                rospy.logwarn("[Bridge] Episode %d ended — %s",
                              self._episode_n + 1, reason)
                self._flush_zone()   # writes CSV and increments _episode_n

                # Auto-shutdown after target_episodes completed
                if self._episode_n >= self._target_episodes:
                    rospy.loginfo(
                        "[Bridge] Reached target_episodes=%d — shutting down",
                        self._target_episodes,
                    )
                    rospy.signal_shutdown("target episodes reached")
                    break

                # Otherwise reset for the next episode
                init_np = np.array(self.init_poses[:self.num_agents], dtype=np.float64)
                self.obs, self.info = self.env.reset(options={"poses": init_np})
                self._warn_spawn_collisions()
                self.action[:] = 0.0
                self._reset_zone()   # fresh collector for next episode
                self._record_frame()

            rate.sleep()


if __name__ == "__main__":
    try:
        F1TenthBridge().run()
    except rospy.ROSInterruptException:
        pass
