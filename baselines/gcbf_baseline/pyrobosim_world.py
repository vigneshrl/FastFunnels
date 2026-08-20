"""PyRoboSim world built from a FastFunnels eval-map corridor.

This is the execution environment for the GCBF+ baseline.  It keeps the shape of
the hand-authored narrow world (one room whose footprint runs broad -> pinch ->
narrow, N agents lined up across the entrance, poses advanced by integrating
velocity commands and reverted on collision) but takes its footprint from a real
eval map instead of hard-coded waypoints.

Collision is PyRoboSim's own test: ``check_occupancy`` against
``world.total_internal_polygon``, i.e. the room polygon deflated by the robot
radius with obstacle holes removed.  Nothing here re-implements it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyrobosim.core.robot import Robot
from pyrobosim.core.world import World
from pyrobosim.utils.pose import Pose
from pyrobosim.utils.world_collision import check_occupancy

from .corridor_geom import CorridorGeom, ZoneWindow

__all__ = ["CorridorWorld", "build_world", "formation_poses",
           "formation_layout"]


def formation_layout(
    geom: CorridorGeom,
    wp: int,
    n_agents: int,
    spacing: float,
    robot_radius: float,
    margin: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``n_agents`` into the corridor at waypoint ``wp``.

    Returns ``(lateral, back)``: each agent's signed lateral offset in metres
    and how many waypoints behind ``wp`` its rank sits.  The requested
    ``spacing`` shrinks to the corridor width, and where even the shrunk rank
    would overlap, the formation spills into successive ranks -- four abreast in
    a wide entrance, single file through a pinch.
    """
    pitch_min = 2.0 * robot_radius + 0.15

    def room(at: int) -> float:
        return max(geom.clearance(at) - robot_radius - margin, 0.0)

    if room(wp) <= 0.0:
        raise ValueError(
            f"{geom.name}: waypoint {wp} has clearance "
            f"{geom.clearance(wp):.2f} m, too tight for a robot of radius "
            f"{robot_radius:.2f} m"
        )

    lateral: list[float] = []
    back: list[int] = []
    remaining, rank = n_agents, 0
    while remaining > 0:
        at = max(0, wp - rank)
        half_room = room(at)
        # How many fit abreast here, at the requested pitch but never tighter
        # than the discs themselves.
        fits = max(1, int(2.0 * half_room / pitch_min) + 1)
        k = min(remaining, fits, max(1, int(2.0 * half_room / spacing) + 1))
        if k == 1:
            lateral.append(0.0)
        else:
            span = min(spacing * (k - 1), 2.0 * half_room)
            lateral.extend(np.linspace(-span / 2.0, span / 2.0, k))
        back.extend([rank] * k)
        remaining -= k
        rank += 1
        if rank > n_agents:  # pragma: no cover - room() > 0 guarantees progress
            raise ValueError(f"{geom.name}: cannot fit {n_agents} agents at {wp}")

    return np.asarray(lateral), np.asarray(back, dtype=int)


def formation_poses(
    geom: CorridorGeom,
    wp: int,
    lateral: np.ndarray,
    back: np.ndarray | None = None,
) -> np.ndarray:
    """Place agents on the given layout around waypoint ``wp``.

    Returns an ``(n, 3)`` array of ``(x, y, yaw)``, yaw along the centreline.
    """
    if back is None:
        back = np.zeros(len(lateral), dtype=int)

    poses = np.empty((len(lateral), 3))
    for i, (off, b) in enumerate(zip(lateral, back)):
        at = max(0, wp - int(b))
        centre = geom.centerline[at]
        tangent = geom.tangent(at)
        normal = geom.normal(at)
        p = centre + normal * float(off)
        poses[i] = (p[0], p[1], float(np.arctan2(tangent[1], tangent[0])))
    return poses


@dataclass
class CorridorWorld:
    """A PyRoboSim world plus the spawn/goal formations for one eval map."""

    world: World
    robots: list[Robot]
    spawn: np.ndarray  # (n, 3) x, y, yaw
    goals: np.ndarray  # (n, 3) x, y, yaw
    slots: np.ndarray  # (n,) signed lateral offset each agent holds, metres
    slot_back: np.ndarray  # (n,) rank each agent holds, in waypoints behind
    geom: CorridorGeom
    window: ZoneWindow
    robot_radius: float

    def slot_poses(self, wp: int, margin: float = 0.10) -> np.ndarray:
        """The formation's slots re-fitted to the corridor width at ``wp``.

        Used for the receding carrot goal: agents keep their lateral ordering
        and their rank, but the formation squeezes together where the corridor
        pinches.
        """
        half_room = max(
            self.geom.clearance(wp) - self.robot_radius - margin, 0.0
        )
        widest = float(np.max(np.abs(self.slots))) if len(self.slots) else 0.0
        shrink = 1.0 if widest <= half_room or widest == 0.0 else half_room / widest
        return formation_poses(self.geom, wp, self.slots * shrink, self.slot_back)

    def set_poses(self, states: np.ndarray) -> None:
        """Write ``(n, 3)`` x/y/yaw into the PyRoboSim robots."""
        for robot, (x, y, yaw) in zip(self.robots, states):
            robot.set_pose(Pose(x=float(x), y=float(y), yaw=float(yaw)))

    def wall_collisions(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask of agents whose centre is in collision with the world."""
        return np.array(
            [check_occupancy((float(x), float(y)), self.world) for x, y in xy],
            dtype=bool,
        )

    def agent_collisions(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask of agents overlapping another agent."""
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return (d < 2.0 * self.robot_radius).any(axis=1)


def build_world(
    geom: CorridorGeom,
    window: ZoneWindow,
    n_agents: int = 4,
    robot_radius: float = 0.29,
    spacing: float = 1.2,
    max_linear_velocity: float = np.inf,
    max_angular_velocity: float = np.inf,
) -> CorridorWorld:
    """Build the PyRoboSim world for one eval map."""
    world = World(name=f"corridor_{geom.name}", inflation_radius=robot_radius)

    # PyRoboSim's Room takes a simple coordinate ring, so seed it with the
    # corridor shell and then swap in the real polygon, whose holes are the
    # obstacles that were rasterised into the eval map clear of the walls.
    shell = [[float(x), float(y)] for x, y in geom.walkable.exterior.coords]
    room = world.add_room(name="corridor", footprint=shell, color=[0.8, 0.8, 0.8])
    if room is None:
        raise RuntimeError(f"{geom.name}: PyRoboSim rejected the corridor footprint")
    room.polygon = geom.walkable
    room.update_collision_polygons(world.inflation_radius)
    room.update_visualization_polygon()
    world.update_polygons()

    slots, slot_back = formation_layout(
        geom, window.start_wp, n_agents, spacing, robot_radius)
    spawn = formation_poses(geom, window.start_wp, slots, slot_back)
    goal_lat, goal_back = formation_layout(
        geom, window.goal_wp, n_agents, spacing, robot_radius)
    goals = formation_poses(geom, window.goal_wp, goal_lat, goal_back)

    robots: list[Robot] = []
    for i in range(n_agents):
        x, y, yaw = spawn[i]
        robot = Robot(
            name=f"agent_{i + 1}",
            radius=robot_radius,
            color=[0.2, 0.6, 0.9],
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            start_sensor_threads=False,
        )
        world.add_robot(robot, pose=Pose(x=float(x), y=float(y), yaw=float(yaw)),
                        show=False)
        robots.append(robot)

    return CorridorWorld(
        world=world,
        robots=robots,
        spawn=spawn,
        goals=goals,
        slots=slots,
        slot_back=slot_back,
        geom=geom,
        window=window,
        robot_radius=robot_radius,
    )
