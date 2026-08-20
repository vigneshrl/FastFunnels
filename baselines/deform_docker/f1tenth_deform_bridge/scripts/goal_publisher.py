#!/usr/bin/env python3
"""
goal_publisher.py
=================
Publishes a single /move_base_simple/goal after all plan_manager nodes
are ready.  Replaces the RViz2 "2D Nav Goal" click for automated runs.

Goal position loaded from ROS params (set in bridge_params.yaml):
  goal_x      : goal x in map frame  (default: -10.5)
  goal_y      : goal y in map frame  (default: 27.0)
  goal_yaw    : goal heading in rad  (default: 1.53  ≈ north)
  goal_delay_s: seconds to wait before publishing (default: 5.0)
  tb_num      : number of agents — waits for this many plan_manager nodes

For full_narrow map, default goal is well past the narrow zone exit.
"""

import math
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft


def main():
    rospy.init_node("goal_publisher", anonymous=True)

    goal_x    = rospy.get_param("~goal_x",       -1.2583960096163425)
    goal_y    = rospy.get_param("~goal_y",         20.228066212248873)
    goal_yaw  = rospy.get_param("~goal_yaw",        1.53)
    delay_s   = rospy.get_param("~goal_delay_s",    5.0)
    tb_num    = rospy.get_param("~tb_num",           4)

    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped,
                          queue_size=1, latch=True)

    # Wait for all plan_manager cmd_vel topics to appear
    rospy.loginfo("[GoalPub] Waiting for %d plan_manager node(s)…", tb_num)
    for i in range(tb_num):
        topic = f"/tb_{i}/cmd_vel"
        rospy.loginfo("[GoalPub] Waiting for publisher on %s", topic)
        while not rospy.is_shutdown():
            pubs = rospy.get_published_topics()
            if any(topic in t for t, _ in pubs):
                break
            rospy.sleep(0.5)

    # Extra settle time for planners to fully init
    rospy.loginfo("[GoalPub] All nodes up — publishing goal in %.1fs…", delay_s)
    rospy.sleep(delay_s)

    msg = PoseStamped()
    msg.header.stamp    = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.position.x = goal_x
    msg.pose.position.y = goal_y
    msg.pose.position.z = 0.0
    q = tft.quaternion_from_euler(0.0, 0.0, goal_yaw)
    msg.pose.orientation.x = q[0]
    msg.pose.orientation.y = q[1]
    msg.pose.orientation.z = q[2]
    msg.pose.orientation.w = q[3]

    pub.publish(msg)
    rospy.loginfo("[GoalPub] Goal published → (%.2f, %.2f) yaw=%.2f rad",
                  goal_x, goal_y, goal_yaw)

    rospy.spin()


if __name__ == "__main__":
    main()
