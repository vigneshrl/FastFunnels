cmake_minimum_required(VERSION 3.0.2)
project(f1tenth_deform_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  tf
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/f1tenth_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
