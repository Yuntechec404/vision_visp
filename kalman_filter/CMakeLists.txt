cmake_minimum_required(VERSION 3.5)
project(kalman_filter)

# Find necessary build dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(visp_bridge REQUIRED)
find_package(VISP REQUIRED COMPONENTS visp_core visp_detection dnn_tracker gui io)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visp_megapose REQUIRED)
# Uncomment if you're using custom messages
# include_directories(/home/user/ros2_ws/src/vision_visp/visp_megapose/msg)
# find_package(rosidl_default_generators REQUIRED)
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Confidence.msg"
#   DEPENDENCIES std_msgs
# )

# Include Eigen3 directories
include_directories(include)

# Specify dependencies for the target executable
add_executable(kalman_filter_ src/kalman_filter_.cpp)
ament_target_dependencies(kalman_filter_
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  tf2_ros
  Eigen3
  visp_bridge
  visp_megapose
  nav_msgs
)
add_executable(average_filter src/average_filter.cpp)
ament_target_dependencies(average_filter
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visp_bridge
  visp_megapose
  nav_msgs
)
# Uncomment if you're generating custom message typesupports
# rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(kalman_filter_ ${typesupport_target})

# Install C++ executable
install(TARGETS
  kalman_filter_
  average_filter
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python node
# install(PROGRAMS
#   src/kalman_filter_node.py
#   DESTINATION lib/${PROJECT_NAME}
# )

# Install other directories (launch, params, etc.)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Export ROS 2 runtime dependencies
ament_export_dependencies(rosidl_default_runtime)

ament_package()
