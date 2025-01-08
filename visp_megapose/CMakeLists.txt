cmake_minimum_required(VERSION 3.0.2)
project(visp_megapose)

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  sensor_msgs
  std_msgs
  image_transport
  roscpp
  tf2_ros
  tf2
  visp_bridge
  cv_bridge
  rospy
  message_generation
)

# Define message files
add_message_files(
  FILES
  Confidence.msg
)

# Define service files
add_service_files(
  FILES
  Init.srv
  Render.srv
  Track.srv
)

# After defining message and service files, call generate_messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs  # Add sensor_msgs as a dependency
)

find_package(VISP REQUIRED COMPONENTS visp_core visp_detection gui io)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
  geometry_msgs 
  message_filters 
  sensor_msgs 
  std_msgs 
  image_transport 
  roscpp 
  tf2_ros 
  tf2 
  visp_bridge 
  cv_bridge 
  DEPENDS VISP
)

add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/megapose_server_.py
  scripts/test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY data/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
)

install(FILES 
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/megapose_variables_final.json
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)
