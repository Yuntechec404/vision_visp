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
  # visual_servoing
)

find_package(VISP REQUIRED COMPONENTS visp_core visp_detection dnn_tracker gui io)

catkin_package(
  INCLUDE_DIRS include
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
  # visual_servoing
  DEPENDS VISP
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
)

add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")

# Build the executable
# add_executable(megapose_client src/megapose_client.cpp)
# target_link_libraries(megapose_client ${catkin_LIBRARIES} ${VISP_LIBRARIES})
# add_dependencies(megapose_client ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Install targets and scripts
# install(TARGETS megapose_client
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

catkin_install_python(PROGRAMS
  scripts/megapose_server_.py
  scripts/image_process.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY data/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
)
