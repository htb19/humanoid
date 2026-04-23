cmake_minimum_required(VERSION 3.8)
project(robot_workspace_quality_analyzer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(workspace_quality_analyzer_node
  src/workspace_quality_analyzer_node.cpp
)

target_include_directories(workspace_quality_analyzer_node
  PRIVATE
    include
)

ament_target_dependencies(workspace_quality_analyzer_node
  geometry_msgs
  moveit_core
  moveit_ros_planning
  rclcpp
  std_msgs
  std_srvs
  tf2_eigen
  visualization_msgs
)

target_link_libraries(workspace_quality_analyzer_node
  Eigen3::Eigen
)

install(TARGETS workspace_quality_analyzer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
