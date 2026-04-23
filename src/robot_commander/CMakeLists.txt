cmake_minimum_required(VERSION 3.8)
project(robot_commander)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(robot_interfaces REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

add_executable(commander src/commander_node.cpp)
ament_target_dependencies(commander rclcpp moveit_ros_planning_interface example_interfaces robot_interfaces tf2_geometry_msgs)

add_executable(move_to_pose_init src/move_to_pose_init_node.cpp)
ament_target_dependencies(move_to_pose_init rclcpp moveit_ros_planning_interface)

install(TARGETS 
  commander
  move_to_pose_init
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
