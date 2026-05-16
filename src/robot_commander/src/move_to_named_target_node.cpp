#include <memory>
#include <string>
#include <thread>

#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/rclcpp.hpp>

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

namespace
{

bool moveToNamedTarget(
  const rclcpp::Logger & logger,
  const std::shared_ptr<MoveGroupInterface> & move_group,
  const std::string & group_name,
  const std::string & target_name)
{
  move_group->setStartStateToCurrentState();
  move_group->setNamedTarget(target_name);

  MoveGroupInterface::Plan plan;
  const auto plan_result = move_group->plan(plan);
  if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(
      logger, "Planning failed for group '%s' to named target '%s'.",
      group_name.c_str(), target_name.c_str());
    return false;
  }

  const auto exec_result = move_group->execute(plan);
  if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(
      logger, "Execution failed for group '%s' to named target '%s'.",
      group_name.c_str(), target_name.c_str());
    return false;
  }

  RCLCPP_INFO(
    logger, "Group '%s' reached named target '%s'.",
    group_name.c_str(), target_name.c_str());
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("move_to_named_target");

  node->declare_parameter<std::string>("pose_name", "Pose_init");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  std::string pose_name;
  node->get_parameter("pose_name", pose_name);

  bool success = false;

  {
    auto right_arm = std::make_shared<MoveGroupInterface>(node, "right_arm");
    auto left_arm = std::make_shared<MoveGroupInterface>(node, "left_arm");

    right_arm->setMaxVelocityScalingFactor(1.0);
    right_arm->setMaxAccelerationScalingFactor(1.0);
    left_arm->setMaxVelocityScalingFactor(1.0);
    left_arm->setMaxAccelerationScalingFactor(1.0);

    const bool right_ok = moveToNamedTarget(node->get_logger(), right_arm, "right_arm", pose_name);
    const bool left_ok = moveToNamedTarget(node->get_logger(), left_arm, "left_arm", pose_name);
    success = right_ok && left_ok;
  }

  executor.cancel();
  if (spinner.joinable()) {
    spinner.join();
  }

  rclcpp::shutdown();
  return success ? 0 : 1;
}
