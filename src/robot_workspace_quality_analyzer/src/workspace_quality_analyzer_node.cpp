#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_srvs/srv/trigger.hpp>
#include "robot_workspace_quality_analyzer/srv/analyze_workspace.hpp"
#include "robot_workspace_quality_analyzer/srv/analyze_orientation.hpp"
#include <tf2_eigen/tf2_eigen.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <future>
#include <mutex>

namespace
{
constexpr double kEpsilon = 1e-9;

double clamp01(double value)
{
  return std::max(0.0, std::min(1.0, value));
}

std_msgs::msg::ColorRGBA color(double r, double g, double b, double a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = static_cast<float>(r);
  c.g = static_cast<float>(g);
  c.b = static_cast<float>(b);
  c.a = static_cast<float>(a);
  return c;
}

std::string timestamp()
{
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}

Eigen::Vector3d axisFromString(const std::string& s)
{
  if (s == "x") return Eigen::Vector3d::UnitX();
  if (s == "y") return Eigen::Vector3d::UnitY();
  return Eigen::Vector3d::UnitZ();
}

Eigen::Quaterniond safeAlignAxis(const Eigen::Vector3d& from, const Eigen::Vector3d& to)
{
  const Eigen::Vector3d a = from.normalized();
  const Eigen::Vector3d b = to.normalized();
  const double dot = a.dot(b);

  if (dot > 1.0 - kEpsilon)
    return Eigen::Quaterniond::Identity();

  if (dot < -1.0 + kEpsilon)
  {
    const Eigen::Vector3d perp = (std::abs(a.z()) < 0.99)
        ? Eigen::Vector3d::UnitZ().cross(a).normalized()
        : Eigen::Vector3d::UnitX().cross(a).normalized();
    return Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, perp));
  }

  return Eigen::Quaterniond::FromTwoVectors(a, b);
}

Eigen::Quaterniond directionRollToQuaternion(
    const Eigen::Vector3d& dir, double roll, const Eigen::Vector3d& tool_axis)
{
  const Eigen::Quaterniond q_align = safeAlignAxis(tool_axis, dir);
  const Eigen::Quaterniond q_roll(Eigen::AngleAxisd(roll, dir));
  return q_roll * q_align;
}

std::vector<Eigen::Vector3d> generateFibonacciSphere(int n_points)
{
  std::vector<Eigen::Vector3d> points;
  points.reserve(n_points);
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  for (int i = 0; i < n_points; ++i)
  {
    const double y = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n_points);
    const double r = std::sqrt(std::max(0.0, 1.0 - y * y));
    const double theta = 2.0 * M_PI * static_cast<double>(i) / phi;
    points.emplace_back(r * std::cos(theta), y, r * std::sin(theta));
  }
  return points;
}

std::vector<Eigen::Vector3d> generateFibonacciCap(
    const Eigen::Vector3d& dir_ref, double half_angle, int n_points)
{
  const double cos_min = std::cos(half_angle);
  const double z_range = 1.0 - cos_min;
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;

  std::vector<Eigen::Vector3d> local_points;
  local_points.reserve(n_points);
  for (int i = 0; i < n_points; ++i)
  {
    const double z = cos_min + z_range * (static_cast<double>(i) + 0.5) / static_cast<double>(n_points);
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double theta = 2.0 * M_PI * static_cast<double>(i) / phi;
    local_points.emplace_back(r * std::cos(theta), r * std::sin(theta), z);
  }

  const Eigen::Quaterniond q_rot = safeAlignAxis(Eigen::Vector3d::UnitZ(), dir_ref);
  std::vector<Eigen::Vector3d> world_points;
  world_points.reserve(local_points.size());
  for (const auto& p : local_points)
  {
    world_points.push_back(q_rot * p);
  }
  return world_points;
}

}  // namespace

class WorkspaceQualityAnalyzer : public rclcpp::Node
{
public:
  WorkspaceQualityAnalyzer() : Node("workspace_quality_analyzer")
  {
    loadParameters();

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 1);
    best_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(best_pose_topic_, 1);
    using AnalyzeWorkspace = robot_workspace_quality_analyzer::srv::AnalyzeWorkspace;
    analyze_srv_ = create_service<AnalyzeWorkspace>(
        "workspace_quality_analyzer/analyze", std::bind(&WorkspaceQualityAnalyzer::analyzeService, this, std::placeholders::_1, std::placeholders::_2));

    move_srv_ = create_service<std_srvs::srv::Trigger>(
        "workspace_quality_analyzer/move_to_best", std::bind(&WorkspaceQualityAnalyzer::moveToBestService, this, std::placeholders::_1, std::placeholders::_2));

    using AnalyzeOrientation = robot_workspace_quality_analyzer::srv::AnalyzeOrientation;
    orientation_srv_ = create_service<AnalyzeOrientation>(
        "workspace_quality_analyzer/analyze_orientation", std::bind(&WorkspaceQualityAnalyzer::analyzeOrientationService, this, std::placeholders::_1, std::placeholders::_2));
  }

  void initializeMoveIt()
  {
    robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(shared_from_this(), robot_description_param_);
    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_)
    {
      throw std::runtime_error("Failed to load robot model from parameter: " + robot_description_param_);
    }

    joint_model_group_ = robot_model_->getJointModelGroup(move_group_name_);
    if (!joint_model_group_)
    {
      throw std::runtime_error("MoveIt planning group not found: " + move_group_name_);
    }

    if (ee_link_.empty())
    {
      ee_link_ = joint_model_group_->getLinkModelNames().back();
      RCLCPP_WARN(get_logger(), "Parameter ee_link is empty. Using last link in group: %s", ee_link_.c_str());
    }

    ee_link_model_ = robot_model_->getLinkModel(ee_link_);
    if (!ee_link_model_)
    {
      throw std::runtime_error("End-effector link not found in robot model: " + ee_link_);
    }

    planning_scene_monitor_ =
        std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(shared_from_this(), robot_description_param_);
    if (!planning_scene_monitor_->getPlanningScene())
    {
      throw std::runtime_error("Failed to create PlanningSceneMonitor.");
    }
    planning_scene_monitor_->startStateMonitor();
    planning_scene_monitor_->startSceneMonitor();
    planning_scene_monitor_->startWorldGeometryMonitor();

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), move_group_name_);
    move_group_->startStateMonitor();
    if (!move_group_->getCurrentState(5.0))
    {
      RCLCPP_ERROR(get_logger(), "Failed to get current robot state after MoveGroup init.");
    }

    RCLCPP_INFO(get_logger(), "Workspace quality analyzer is ready. Call service: %s/analyze", get_fully_qualified_name());
  }

private:
  enum class RiskLevel
  {
    GOOD,
    WARNING,
    BAD,
    UNREACHABLE,
    COLLISION
  };

  enum class BestResultSource
  {
    POSITION_SCAN,
    ORIENTATION_SCAN
  };

  struct CurrentStateSnapshot
  {
    moveit::core::RobotState state;
    std::vector<double> joint_values;
    Eigen::Isometry3d ee_pose;
    Eigen::Quaterniond ee_orientation;
  };

  struct CandidateResult
  {
    int index{ 0 };
    geometry_msgs::msg::Pose pose;
    bool ik_success{ false };
    bool collision{ false };
    double score{ 0.0 };
    RiskLevel risk_level{ RiskLevel::UNREACHABLE };
    std::string risk_reason{ "IK_FAILED" };
    double sigma_min{ 0.0 };
    double condition_number{ std::numeric_limits<double>::infinity() };
    double manipulability{ 0.0 };
    double joint_margin_min{ 0.0 };
    double q_distance{ 0.0 };
    double ik_time_ms{ 0.0 };
    double orientation_error_rad{ 0.0 };
    BestResultSource source{ BestResultSource::POSITION_SCAN };
    std::vector<double> joint_values;
  };

  void loadParameters()
  {
    move_group_name_ = declare_parameter<std::string>("move_group_name", "arm");
    ee_link_ = declare_parameter<std::string>("ee_link", "");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    robot_description_param_ = declare_parameter<std::string>("robot_description_param", "robot_description");

    sphere_radius_ = declare_parameter<double>("sphere_radius", 0.15);
    sample_shape_ = declare_parameter<std::string>("sample_shape", "box");
    samples_per_dim_ = declare_parameter<int>("samples_per_dim", 11);

    ik_timeout_ = declare_parameter<double>("ik_timeout", 0.02);
    state_wait_timeout_ = declare_parameter<double>("state_wait_timeout", 2.0);

    sigma_min_warn_ = declare_parameter<double>("sigma_min_warn", 0.05);
    sigma_min_bad_ = declare_parameter<double>("sigma_min_bad", 0.02);
    condition_warn_ = declare_parameter<double>("condition_warn", 80.0);
    condition_bad_ = declare_parameter<double>("condition_bad", 200.0);
    joint_margin_warn_ = declare_parameter<double>("joint_margin_warn", 0.08);
    joint_margin_bad_ = declare_parameter<double>("joint_margin_bad", 0.03);
    q_distance_warn_ = declare_parameter<double>("q_distance_warn", 0.35);
    q_distance_bad_ = declare_parameter<double>("q_distance_bad", 0.65);

    sigma_min_good_ = declare_parameter<double>("sigma_min_good", 0.12);
    joint_margin_good_ = declare_parameter<double>("joint_margin_good", 0.20);

    csv_output_dir_ = declare_parameter<std::string>("csv_output_dir", "/tmp/workspace_quality");
    marker_topic_ = declare_parameter<std::string>("marker_topic", "/workspace_quality/markers");
    best_pose_topic_ = declare_parameter<std::string>("best_pose_topic", "/workspace_quality/best_pose");
    publish_markers_ = declare_parameter<bool>("publish_markers", true);
    write_csv_ = declare_parameter<bool>("write_csv", true);
    marker_scale_ = declare_parameter<double>("marker_scale", 1.0);

    so3_sample_count_ = declare_parameter<int>("so3_sample_count", 125);
    max_orientation_samples_ = declare_parameter<int>("max_orientation_samples", 1000);
    orientation_range_deg_ = declare_parameter<double>("orientation_range_deg", 90.0);
    orientation_scan_timeout_ = declare_parameter<double>("orientation_scan_timeout", 30.0);
    orientation_shell_radius_ = declare_parameter<double>("orientation_shell_radius", 0.08);
    orientation_visual_axis_ = declare_parameter<std::string>("orientation_visual_axis", "z");

    if (samples_per_dim_ < 2)
    {
      RCLCPP_WARN(get_logger(), "samples_per_dim must be >= 2. Using 11.");
      samples_per_dim_ = 11;
    }
  }

  CurrentStateSnapshot acquireCurrentState()
  {
    if (!planning_scene_monitor_->getStateMonitor()->waitForCurrentState(now(), state_wait_timeout_))
    {
      RCLCPP_WARN(get_logger(), "Timed out waiting for current robot state. Continuing with latest available state.");
    }

    planning_scene_monitor::LockedPlanningSceneRO scene(planning_scene_monitor_);
    moveit::core::RobotState current_state = scene->getCurrentState();
    current_state.update();

    Eigen::Isometry3d ee_pose = current_state.getGlobalLinkTransform(ee_link_model_);
    Eigen::Quaterniond ee_orientation(ee_pose.rotation());
    std::vector<double> joint_values;
    current_state.copyJointGroupPositions(joint_model_group_, joint_values);

    return { std::move(current_state), std::move(joint_values), ee_pose, ee_orientation };
  }

  void analyzeService(const std::shared_ptr<robot_workspace_quality_analyzer::srv::AnalyzeWorkspace::Request> request,
                      std::shared_ptr<robot_workspace_quality_analyzer::srv::AnalyzeWorkspace::Response> response)
  {
    try
    {
      double sphere_radius = request->sphere_radius > 0.0 ? request->sphere_radius : sphere_radius_;
      int samples_per_dim = request->samples_per_dim > 1 ? request->samples_per_dim : samples_per_dim_;

      const auto results = analyze(sphere_radius, samples_per_dim);
      const auto csv_path = write_csv_ ? writeCsv(results) : "";
      const double pos_scale = sphere_radius / static_cast<double>(samples_per_dim) * 1.2 * marker_scale_;
      publishResults(results, pos_scale);

      const auto best_it = std::max_element(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score < rhs.score;
      });

      if (best_it != results.end())
      {
        std::lock_guard<std::mutex> lock(best_result_mutex_);
        best_result_ = *best_it;

        std::ostringstream jv;
        for (size_t i = 0; i < best_it->joint_values.size(); ++i)
        {
          if (i > 0) jv << ", ";
          jv << std::fixed << std::setprecision(4) << best_it->joint_values[i];
        }
        RCLCPP_INFO(get_logger(), "Best position joint_values: [%s]", jv.str().c_str());
      }

      const int good_count = static_cast<int>(std::count_if(results.begin(), results.end(), [](const auto& result) {
        return result.risk_level == RiskLevel::GOOD;
      }));

      std::ostringstream message;
      message << "samples=" << results.size() << ", good=" << good_count << ", csv=" << csv_path
              << ", source=POSITION_SCAN";
      if (best_it != results.end())
      {
        message << ", best_score=" << std::fixed << std::setprecision(2) << best_it->score;
      }

      response->success = true;
      response->message = message.str();
    }
    catch (const std::exception& ex)
    {
      response->success = false;
      response->message = ex.what();
      RCLCPP_ERROR(get_logger(), "Analysis failed: %s", ex.what());
    }
  }

  void moveToBestService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    CandidateResult target;
    {
      std::lock_guard<std::mutex> lock(best_result_mutex_);
      target = best_result_;
    }

    if (!target.ik_success || target.collision)
    {
      RCLCPP_ERROR(get_logger(), "move_to_best: no valid best pose. Run /analyze first.");
      response->success = false;
      response->message = "No valid best pose available. Run analysis first.";
      return;
    }

    if (target.joint_values.empty() ||
        target.joint_values.size() != joint_model_group_->getVariableCount())
    {
      RCLCPP_ERROR(get_logger(), "move_to_best: joint_values size mismatch (expected %zu, got %zu).",
                   static_cast<size_t>(joint_model_group_->getVariableCount()), target.joint_values.size());
      response->success = false;
      response->message = "Joint values dimension mismatch.";
      return;
    }

    move_group_->setStartStateToCurrentState();

    bool ok = move_group_->setJointValueTarget(target.joint_values);
    if (!ok)
    {
      RCLCPP_ERROR(get_logger(), "move_to_best: setJointValueTarget failed (joint limit or NaN).");
      response->success = false;
      response->message = "Invalid joint target (limit or NaN).";
      return;
    }

    move_group_->setMaxVelocityScalingFactor(0.2);
    move_group_->setMaxAccelerationScalingFactor(0.2);
    move_group_->setPlanningTime(5.0);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto plan_error = move_group_->plan(plan);
    if (plan_error != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(get_logger(), "move_to_best: planning failed.");
      response->success = false;
      response->message = "Planning failed.";
      return;
    }

    auto exec_future = std::async(std::launch::async, [this, &plan]() {
      return move_group_->execute(plan);
    });
    auto status = exec_future.wait_for(std::chrono::seconds(30));
    if (status != std::future_status::ready)
    {
      RCLCPP_ERROR(get_logger(), "move_to_best: execution timed out.");
      response->success = false;
      response->message = "Execution timed out.";
      return;
    }

    auto exec_error = exec_future.get();
    if (exec_error != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(get_logger(), "move_to_best: execution failed.");
      response->success = false;
      response->message = "Execution failed.";
      return;
    }

    const std::string source_str =
        target.source == BestResultSource::ORIENTATION_SCAN ? "ORIENTATION_SCAN" : "POSITION_SCAN";
    RCLCPP_INFO(get_logger(), "move_to_best: success (source=%s).", source_str.c_str());
    response->success = true;
    response->message = "Moved to best pose. source=" + source_str;
  }

  void analyzeOrientationService(
      const std::shared_ptr<robot_workspace_quality_analyzer::srv::AnalyzeOrientation::Request> request,
      std::shared_ptr<robot_workspace_quality_analyzer::srv::AnalyzeOrientation::Response> response)
  {
    try
    {
      int so3_count = request->so3_sample_count > 0
          ? request->so3_sample_count : so3_sample_count_;
      double orient_range_deg = request->orientation_range_deg > 0.0
          ? request->orientation_range_deg : orientation_range_deg_;
      double orient_range = orient_range_deg * M_PI / 180.0;

      if (so3_count > max_orientation_samples_)
      {
        RCLCPP_WARN(get_logger(),
            "so3_sample_count %d exceeds max %d, clamping.", so3_count, max_orientation_samples_);
        so3_count = max_orientation_samples_;
      }
      if (so3_count < 2)
      {
        throw std::runtime_error("so3_sample_count must be >= 2.");
      }

      CurrentStateSnapshot snap = acquireCurrentState();
      const auto results = sampleOrientations(
          snap.ee_pose.translation(), snap.ee_orientation, snap,
          orient_range, so3_count, orientation_scan_timeout_);

      const auto csv_path = write_csv_ ? writeCsv(results) : "";
      const double orient_scale = orientation_shell_radius_
          / std::cbrt(static_cast<double>(so3_count)) * 1.5 * marker_scale_;
      publishOrientationResults(results, snap.ee_pose.translation(), orient_scale);

      // Layer 1: filter safe candidates
      std::vector<CandidateResult> valid;
      for (const auto& r : results)
      {
        if (r.risk_level == RiskLevel::GOOD || r.risk_level == RiskLevel::WARNING)
          valid.push_back(r);
      }

      const CandidateResult* best = nullptr;
      if (!valid.empty())
      {
        // Layer 2+3: multi-level sort (risk → score → orientation_error)
        std::sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
          if (a.risk_level != b.risk_level)
            return static_cast<int>(a.risk_level) < static_cast<int>(b.risk_level);
          if (std::abs(a.score - b.score) > kEpsilon)
            return a.score > b.score;
          return a.orientation_error_rad < b.orientation_error_rad;
        });
        best = &valid[0];
      }
      else
      {
        // Fallback: pick best among all by score then alignment
        auto best_it = std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) {
              if (std::abs(a.score - b.score) > kEpsilon)
                return a.score < b.score;
              return a.orientation_error_rad > b.orientation_error_rad;
            });
        if (best_it != results.end())
          best = &(*best_it);
      }

      if (best)
      {
        std::lock_guard<std::mutex> lock(best_result_mutex_);
        best_result_ = *best;

        std::ostringstream jv;
        for (size_t i = 0; i < best->joint_values.size(); ++i)
        {
          if (i > 0) jv << ", ";
          jv << std::fixed << std::setprecision(4) << best->joint_values[i];
        }
        RCLCPP_INFO(get_logger(), "Best orientation joint_values: [%s]", jv.str().c_str());
      }

      const int good_count = static_cast<int>(std::count_if(
          results.begin(), results.end(),
          [](const auto& r) { return r.risk_level == RiskLevel::GOOD; }));

      std::ostringstream message;
      message << "orientation_samples=" << results.size()
              << ", good=" << good_count
              << ", csv=" << csv_path;
      if (best)
      {
        message << ", best_score=" << std::fixed << std::setprecision(2) << best->score
                << ", best_error_rad=" << best->orientation_error_rad
                << ", source=ORIENTATION_SCAN";
      }

      response->success = true;
      response->message = message.str();
    }
    catch (const std::exception& ex)
    {
      response->success = false;
      response->message = ex.what();
      RCLCPP_ERROR(get_logger(), "Orientation analysis failed: %s", ex.what());
    }
  }

  std::vector<CandidateResult> analyze(double sphere_radius, int samples_per_dim)
  {
    const double radius = sphere_radius;
    const double resolution = 2.0 * sphere_radius / static_cast<double>(samples_per_dim - 1);

    CurrentStateSnapshot snap = acquireCurrentState();

    std::vector<CandidateResult> results;
    int index = 0;
    for (double dx = -radius; dx <= radius + kEpsilon; dx += resolution)
    {
      for (double dy = -radius; dy <= radius + kEpsilon; dy += resolution)
      {
        for (double dz = -radius; dz <= radius + kEpsilon; dz += resolution)
        {
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 > radius * radius + kEpsilon)
            continue;

          Eigen::Isometry3d target_pose = snap.ee_pose;
          target_pose.translation() = snap.ee_pose.translation() + Eigen::Vector3d(dx, dy, dz);

          CandidateResult result;
          result.index = index++;
          result.source = BestResultSource::POSITION_SCAN;
          result.pose.position.x = target_pose.translation().x();
          result.pose.position.y = target_pose.translation().y();
          result.pose.position.z = target_pose.translation().z();
          result.pose.orientation.x = snap.ee_orientation.x();
          result.pose.orientation.y = snap.ee_orientation.y();
          result.pose.orientation.z = snap.ee_orientation.z();
          result.pose.orientation.w = snap.ee_orientation.w();

          evaluateCandidate(snap.state, snap.joint_values, target_pose, result);
          results.push_back(result);
        }
      }
    }

    RCLCPP_INFO(get_logger(), "Finished workspace analysis with %zu samples.", results.size());
    return results;
  }

  void evaluateCandidate(const moveit::core::RobotState& ik_seed, const std::vector<double>& current_joint_values,
                         const Eigen::Isometry3d& target_pose, CandidateResult& result) const
  {
    moveit::core::RobotState candidate_state(ik_seed);

    const auto ik_start = std::chrono::steady_clock::now();
    result.ik_success = candidate_state.setFromIK(joint_model_group_, target_pose, ee_link_, ik_timeout_);
    const auto ik_end = std::chrono::steady_clock::now();
    result.ik_time_ms = std::chrono::duration<double, std::milli>(ik_end - ik_start).count();

    if (!result.ik_success)
    {
      result.risk_level = RiskLevel::UNREACHABLE;
      result.risk_reason = "IK_FAILED";
      return;
    }

    candidate_state.update();
    candidate_state.copyJointGroupPositions(joint_model_group_, result.joint_values);

    collision_detection::CollisionRequest collision_request;
    collision_detection::CollisionResult collision_result;
    collision_request.group_name = move_group_name_;
    {
      planning_scene_monitor::LockedPlanningSceneRO scene(planning_scene_monitor_);
      scene->checkCollision(collision_request, collision_result, candidate_state);
    }
    result.collision = collision_result.collision;
    if (result.collision)
    {
      result.risk_level = RiskLevel::COLLISION;
      result.risk_reason = "COLLISION";
      return;
    }

    computeJacobianMetrics(candidate_state, result);
    result.joint_margin_min = computeJointMargin(candidate_state);
    result.q_distance = computeJointDistance(current_joint_values, result.joint_values);
    classifyAndScore(result);
  }

  void computeJacobianMetrics(const moveit::core::RobotState& state, CandidateResult& result) const
  {
    Eigen::MatrixXd jacobian;
    state.getJacobian(joint_model_group_, ee_link_model_, Eigen::Vector3d::Zero(), jacobian);

    if (jacobian.rows() == 0 || jacobian.cols() == 0)
    {
      result.sigma_min = 0.0;
      result.condition_number = std::numeric_limits<double>::infinity();
      result.manipulability = 0.0;
      return;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian);
    const auto singular_values = svd.singularValues();
    const double sigma_max = singular_values.size() > 0 ? singular_values.maxCoeff() : 0.0;
    result.sigma_min = singular_values.size() > 0 ? singular_values.minCoeff() : 0.0;
    result.condition_number = result.sigma_min > kEpsilon ? sigma_max / result.sigma_min :
                                                        std::numeric_limits<double>::infinity();

    const Eigen::MatrixXd jj_t = jacobian * jacobian.transpose();
    const double determinant = jj_t.determinant();
    result.manipulability = determinant > 0.0 ? std::sqrt(determinant) : 0.0;
  }

  double computeJointMargin(const moveit::core::RobotState& state) const
  {
    double margin_min = std::numeric_limits<double>::infinity();
    const auto& active_joint_models = joint_model_group_->getActiveJointModels();
    for (const auto* joint_model : active_joint_models)
    {
      if (joint_model->getVariableCount() != 1)
      {
        continue;
      }

      const auto& bounds = joint_model->getVariableBounds()[0];
      if (!bounds.position_bounded_)
      {
        continue;
      }

      const double value = state.getVariablePosition(joint_model->getVariableNames()[0]);
      const double range = bounds.max_position_ - bounds.min_position_;
      if (range <= kEpsilon)
      {
        continue;
      }

      const double margin = std::min(value - bounds.min_position_, bounds.max_position_ - value) / range;
      margin_min = std::min(margin_min, margin);
    }

    return std::isfinite(margin_min) ? margin_min : 1.0;
  }

  double computeJointDistance(const std::vector<double>& current_joint_values,
                              const std::vector<double>& candidate_joint_values) const
  {
    if (current_joint_values.size() != candidate_joint_values.size() || current_joint_values.empty())
    {
      return 0.0;
    }

    double sum = 0.0;
    const auto& variable_names = joint_model_group_->getVariableNames();
    for (size_t i = 0; i < current_joint_values.size(); ++i)
    {
      double range = 1.0;
      if (i < variable_names.size())
      {
        const auto& bounds = robot_model_->getVariableBounds(variable_names[i]);
        if (bounds.position_bounded_)
        {
          const double bounded_range = bounds.max_position_ - bounds.min_position_;
          if (bounded_range > kEpsilon)
          {
            range = bounded_range;
          }
        }
      }
      const double diff = (candidate_joint_values[i] - current_joint_values[i]) / range;
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

  std::vector<CandidateResult> sampleOrientations(
      const Eigen::Vector3d& fixed_position,
      const Eigen::Quaterniond& reference_orientation,
      const CurrentStateSnapshot& snap,
      double half_range, int so3_sample_count, double timeout_sec) const
  {
    const Eigen::Vector3d tool_axis = axisFromString(orientation_visual_axis_);
    const Eigen::Vector3d dir_ref = reference_orientation * tool_axis;

    const double k_double = std::cbrt(static_cast<double>(so3_sample_count));
    const int k = std::max(2, static_cast<int>(std::round(k_double)));
    const int N_dir = k * k;
    const int N_rolls = k;

    std::vector<Eigen::Vector3d> cone_dirs;
    if (half_range < M_PI - kEpsilon)
    {
      cone_dirs = generateFibonacciCap(dir_ref, half_range, N_dir);
    }
    else
    {
      cone_dirs = generateFibonacciSphere(N_dir);
    }

    const auto scan_start = std::chrono::steady_clock::now();
    std::vector<CandidateResult> results;

    moveit::core::RobotState ik_seed(snap.state);

    for (const auto& dir : cone_dirs)
    {
      const auto elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - scan_start).count();
      if (elapsed > timeout_sec)
      {
        RCLCPP_WARN(get_logger(),
            "Orientation scan timed out after %.1fs, %zu results so far.",
            elapsed, results.size());
        break;
      }

      for (int ri = 0; ri < N_rolls; ++ri)
      {
        const auto elapsed_inner = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - scan_start).count();
        if (elapsed_inner > timeout_sec)
          break;

        double alpha;
        if (half_range < M_PI - kEpsilon)
        {
          alpha = (N_rolls > 1)
              ? -half_range + 2.0 * half_range * static_cast<double>(ri) / static_cast<double>(N_rolls - 1)
              : 0.0;
        }
        else
        {
          alpha = 2.0 * M_PI * static_cast<double>(ri) / static_cast<double>(N_rolls);
        }

        const Eigen::Quaterniond q = directionRollToQuaternion(dir, alpha, tool_axis);

        const double dot = std::abs(
            static_cast<double>(reference_orientation.coeffs().dot(q.coeffs())));
        const double theta = 2.0 * std::acos(clamp01(dot));

        Eigen::Isometry3d target_pose = Eigen::Isometry3d::Identity();
        target_pose.translation() = fixed_position;
        target_pose.linear() = q.toRotationMatrix();

        CandidateResult result;
        result.index = static_cast<int>(results.size());
        result.source = BestResultSource::ORIENTATION_SCAN;
        result.pose.position.x = fixed_position.x();
        result.pose.position.y = fixed_position.y();
        result.pose.position.z = fixed_position.z();
        result.pose.orientation.x = q.x();
        result.pose.orientation.y = q.y();
        result.pose.orientation.z = q.z();
        result.pose.orientation.w = q.w();
        result.orientation_error_rad = theta;

        evaluateCandidate(ik_seed, snap.joint_values, target_pose, result);

        if (result.ik_success)
        {
          ik_seed.setJointGroupPositions(joint_model_group_, result.joint_values);
          ik_seed.update();
        }
        else
        {
          ik_seed = moveit::core::RobotState(snap.state);
        }

        results.push_back(result);
      }
    }

    RCLCPP_INFO(get_logger(),
        "Orientation sweep: %zu samples at (%.3f, %.3f, %.3f).",
        results.size(), fixed_position.x(), fixed_position.y(), fixed_position.z());
    return results;
  }

  void classifyAndScore(CandidateResult& result) const
  {
    std::vector<std::string> reasons;
    bool bad = false;
    bool warn = false;

    if (result.sigma_min < sigma_min_bad_ || result.condition_number > condition_bad_)
    {
      bad = true;
      reasons.push_back("NEAR_SINGULARITY");
    }
    else if (result.sigma_min < sigma_min_warn_ || result.condition_number > condition_warn_)
    {
      warn = true;
      reasons.push_back("SINGULARITY_WARNING");
    }

    if (result.joint_margin_min < joint_margin_bad_)
    {
      bad = true;
      reasons.push_back("NEAR_JOINT_LIMIT");
    }
    else if (result.joint_margin_min < joint_margin_warn_)
    {
      warn = true;
      reasons.push_back("JOINT_LIMIT_WARNING");
    }

    if (result.q_distance > q_distance_bad_)
    {
      bad = true;
      reasons.push_back("FAR_FROM_CURRENT_STATE");
    }
    else if (result.q_distance > q_distance_warn_)
    {
      warn = true;
      reasons.push_back("Q_DISTANCE_WARNING");
    }

    const double singularity_score = clamp01(result.sigma_min / sigma_min_good_);
    const double joint_limit_score = clamp01(result.joint_margin_min / joint_margin_good_);
    const double q_distance_score = clamp01(1.0 - result.q_distance / q_distance_bad_);
    result.score = 45.0 * singularity_score + 35.0 * joint_limit_score + 20.0 * q_distance_score;

    if (bad)
    {
      result.risk_level = RiskLevel::BAD;
    }
    else if (warn)
    {
      result.risk_level = RiskLevel::WARNING;
    }
    else
    {
      result.risk_level = RiskLevel::GOOD;
      reasons.push_back("OK");
    }

    result.risk_reason.clear();
    for (size_t i = 0; i < reasons.size(); ++i)
    {
      if (i > 0)
      {
        result.risk_reason += "|";
      }
      result.risk_reason += reasons[i];
    }
  }

  std::string writeCsv(const std::vector<CandidateResult>& results) const
  {
    std::filesystem::create_directories(csv_output_dir_);
    const std::filesystem::path csv_path =
        std::filesystem::path(csv_output_dir_) / ("workspace_quality_" + timestamp() + ".csv");
    std::ofstream csv(csv_path);
    if (!csv)
    {
      throw std::runtime_error("Failed to open CSV path: " + csv_path.string());
    }

    csv << "index,x,y,z,qx,qy,qz,qw,ik_success,collision,score,risk_level,risk_reason,"
           "sigma_min,condition_number,manipulability,joint_margin_min,q_distance,"
           "orientation_error_rad,source,ik_time_ms,joint_values\n";

    for (const auto& result : results)
    {
      csv << result.index << "," << result.pose.position.x << "," << result.pose.position.y << ","
          << result.pose.position.z << "," << result.pose.orientation.x << "," << result.pose.orientation.y << ","
          << result.pose.orientation.z << "," << result.pose.orientation.w << "," << result.ik_success << ","
          << result.collision << "," << result.score << "," << riskLevelToString(result.risk_level) << ","
          << result.risk_reason << "," << result.sigma_min << "," << result.condition_number << ","
          << result.manipulability << "," << result.joint_margin_min << "," << result.q_distance << ","
          << result.orientation_error_rad << ","
          << (result.source == BestResultSource::ORIENTATION_SCAN ? "orientation" : "position") << ","
          << result.ik_time_ms << ",";

      for (size_t i = 0; i < result.joint_values.size(); ++i)
      {
        if (i > 0)
        {
          csv << ";";
        }
        csv << result.joint_values[i];
      }
      csv << "\n";
    }

    return csv_path.string();
  }

  void publishResults(const std::vector<CandidateResult>& results, double marker_scale) const
  {
    if (results.empty())
    {
      return;
    }

    const auto best_it = std::max_element(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.score < rhs.score;
    });

    geometry_msgs::msg::PoseStamped best_pose;
    best_pose.header.frame_id = base_frame_;
    best_pose.header.stamp = now();
    best_pose.pose = best_it->pose;
    best_pose_pub_->publish(best_pose);

    if (!publish_markers_)
    {
      return;
    }

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker delete_all;
    delete_all.header.frame_id = base_frame_;
    delete_all.header.stamp = now();
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_all);

    for (const auto& result : results)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = base_frame_;
      marker.header.stamp = now();
      marker.ns = "workspace_quality_samples";
      marker.id = result.index;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = result.pose;
      marker.scale.x = marker_scale;
      marker.scale.y = marker_scale;
      marker.scale.z = marker_scale;
      marker.color = colorForResult(result);
      markers.markers.push_back(marker);
    }

    visualization_msgs::msg::Marker best_marker;
    best_marker.header.frame_id = base_frame_;
    best_marker.header.stamp = now();
    best_marker.ns = "workspace_quality_best";
    best_marker.id = 0;
    best_marker.type = visualization_msgs::msg::Marker::SPHERE;
    best_marker.action = visualization_msgs::msg::Marker::ADD;
    best_marker.pose = best_it->pose;
    best_marker.scale.x = marker_scale * 2.0;
    best_marker.scale.y = marker_scale * 2.0;
    best_marker.scale.z = marker_scale * 2.0;
    best_marker.color = color(1.0, 1.0, 1.0, 1.0);
    markers.markers.push_back(best_marker);

    marker_pub_->publish(markers);
  }

  void publishOrientationResults(const std::vector<CandidateResult>& results,
                                  const Eigen::Vector3d& fixed_position,
                                  double marker_scale) const
  {
    if (results.empty())
      return;

    const auto best_it = std::max_element(results.begin(), results.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.score < rhs.score; });

    geometry_msgs::msg::PoseStamped best_pose;
    best_pose.header.frame_id = base_frame_;
    best_pose.header.stamp = now();
    best_pose.pose = best_it->pose;
    best_pose_pub_->publish(best_pose);

    if (!publish_markers_)
      return;

    const Eigen::Vector3d tool_axis = axisFromString(orientation_visual_axis_);

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker delete_all;
    delete_all.header.frame_id = base_frame_;
    delete_all.header.stamp = now();
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_all);

    for (const auto& result : results)
    {
      const Eigen::Quaterniond q(
          result.pose.orientation.w,
          result.pose.orientation.x,
          result.pose.orientation.y,
          result.pose.orientation.z);
      const Eigen::Vector3d shell_point =
          fixed_position + (q * tool_axis) * orientation_shell_radius_;

      // Shell point marker
      visualization_msgs::msg::Marker shell_marker;
      shell_marker.header.frame_id = base_frame_;
      shell_marker.header.stamp = now();
      shell_marker.ns = "workspace_quality_orientation_shell";
      shell_marker.id = result.index;
      shell_marker.type = visualization_msgs::msg::Marker::SPHERE;
      shell_marker.action = visualization_msgs::msg::Marker::ADD;
      shell_marker.pose.position.x = shell_point.x();
      shell_marker.pose.position.y = shell_point.y();
      shell_marker.pose.position.z = shell_point.z();
      shell_marker.pose.orientation.w = 1.0;
      shell_marker.scale.x = marker_scale;
      shell_marker.scale.y = marker_scale;
      shell_marker.scale.z = marker_scale;
      shell_marker.color = colorForResult(result, 0.7);
      markers.markers.push_back(shell_marker);
    }

    // Best marker (white, larger) on shell
    {
      const Eigen::Quaterniond q_best(
          best_it->pose.orientation.w,
          best_it->pose.orientation.x,
          best_it->pose.orientation.y,
          best_it->pose.orientation.z);
      const Eigen::Vector3d best_shell_point =
          fixed_position + (q_best * tool_axis) * orientation_shell_radius_;

      visualization_msgs::msg::Marker best_marker;
      best_marker.header.frame_id = base_frame_;
      best_marker.header.stamp = now();
      best_marker.ns = "workspace_quality_best";
      best_marker.id = 0;
      best_marker.type = visualization_msgs::msg::Marker::SPHERE;
      best_marker.action = visualization_msgs::msg::Marker::ADD;
      best_marker.pose.position.x = best_shell_point.x();
      best_marker.pose.position.y = best_shell_point.y();
      best_marker.pose.position.z = best_shell_point.z();
      best_marker.pose.orientation.w = 1.0;
      best_marker.scale.x = marker_scale * 2.0;
      best_marker.scale.y = marker_scale * 2.0;
      best_marker.scale.z = marker_scale * 2.0;
      best_marker.color = color(1.0, 1.0, 1.0, 1.0);
      markers.markers.push_back(best_marker);
    }

    marker_pub_->publish(markers);
  }

  std_msgs::msg::ColorRGBA colorForResult(const CandidateResult& result, double alpha_boost = 1.0) const
  {
    switch (result.risk_level)
    {
      case RiskLevel::GOOD:
        return color(0.0, 0.85, 0.25, 0.45 * alpha_boost);
      case RiskLevel::WARNING:
        return color(1.0, 0.85, 0.0, 0.45 * alpha_boost);
      case RiskLevel::BAD:
        return color(1.0, 0.05, 0.0, 0.45 * alpha_boost);
      case RiskLevel::UNREACHABLE:
        return color(0.45, 0.45, 0.45, 0.20 * alpha_boost);
      case RiskLevel::COLLISION:
        return color(0.65, 0.0, 0.85, 0.45 * alpha_boost);
    }
    return color(1.0, 1.0, 1.0, 1.0);
  }

  std::string riskLevelToString(RiskLevel risk_level) const
  {
    switch (risk_level)
    {
      case RiskLevel::GOOD:
        return "GOOD";
      case RiskLevel::WARNING:
        return "WARNING";
      case RiskLevel::BAD:
        return "BAD";
      case RiskLevel::UNREACHABLE:
        return "UNREACHABLE";
      case RiskLevel::COLLISION:
        return "COLLISION";
    }
    return "UNKNOWN";
  }

  std::string move_group_name_;
  std::string ee_link_;
  std::string base_frame_;
  std::string robot_description_param_;
  double sphere_radius_{ 0.15 };
  std::string sample_shape_;
  int samples_per_dim_{ 11 };
  moveit::planning_interface::MoveGroupInterfacePtr move_group_;
  CandidateResult best_result_;
  std::mutex best_result_mutex_;
  double ik_timeout_{ 0.02 };
  double state_wait_timeout_{ 2.0 };
  double sigma_min_warn_{ 0.05 };
  double sigma_min_bad_{ 0.02 };
  double condition_warn_{ 80.0 };
  double condition_bad_{ 200.0 };
  double joint_margin_warn_{ 0.08 };
  double joint_margin_bad_{ 0.03 };
  double q_distance_warn_{ 0.35 };
  double q_distance_bad_{ 0.65 };
  double sigma_min_good_{ 0.12 };
  double joint_margin_good_{ 0.20 };
  std::string csv_output_dir_;
  std::string marker_topic_;
  std::string best_pose_topic_;
  bool publish_markers_{ true };
  bool write_csv_{ true };
  double marker_scale_{ 1.0 };

  robot_model_loader::RobotModelLoaderPtr robot_model_loader_;
  moveit::core::RobotModelPtr robot_model_;
  const moveit::core::JointModelGroup* joint_model_group_{ nullptr };
  const moveit::core::LinkModel* ee_link_model_{ nullptr };
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

  int so3_sample_count_{ 125 };
  int max_orientation_samples_{ 1000 };
  double orientation_range_deg_{ 90.0 };
  double orientation_scan_timeout_{ 30.0 };
  double orientation_shell_radius_{ 0.08 };
  std::string orientation_visual_axis_{ "z" };

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr best_pose_pub_;
  rclcpp::Service<robot_workspace_quality_analyzer::srv::AnalyzeWorkspace>::SharedPtr analyze_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr move_srv_;
  rclcpp::Service<robot_workspace_quality_analyzer::srv::AnalyzeOrientation>::SharedPtr orientation_srv_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<WorkspaceQualityAnalyzer>();
    node->initializeMoveIt();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  }
  catch (const std::exception& ex)
  {
    std::cerr << "workspace_quality_analyzer_node failed: " << ex.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
