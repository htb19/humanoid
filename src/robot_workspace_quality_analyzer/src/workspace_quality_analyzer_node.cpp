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
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

std::vector<double> get_range_param(rclcpp::Node& node, const std::string& name, const std::vector<double>& fallback)
{
  std::vector<double> values = node.declare_parameter<std::vector<double>>(name, fallback);
  if (values.size() != 2 || values[0] > values[1])
  {
    RCLCPP_WARN(node.get_logger(), "Parameter '%s' is invalid. Using fallback range.", name.c_str());
    return fallback;
  }
  return values;
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
    analyze_srv_ = create_service<std_srvs::srv::Trigger>(
        "analyze", std::bind(&WorkspaceQualityAnalyzer::analyzeService, this, std::placeholders::_1, std::placeholders::_2));
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
    std::vector<double> joint_values;
  };

  void loadParameters()
  {
    move_group_name_ = declare_parameter<std::string>("move_group_name", "arm");
    ee_link_ = declare_parameter<std::string>("ee_link", "");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    robot_description_param_ = declare_parameter<std::string>("robot_description_param", "robot_description");

    sample_shape_ = declare_parameter<std::string>("sample_shape", "box");
    x_range_ = get_range_param(*this, "x_range", { -0.25, 0.25 });
    y_range_ = get_range_param(*this, "y_range", { -0.25, 0.25 });
    z_range_ = get_range_param(*this, "z_range", { -0.15, 0.25 });
    resolution_ = declare_parameter<double>("resolution", 0.05);

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
    marker_scale_ = declare_parameter<double>("marker_scale", 0.025);

    if (resolution_ <= 0.0)
    {
      RCLCPP_WARN(get_logger(), "resolution must be positive. Using 0.05.");
      resolution_ = 0.05;
    }
  }

  void analyzeService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    try
    {
      const auto results = analyze();
      const auto csv_path = writeCsv(results);
      publishResults(results);

      const auto best_it = std::max_element(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score < rhs.score;
      });

      const int good_count = static_cast<int>(std::count_if(results.begin(), results.end(), [](const auto& result) {
        return result.risk_level == RiskLevel::GOOD;
      }));

      std::ostringstream message;
      message << "samples=" << results.size() << ", good=" << good_count << ", csv=" << csv_path;
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

  std::vector<CandidateResult> analyze()
  {
    if (!planning_scene_monitor_->getStateMonitor()->waitForCurrentState(now(), state_wait_timeout_))
    {
      RCLCPP_WARN(get_logger(), "Timed out waiting for current robot state. Continuing with latest available state.");
    }

    planning_scene_monitor::LockedPlanningSceneRO scene(planning_scene_monitor_);
    moveit::core::RobotState current_state = scene->getCurrentState();
    current_state.update();

    const Eigen::Isometry3d current_ee_pose = current_state.getGlobalLinkTransform(ee_link_model_);
    const Eigen::Quaterniond current_orientation(current_ee_pose.rotation());
    std::vector<double> current_joint_values;
    current_state.copyJointGroupPositions(joint_model_group_, current_joint_values);

    std::vector<CandidateResult> results;
    int index = 0;
    for (double dx = x_range_[0]; dx <= x_range_[1] + kEpsilon; dx += resolution_)
    {
      for (double dy = y_range_[0]; dy <= y_range_[1] + kEpsilon; dy += resolution_)
      {
        for (double dz = z_range_[0]; dz <= z_range_[1] + kEpsilon; dz += resolution_)
        {
          Eigen::Isometry3d target_pose = current_ee_pose;
          target_pose.translation() = current_ee_pose.translation() + Eigen::Vector3d(dx, dy, dz);

          CandidateResult result;
          result.index = index++;
          result.pose.position.x = target_pose.translation().x();
          result.pose.position.y = target_pose.translation().y();
          result.pose.position.z = target_pose.translation().z();
          result.pose.orientation.x = current_orientation.x();
          result.pose.orientation.y = current_orientation.y();
          result.pose.orientation.z = current_orientation.z();
          result.pose.orientation.w = current_orientation.w();

          evaluateCandidate(current_state, current_joint_values, target_pose, result, *scene);
          results.push_back(result);
        }
      }
    }

    RCLCPP_INFO(get_logger(), "Finished workspace analysis with %zu samples.", results.size());
    return results;
  }

  void evaluateCandidate(const moveit::core::RobotState& current_state, const std::vector<double>& current_joint_values,
                         const Eigen::Isometry3d& target_pose, CandidateResult& result,
                         const planning_scene::PlanningScene& scene) const
  {
    moveit::core::RobotState candidate_state(current_state);

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
    scene.checkCollision(collision_request, collision_result, candidate_state);
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
           "sigma_min,condition_number,manipulability,joint_margin_min,q_distance,ik_time_ms,joint_values\n";

    for (const auto& result : results)
    {
      csv << result.index << "," << result.pose.position.x << "," << result.pose.position.y << ","
          << result.pose.position.z << "," << result.pose.orientation.x << "," << result.pose.orientation.y << ","
          << result.pose.orientation.z << "," << result.pose.orientation.w << "," << result.ik_success << ","
          << result.collision << "," << result.score << "," << riskLevelToString(result.risk_level) << ","
          << result.risk_reason << "," << result.sigma_min << "," << result.condition_number << ","
          << result.manipulability << "," << result.joint_margin_min << "," << result.q_distance << ","
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

  void publishResults(const std::vector<CandidateResult>& results) const
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
      marker.scale.x = marker_scale_;
      marker.scale.y = marker_scale_;
      marker.scale.z = marker_scale_;
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
    best_marker.scale.x = marker_scale_ * 2.2;
    best_marker.scale.y = marker_scale_ * 2.2;
    best_marker.scale.z = marker_scale_ * 2.2;
    best_marker.color = color(1.0, 1.0, 1.0, 1.0);
    markers.markers.push_back(best_marker);

    marker_pub_->publish(markers);
  }

  std_msgs::msg::ColorRGBA colorForResult(const CandidateResult& result) const
  {
    switch (result.risk_level)
    {
      case RiskLevel::GOOD:
        return color(0.0, 0.85, 0.25, 0.85);
      case RiskLevel::WARNING:
        return color(1.0, 0.85, 0.0, 0.85);
      case RiskLevel::BAD:
        return color(1.0, 0.05, 0.0, 0.85);
      case RiskLevel::UNREACHABLE:
        return color(0.45, 0.45, 0.45, 0.35);
      case RiskLevel::COLLISION:
        return color(0.65, 0.0, 0.85, 0.85);
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
  std::string sample_shape_;
  std::vector<double> x_range_;
  std::vector<double> y_range_;
  std::vector<double> z_range_;
  double resolution_{ 0.05 };
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
  double marker_scale_{ 0.025 };

  robot_model_loader::RobotModelLoaderPtr robot_model_loader_;
  moveit::core::RobotModelPtr robot_model_;
  const moveit::core::JointModelGroup* joint_model_group_{ nullptr };
  const moveit::core::LinkModel* ee_link_model_{ nullptr };
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr best_pose_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr analyze_srv_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<WorkspaceQualityAnalyzer>();
    node->initializeMoveIt();
    rclcpp::spin(node);
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
