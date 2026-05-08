#include "robot_hardware/sim_robot_hardware_interface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pluginlib/class_list_macros.hpp"
#include "robot_hardware/motor_protocol.hpp"

namespace robot_hardware
{
namespace
{
constexpr double kMinPeriodSec = 1e-6;
}  // namespace

hardware_interface::CallbackReturn
SimHumanoidHardware::on_init(const hardware_interface::HardwareInfo &info)
{
    if (hardware_interface::SystemInterface::on_init(info) !=
        hardware_interface::CallbackReturn::SUCCESS)
    {
        return hardware_interface::CallbackReturn::ERROR;
    }

    hw_info_ = info;
    n_joints_ = static_cast<int>(info.joints.size());

    if (!validateJointInterfaces_())
        return hardware_interface::CallbackReturn::ERROR;

    simulate_following_ = getParamBool("simulate_following", false);
    position_lag_alpha_ = getParamDouble("position_lag_alpha", 1.0);
    max_follow_count_per_cycle_ = getParamInt("max_follow_count_per_cycle", 100);
    enable_trajectory_shaping_ = getParamBool("enable_trajectory_shaping", true);
    max_vel_cnt_per_cycle_    = getParamInt("max_vel_cnt_per_cycle", 100);
    warning_cnt_threshold_    = getParamInt("warning_cnt_threshold", 50);
    dangerous_cnt_threshold_  = getParamInt("dangerous_cnt_threshold", 70);
    emergency_cnt_threshold_  = getParamInt("emergency_cnt_threshold", 90);

    if (position_lag_alpha_ < 0.0)
        position_lag_alpha_ = 0.0;
    if (position_lag_alpha_ > 1.0)
        position_lag_alpha_ = 1.0;
    if (max_follow_count_per_cycle_ < 1)
        max_follow_count_per_cycle_ = 1;
    if (max_vel_cnt_per_cycle_ < 1)
        max_vel_cnt_per_cycle_ = 1;

    hw_pos_.assign(n_joints_, 0.0);
    hw_vel_.assign(n_joints_, 0.0);
    hw_cmd_.assign(n_joints_, 0.0);
    last_hw_cmd_.assign(n_joints_, 0.0);
    sim_motors_.assign(n_joints_, SimMotorState{});

    clock_ = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);

    RCLCPP_INFO(
        rclcpp::get_logger("SimHumanoidHardware"),
        "on_init: %d joints simulate_following=%s lag_alpha=%.3f "
        "max_follow=%d shaping=%s max_vel=%d cnt",
        n_joints_,
        simulate_following_ ? "true" : "false",
        position_lag_alpha_,
        max_follow_count_per_cycle_,
        enable_trajectory_shaping_ ? "true" : "false",
        max_vel_cnt_per_cycle_);

    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SimHumanoidHardware::on_configure(const rclcpp_lifecycle::State & /*previous_state*/)
{
    RCLCPP_INFO(rclcpp::get_logger("SimHumanoidHardware"), "on_configure");

    hardware_fault_ = false;

    for (int i = 0; i < n_joints_; ++i)
    {
        const double initial_rad = parseInitialPositionRad_(i);
        const int32_t initial_count = positionRadToCount_(initial_rad);

        SimMotorState state;
        state.actual_count = initial_count;
        state.target_count = initial_count;
        state.actual_rad = positionCountToRad_(initial_count);
        state.prev_actual_rad = state.actual_rad;
        sim_motors_[i] = state;

        hw_pos_[i] = state.actual_rad;
        hw_cmd_[i] = state.actual_rad;
        hw_vel_[i] = 0.0;
    }

    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SimHumanoidHardware::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    RCLCPP_INFO(rclcpp::get_logger("SimHumanoidHardware"), "on_activate");

    hardware_fault_ = false;
    for (int i = 0; i < n_joints_; ++i)
    {
        SimMotorState &state = sim_motors_[i];
        state.actual_count = positionRadToCount_(hw_cmd_[i]);
        state.target_count = state.actual_count;
        state.actual_rad = positionCountToRad_(state.actual_count);
        state.prev_actual_rad = state.actual_rad;
        state.velocity_rad_s = 0.0;
        hw_pos_[i] = state.actual_rad;
        hw_vel_[i] = 0.0;
        last_hw_cmd_[i] = hw_cmd_[i];
    }

    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SimHumanoidHardware::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    RCLCPP_INFO(rclcpp::get_logger("SimHumanoidHardware"), "on_deactivate");
    return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
SimHumanoidHardware::export_state_interfaces()
{
    std::vector<hardware_interface::StateInterface> ifaces;
    ifaces.reserve(n_joints_ * 2);

    for (int i = 0; i < n_joints_; ++i)
    {
        const std::string &name = hw_info_.joints[i].name;
        ifaces.emplace_back(name, hardware_interface::HW_IF_POSITION, &hw_pos_[i]);
        ifaces.emplace_back(name, hardware_interface::HW_IF_VELOCITY, &hw_vel_[i]);
    }

    return ifaces;
}

std::vector<hardware_interface::CommandInterface>
SimHumanoidHardware::export_command_interfaces()
{
    std::vector<hardware_interface::CommandInterface> ifaces;
    ifaces.reserve(n_joints_);

    for (int i = 0; i < n_joints_; ++i)
    {
        const std::string &name = hw_info_.joints[i].name;
        ifaces.emplace_back(name, hardware_interface::HW_IF_POSITION, &hw_cmd_[i]);
    }

    return ifaces;
}

hardware_interface::return_type
SimHumanoidHardware::read(const rclcpp::Time & /*time*/, const rclcpp::Duration &period)
{
    const double period_sec = std::max(period.seconds(), kMinPeriodSec);

    for (int i = 0; i < n_joints_; ++i)
    {
        SimMotorState &state = sim_motors_[i];
        state.prev_actual_rad = state.actual_rad;

        if (simulate_following_)
        {
            const int32_t error_count = state.target_count - state.actual_count;
            int32_t step = static_cast<int32_t>(
                std::lround(static_cast<double>(error_count) * position_lag_alpha_));

            if (step == 0 && error_count != 0)
                step = signum_(error_count);

            if (step > max_follow_count_per_cycle_)
                step = max_follow_count_per_cycle_;
            if (step < -max_follow_count_per_cycle_)
                step = -max_follow_count_per_cycle_;

            state.actual_count += step;
        }
        else
        {
            state.actual_count = state.target_count;
        }

        state.actual_rad = positionCountToRad_(state.actual_count);
        state.velocity_rad_s = (state.actual_rad - state.prev_actual_rad) / period_sec;

        hw_pos_[i] = state.actual_rad;
        hw_vel_[i] = state.velocity_rad_s;

        const int32_t error_cnt =
            std::abs(state.target_count - state.actual_count);
        if (error_cnt >= emergency_cnt_threshold_)
            RCLCPP_ERROR_THROTTLE(
                rclcpp::get_logger("SimHumanoidHardware"),
                *clock_, 2000,
                "EMERGENCY: joint='%s' target=%d actual=%d error=%d cnt",
                hw_info_.joints[i].name.c_str(),
                state.target_count, state.actual_count, error_cnt);
        else if (error_cnt >= dangerous_cnt_threshold_)
            RCLCPP_WARN_THROTTLE(
                rclcpp::get_logger("SimHumanoidHardware"),
                *clock_, 1000,
                "DANGEROUS: joint='%s' target=%d actual=%d error=%d cnt",
                hw_info_.joints[i].name.c_str(),
                state.target_count, state.actual_count, error_cnt);
        else if (error_cnt >= warning_cnt_threshold_)
            RCLCPP_WARN_THROTTLE(
                rclcpp::get_logger("SimHumanoidHardware"),
                *clock_, 500,
                "WARNING: joint='%s' target=%d actual=%d error=%d cnt",
                hw_info_.joints[i].name.c_str(),
                state.target_count, state.actual_count, error_cnt);
    }

    if (hardware_fault_)
        return hardware_interface::return_type::ERROR;

    return hardware_interface::return_type::OK;
}

hardware_interface::return_type
SimHumanoidHardware::write(const rclcpp::Time &, const rclcpp::Duration &)
{
    for (int i = 0; i < n_joints_; ++i)
    {
        if (enable_trajectory_shaping_)
        {
            const double delta = hw_cmd_[i] - last_hw_cmd_[i];
            const int32_t delta_cnt = positionRadToCount_(delta);
            if (delta_cnt < -max_vel_cnt_per_cycle_)
                hw_cmd_[i] = last_hw_cmd_[i] - positionCountToRad_(max_vel_cnt_per_cycle_);
            else if (delta_cnt > max_vel_cnt_per_cycle_)
                hw_cmd_[i] = last_hw_cmd_[i] + positionCountToRad_(max_vel_cnt_per_cycle_);
        }
        sim_motors_[i].target_count = positionRadToCount_(hw_cmd_[i]);
        last_hw_cmd_[i] = hw_cmd_[i];
    }

    if (hardware_fault_)
        return hardware_interface::return_type::ERROR;
    return hardware_interface::return_type::OK;
}

std::string SimHumanoidHardware::getParam(
    const std::string &key,
    const std::string &default_val) const
{
    const auto it = hw_info_.hardware_parameters.find(key);
    if (it == hw_info_.hardware_parameters.end())
        return default_val;
    return it->second;
}

int SimHumanoidHardware::getParamInt(const std::string &key, int default_val) const
{
    const std::string value = getParam(key);
    if (value.empty())
        return default_val;

    try
    {
        return std::stoi(value);
    }
    catch (...)
    {
        return default_val;
    }
}

double SimHumanoidHardware::getParamDouble(
    const std::string &key,
    double default_val) const
{
    const std::string value = getParam(key);
    if (value.empty())
        return default_val;

    try
    {
        return std::stod(value);
    }
    catch (...)
    {
        return default_val;
    }
}

bool SimHumanoidHardware::getParamBool(const std::string &key, bool default_val) const
{
    const std::string value = getParam(key);
    if (value.empty())
        return default_val;

    if (value == "true" || value == "1" || value == "True" || value == "TRUE")
        return true;
    if (value == "false" || value == "0" || value == "False" || value == "FALSE")
        return false;
    return default_val;
}

bool SimHumanoidHardware::validateJointInterfaces_() const
{
    for (const auto &joint : hw_info_.joints)
    {
        bool has_pos_state = false;
        bool has_vel_state = false;
        bool has_pos_cmd = false;

        for (const auto &si : joint.state_interfaces)
        {
            if (si.name == hardware_interface::HW_IF_POSITION)
                has_pos_state = true;
            if (si.name == hardware_interface::HW_IF_VELOCITY)
                has_vel_state = true;
        }

        for (const auto &ci : joint.command_interfaces)
        {
            if (ci.name == hardware_interface::HW_IF_POSITION)
                has_pos_cmd = true;
        }

        if (!has_pos_state || !has_vel_state || !has_pos_cmd)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("SimHumanoidHardware"),
                "Joint '%s' must expose position state, velocity state, and position command interfaces.",
                joint.name.c_str());
            return false;
        }
    }

    return true;
}

double SimHumanoidHardware::parseInitialPositionRad_(int joint_idx) const
{
    const auto &joint = hw_info_.joints[joint_idx];
    for (const auto &si : joint.state_interfaces)
    {
        if (si.name != hardware_interface::HW_IF_POSITION)
            continue;
        if (si.initial_value.empty())
            continue;

        try
        {
            return std::stod(si.initial_value);
        }
        catch (...)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("SimHumanoidHardware"),
                "Joint '%s' has invalid initial position '%s', falling back to 0.0 rad.",
                joint.name.c_str(),
                si.initial_value.c_str());
            return 0.0;
        }
    }

    return 0.0;
}

int32_t SimHumanoidHardware::positionRadToCount_(double rad)
{
    return radToCount(rad);
}

double SimHumanoidHardware::positionCountToRad_(int32_t count)
{
    return countToRad(count);
}

int32_t SimHumanoidHardware::signum_(int32_t value)
{
    if (value > 0)
        return 1;
    if (value < 0)
        return -1;
    return 0;
}

}  // namespace robot_hardware

PLUGINLIB_EXPORT_CLASS(
    robot_hardware::SimHumanoidHardware,
    hardware_interface::SystemInterface)
