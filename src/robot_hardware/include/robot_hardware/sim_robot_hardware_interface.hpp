#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace robot_hardware
{
class SimHumanoidHardware : public hardware_interface::SystemInterface
{
public:
    SimHumanoidHardware() = default;
    ~SimHumanoidHardware() override = default;

    hardware_interface::CallbackReturn
    on_init(const hardware_interface::HardwareInfo &info) override;

    hardware_interface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &previous_state) override;

    hardware_interface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &previous_state) override;

    hardware_interface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

    std::vector<hardware_interface::StateInterface>
    export_state_interfaces() override;

    std::vector<hardware_interface::CommandInterface>
    export_command_interfaces() override;

    hardware_interface::return_type
    read(const rclcpp::Time &time, const rclcpp::Duration &period) override;

    hardware_interface::return_type
    write(const rclcpp::Time &time, const rclcpp::Duration &period) override;

private:
    struct SimMotorState
    {
        int32_t actual_count = 0;
        int32_t target_count = 0;
        int32_t prev_target_count = 0;
        double actual_rad = 0.0;
        double prev_actual_rad = 0.0;
        double velocity_rad_s = 0.0;
        uint16_t error_code = 0;
        bool has_error = false;
    };

    std::string getParam(
        const std::string &key,
        const std::string &default_val = "") const;
    int getParamInt(const std::string &key, int default_val = 0) const;
    double getParamDouble(const std::string &key, double default_val = 0.0) const;
    bool getParamBool(const std::string &key, bool default_val = false) const;

    bool validateJointInterfaces_() const;
    double parseInitialPositionRad_(int joint_idx) const;

    static int32_t positionRadToCount_(double rad);
    static double positionCountToRad_(int32_t count);
    static int32_t signum_(int32_t value);

private:
    hardware_interface::HardwareInfo hw_info_;

    int n_joints_ = 0;
    std::vector<double> hw_pos_;
    std::vector<double> hw_vel_;
    std::vector<double> hw_cmd_;
    std::vector<SimMotorState> sim_motors_;

    bool simulate_following_ = false;
    double position_lag_alpha_ = 1.0;
    int max_follow_count_per_cycle_ = 100;
    bool enable_fault_on_large_delta_ = true;
    int max_delta_count_per_write_ = 100;
    bool return_error_on_fault_ = false;
    bool log_state_transitions_ = false;

    bool hardware_fault_ = false;
    rclcpp::Clock::SharedPtr clock_;
};
}  // namespace robot_hardware
