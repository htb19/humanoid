#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstdint>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "robot_hardware/motor_manager.hpp"
#include "robot_hardware/motor_protocol.hpp"

// ============================================================
//  HumanoidRobotHardware
//
//  ros2_control SystemInterface 实现，对接 MotorManager 多轴 CSP
//  广播控制（64 字节 CANFD 帧，CAN ID = 0x200）。
//
//  关节顺序与 URDF <joint> 声明顺序一致，每个关节对应一个电机，
//  通道分配由 URDF <param> 中的 "channel" 字段决定。
//
//  状态接口：position（rad）、velocity（rad/s）
//  命令接口：position（rad）
//
//  生命周期：
//    on_init      → 解析 URDF 参数，构造 MotorConfig
//    on_configure → MotorManager::initialize + initMultiAxisCSP（按通道）
//    on_activate  → enableAll + SDO 读各轴初始位置
//    on_deactivate→ disableAll
//    ~析构        → shutdown（由 on_deactivate 后 ros2_control 框架触发）
//
//  read()  → popPDOFeedback（非阻塞，排空队列取最新帧）
//  write() → sendMultiAxisCSP（按通道，只修改 target1 字段）
// ============================================================

namespace robot_hardware
{
class HumanoidRobotHardware : public hardware_interface::SystemInterface
{
public:
    HumanoidRobotHardware();
    ~HumanoidRobotHardware() override;

    // ── 生命周期回调 ─────────────────────────────────────────────────────────
    hardware_interface::CallbackReturn
    on_init(const hardware_interface::HardwareInfo &info) override;

    hardware_interface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &previous_state) override;

    hardware_interface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &previous_state) override;

    hardware_interface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

    // ── 接口导出 ─────────────────────────────────────────────────────────────
    std::vector<hardware_interface::StateInterface>
    export_state_interfaces() override;

    std::vector<hardware_interface::CommandInterface>
    export_command_interfaces() override;

    // ── 实时读写 ─────────────────────────────────────────────────────────────
    hardware_interface::return_type
    read(const rclcpp::Time &time, const rclcpp::Duration &period) override;

    hardware_interface::return_type
    write(const rclcpp::Time &time, const rclcpp::Duration &period) override;

private:
    std::string getParam(const std::string &key,
                         const std::string &default_val = "") const;
    int getParamInt(const std::string &key, int default_val = 0) const;

    void drainFeedback(int channel);
    bool parseMotorConfig_(int joint_idx, MotorConfig &cfg_out) const;

    static int makeDevKey_(int channel, int dev_id)
    {
        return (channel << 8) | dev_id;
    }

private:
    std::unique_ptr<MotorManager> mgr_;   ///< 电机管理器（configure 时构造）

    // ── 硬件参数（从 URDF <hardware><param> 读取）────────────────────────────
    std::string lib_path_;        ///< libcontrolcanfd.so 绝对路径
    int         device_type_;     ///< USBCANFD_200U 等设备类型
    int         device_index_;    ///< 设备索引（同类型第 N 个设备）
    int         abit_baud_;       ///< 仲裁段波特率（默认 1000000）
    int         dbit_baud_;       ///< 数据段波特率（默认 5000000）

    // ── 关节状态与指令缓冲（与 StateInterface / CommandInterface 共享内存）──
    int                  n_joints_;      ///< 关节总数
    std::vector<double>  hw_pos_;        ///< 当前关节位置（rad）
    std::vector<double>  hw_vel_;        ///< 当前关节速度（rad/s）
    std::vector<double>  hw_cmd_;        ///< 目标关节位置（rad）

    // ── dev_id → 关节索引 映射（用于 PDO 反馈归位）──────────────────────────
    std::unordered_map<int, int> devkey_to_joint_;
    // ── 虚拟关节索引集合 ──────────────────────────────────────────────────────
    std::unordered_set<int> virtual_joints_;
    // ── 已使用的 CAN 通道列表（去重，configure 时确定）───────────────────────
    std::vector<int> used_channels_;
    bool hardware_fault_ = false;
    int fault_joint_index_ = -1;
    uint16_t fault_error_code_ = 0;
    // ── 硬件信息（on_init 时保存，用于 getParam）───────────────────────────
    hardware_interface::HardwareInfo hw_info_;

    rclcpp::Clock::SharedPtr clock_;
};

}  // namespace robot_hardware
