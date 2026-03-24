#include "robot_hardware/robot_hardware_interface.hpp"
#include <unordered_set>
#include <algorithm>

#include <stdexcept>
#include <thread>
#include <chrono>

namespace robot_hardware
{
HumanoidRobotHardware::HumanoidRobotHardware()
    : device_type_(0),
      device_index_(0),
      abit_baud_(1000000),
      dbit_baud_(5000000),
      n_joints_(0)
{}

HumanoidRobotHardware::~HumanoidRobotHardware()
{
    if (mgr_)
        mgr_->shutdown();
}

std::string HumanoidRobotHardware::getParam(const std::string &key,
                                     const std::string &default_val) const
{
    auto it = hw_info_.hardware_parameters.find(key);
    if (it == hw_info_.hardware_parameters.end())
        return default_val;
    return it->second;
}

int HumanoidRobotHardware::getParamInt(const std::string &key, int default_val) const
{
    const std::string val = getParam(key);
    if (val.empty())
        return default_val;
    try { return std::stoi(val); }
    catch (...) { return default_val; }
}

// ============================================================
//  on_init
//  解析 URDF hardware_parameters 和各关节参数，建立 MotorConfig 列表。
//
//  URDF 硬件参数示例：
//    <hardware>
//      <plugin>z1_robot_hardware/HumanoidRobotHardware</plugin>
//      <param name="lib_path">/path/to/libcontrolcanfd.so</param>
//      <param name="device_type">1</param>
//      <param name="device_index">0</param>
//      <param name="abit_baud">1000000</param>
//      <param name="dbit_baud">5000000</param>
//    </hardware>
//
//  关节参数示例（每个 <joint> 下的 <param>）：
//    <joint name="joint_0">
//      <param name="dev_id">1</param>
//      <param name="channel">0</param>
//    </joint>
// ============================================================
hardware_interface::CallbackReturn
HumanoidRobotHardware::on_init(const hardware_interface::HardwareInfo &info)
{
    // 调用父类 on_init，会将 info 存入 info_ 成员（ros2_control 规范）
    if (hardware_interface::SystemInterface::on_init(info) !=
        hardware_interface::CallbackReturn::SUCCESS)
    {
        return hardware_interface::CallbackReturn::ERROR;
    }

    hw_info_  = info;
    n_joints_ = static_cast<int>(info.joints.size());

    // ── 读取硬件级参数 ──────────────────────────────────────────────────────
    lib_path_     = getParam("lib_path");
    device_type_  = getParamInt("device_type",  USBCANFD_200U);        // USBCANFD_200U = 41
    device_index_ = getParamInt("device_index", 0);
    abit_baud_    = getParamInt("abit_baud",    1000000);
    dbit_baud_    = getParamInt("dbit_baud",    5000000);

    if (lib_path_.empty())
    {
        RCLCPP_ERROR(rclcpp::get_logger("HumanoidRobotHardware"),
                     "Parameter 'lib_path' is required but not set in URDF.");
        return hardware_interface::CallbackReturn::ERROR;
    }

    RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"),
                "on_init: %d joints, lib=%s dev_type=%d dev_idx=%d",
                n_joints_, lib_path_.c_str(), device_type_, device_index_);

    // ── 校验每个关节的接口声明 ──────────────────────────────────────────────
    //    实体关节（有 dev_id/channel）：position state + velocity state + position cmd
    //    虚拟关节（无 dev_id/channel，如夹爪）：仅要求 position state + position cmd，
    //    velocity state 为可选（URDF 可能不声明）。
    for (const auto &joint : info.joints)
    {
        bool has_pos_state = false;
        bool has_vel_state = false;
        bool has_pos_cmd   = false;

        for (const auto &si : joint.state_interfaces)
        {
            if (si.name == hardware_interface::HW_IF_POSITION) has_pos_state = true;
            if (si.name == hardware_interface::HW_IF_VELOCITY)  has_vel_state = true;
        }

        for (const auto &ci : joint.command_interfaces)
            if (ci.name == hardware_interface::HW_IF_POSITION) has_pos_cmd = true;

        if (!has_pos_state)
        {
            RCLCPP_ERROR(rclcpp::get_logger("HumanoidRobotHardware"),
                         "Joint '%s' is missing state interface 'position'.",
                         joint.name.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
        if (!has_vel_state)
        {
            RCLCPP_ERROR(rclcpp::get_logger("HumanoidRobotHardware"),
                         "Joint '%s' is missing state interface 'velocity'.",
                         joint.name.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
        if (!has_pos_cmd)
        {
            RCLCPP_ERROR(rclcpp::get_logger("HumanoidRobotHardware"),
                         "Joint '%s' is missing command interface 'position'.",
                         joint.name.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
    }

    // ── 初始化状态/命令缓冲 ────────────────────────────────────────────────
    hw_pos_.assign(n_joints_, 0.0);
    hw_vel_.assign(n_joints_, 0.0);
    hw_cmd_.assign(n_joints_, 0.0);

    clock_ = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);

    return hardware_interface::CallbackReturn::SUCCESS;
}

// ============================================================
//  on_configure
//  构造 MotorManager，注册电机，初始化 CAN，编码多轴 CSP 模板帧。
// ============================================================
hardware_interface::CallbackReturn
HumanoidRobotHardware::on_configure(const rclcpp_lifecycle::State & /*previous_state*/)
{
    RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"), "on_configure");

    // ── 构造 MotorManager ───────────────────────────────────────────────────
    mgr_ = std::make_unique<MotorManager>(device_type_, device_index_);

    // ── 按关节顺序注册电机 ──────────────────────────────────────────────────
    std::unordered_set<int> ch_set;
    devkey_to_joint_.clear();
    virtual_joints_.clear();

    for (int i = 0; i < n_joints_; i++)
    {
        MotorConfig cfg;
        if (!parseMotorConfig_(i, cfg))
        {
            virtual_joints_.insert(i);
            RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"),
                        "  joint[%d] '%s' -> virtual (passthrough)",
                        i, hw_info_.joints[i].name.c_str());
            continue;
        }

        mgr_->addMotor(cfg);
        ch_set.insert(cfg.channel);

        // 建立 (channel, dev_id) → 关节索引 映射
        devkey_to_joint_[makeDevKey_(cfg.channel, cfg.dev_id)] = i;

        RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"),
                    "  joint[%d] '%s'  dev_id=%d  ch=%d",
                    i, cfg.joint_name.c_str(), cfg.dev_id, cfg.channel);
    }

    // 收集去重后的通道列表（后续按通道初始化模板帧和发送）
    used_channels_.assign(ch_set.begin(), ch_set.end());
    std::sort(used_channels_.begin(), used_channels_.end());

    // ── 初始化 CAN 驱动 ─────────────────────────────────────────────────────
    if (!mgr_->initialize(lib_path_, abit_baud_, dbit_baud_))
    {
        RCLCPP_ERROR(rclcpp::get_logger("HumanoidRobotHardware"),
                     "MotorManager::initialize() failed.");
        return hardware_interface::CallbackReturn::ERROR;
    }

    // ── 按通道初始化多轴 CSP 模板帧 ─────────────────────────────────────────
    for (int ch : used_channels_)
    {
        if (!mgr_->initMultiAxisCSP(ch))
        {
            RCLCPP_WARN(rclcpp::get_logger("HumanoidRobotHardware"),
                        "initMultiAxisCSP(ch=%d) returned false (no motors on channel?)", ch);
        }
    }

    RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"), "on_configure done.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

// ============================================================
//  on_activate
//  使能所有电机，通过 SDO 读取各轴初始位置作为命令初值。
// ============================================================
hardware_interface::CallbackReturn
HumanoidRobotHardware::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"), "on_activate");

    // if (!mgr_->enableAll())
    // {
    //     RCLCPP_ERROR(rclcpp::get_logger("HumanoidRobotHardware"),
    //                  "enableAll() failed.");
    //     return hardware_interface::CallbackReturn::ERROR;
    // }

    // 等待驱动器完成状态机切换
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ── 初始位置读取 ─────────────────────────────────────────────────────────
    //    实体关节：SDO 读 0x6064，真实位置覆盖 hw_pos_ 和 hw_cmd_
    //              URDF 中的 initial_value 对实体关节无实际意义，可不填
    //    虚拟关节：跳过 SDO，hw_cmd_ 保持 ros2_control 框架写入的 initial_value，
    //              hw_pos_ 直接透传
    int motor_idx = 0;
    for (int k = 0; k < n_joints_; k++)
    {
        if (virtual_joints_.count(k))
        {
            hw_pos_[k] = hw_cmd_[k];   // 透传 initial_value
            RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"),
                        "  joint[%d] '%s' virtual, init_pos=%.4f rad",
                        k, hw_info_.joints[k].name.c_str(), hw_pos_[k]);
            continue;
        }

        SDOResponse resp;
        if (mgr_->sendSDOReadAndWait(motor_idx, 0x6064, resp, 500))
        {
            hw_pos_[k] = static_cast<double>(resp.position_rad);
            hw_cmd_[k] = static_cast<double>(resp.position_rad);
            RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"),
                        "  joint[%d] '%s' init_pos=%.3f deg (%.4f rad)",
                        k, hw_info_.joints[k].name.c_str(),
                        resp.position_deg, resp.position_rad);
        }
        else
        {
            RCLCPP_WARN(rclcpp::get_logger("HumanoidRobotHardware"),
                        "  joint[%d] SDO read 0x6064 timeout, using pos=0", k);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        motor_idx++;
    }

    // 以初始位置为目标，发送一帧 CSP 确保电机持仓不动
    // hw_cmd_[k] 顺序与 addMotor 注册顺序一致（全局索引 k），
    // sendMultiAxisCSP 内部通过 slot_to_global 自动取本通道各轴的值
    for (int ch : used_channels_)
        mgr_->sendMultiAxisCSP(ch, hw_cmd_.data());

    RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"), "on_activate done.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

// ============================================================
//  on_deactivate
//  下使能所有电机，停止输出力矩。
// ============================================================
hardware_interface::CallbackReturn
HumanoidRobotHardware::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"), "on_deactivate");

    if (mgr_)
    {
        mgr_->disableAll();
    }

    RCLCPP_INFO(rclcpp::get_logger("HumanoidRobotHardware"), "on_deactivate done.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

// ============================================================
//  export_state_interfaces
//  导出 position 和 velocity 状态接口（共享 hw_pos_ / hw_vel_ 内存）
// ============================================================
std::vector<hardware_interface::StateInterface>
HumanoidRobotHardware::export_state_interfaces()
{
    std::vector<hardware_interface::StateInterface> ifaces;
    ifaces.reserve(n_joints_ * 2);

    for (int i = 0; i < n_joints_; i++)
    {
        const std::string &name = hw_info_.joints[i].name;
        ifaces.emplace_back(name, hardware_interface::HW_IF_POSITION, &hw_pos_[i]);
        ifaces.emplace_back(name, hardware_interface::HW_IF_VELOCITY, &hw_vel_[i]);
    }

    return ifaces;
}

// ============================================================
//  export_command_interfaces
//  导出 position 命令接口（共享 hw_cmd_ 内存）
// ============================================================
std::vector<hardware_interface::CommandInterface>
HumanoidRobotHardware::export_command_interfaces()
{
    std::vector<hardware_interface::CommandInterface> ifaces;
    ifaces.reserve(n_joints_);

    for (int i = 0; i < n_joints_; i++)
    {
        const std::string &name = hw_info_.joints[i].name;
        ifaces.emplace_back(name, hardware_interface::HW_IF_POSITION, &hw_cmd_[i]);
    }

    return ifaces;
}

// ============================================================
//  read()
//  从每条 CAN 通道的 PDO 接收队列中排空所有反馈帧，
//  取每个电机的「最新」反馈更新 hw_pos_ / hw_vel_。
//  采用「后覆盖」策略：循环弹帧，后收到的帧覆盖先收到的帧。
// ============================================================
hardware_interface::return_type
HumanoidRobotHardware::read(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
    for (int ch : used_channels_)
        drainFeedback(ch);

    // 虚拟关节无 PDO 反馈，直接将命令值透传为状态值
    for (int k : virtual_joints_)
        hw_pos_[k] = hw_cmd_[k];

    return hardware_interface::return_type::OK;
}

// ============================================================
//  parseMotorConfig_
//  从指定关节的 HardwareInfo 解析 MotorConfig。
//  dev_id 和 channel 为必填参数，缺少任意一个返回 false。
// ============================================================
bool HumanoidRobotHardware::parseMotorConfig_(int joint_idx, MotorConfig &cfg_out) const
{
    const auto &joint = hw_info_.joints[joint_idx];

    bool has_dev_id  = false;
    bool has_channel = false;

    cfg_out.joint_name   = joint.name;


    for (const auto &p : joint.parameters)
    {
        if (p.first == "dev_id")
        {
            cfg_out.dev_id = std::stoi(p.second);
            has_dev_id     = true;
        }
        else if (p.first == "channel")
        {
            cfg_out.channel = std::stoi(p.second);
            has_channel     = true;
        }
    }

    if (!has_dev_id || !has_channel)
    {
        return false;
    }
    return true;
}

// ── drainFeedback 辅助实现 ────────────────────────────────────────────────
//
//  非阻塞：popPDOFeedback 内部调用 RingBuffer::pop，
//  队列为空时立即返回 false，while 循环随即退出，不会阻塞实时线程。
//
//  「后覆盖」策略：同一电机在一个控制周期内可能累积多帧，
//  循环持续弹出并覆盖，保证 hw_pos_ / hw_vel_ 始终为最新值。
// ─────────────────────────────────────────────────────────────────────────
void HumanoidRobotHardware::drainFeedback(int channel)
{
    MotorFeedback fb;
    while (mgr_->popPDOFeedback(channel, fb))
    {
        // 通过 devkey_to_joint_ 映射找到对应关节索引
        auto it = devkey_to_joint_.find(makeDevKey_(channel, fb.dev_id));
        if (it == devkey_to_joint_.end())
            continue;   // 未注册的 dev_id，忽略

        const int idx = it->second;

        // 更新状态缓冲（弧度制，与 ros2_control 约定一致）
        hw_pos_[idx] = static_cast<double>(fb.position_rad);
        hw_vel_[idx] = static_cast<double>(fb.velocity_rad_s);
    }
}

// ============================================================
//  write()
//  hw_cmd_[k] 按全局关节索引排列，与 addMotor 注册顺序一致。
//  hw_cmd_（rad，double[]）直接传入 sendMultiAxisCSP，
//  每条通道通过 slot_to_global 各自取属于自己的轴，只修改 target1 字段（2 字节）。
//  sendMultiAxisCSP 内部通过 slot_to_global 映射自动取出本通道各轴目标值，
// ============================================================
hardware_interface::return_type
HumanoidRobotHardware::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
    for (int ch : used_channels_)
    {
        if (!mgr_->sendMultiAxisCSP(ch, hw_cmd_.data()))
        {
            RCLCPP_ERROR_THROTTLE(rclcpp::get_logger("HumanoidRobotHardware"),
                                  *clock_, 1000,
                                  "sendMultiAxisCSP(ch=%d) failed", ch);
            return hardware_interface::return_type::ERROR;
        }
    }

    return hardware_interface::return_type::OK;
}

}  // namespace robot_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
    robot_hardware::HumanoidRobotHardware,
    hardware_interface::SystemInterface)