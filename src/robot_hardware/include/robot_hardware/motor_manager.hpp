#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
#include <thread>

#include "canfd_driver.hpp"
#include "motor_protocol.hpp"
#include "ring_buffer.hpp"

struct MotorConfig
{
    int         dev_id       = 1;      // 电机CAN ID（1~127）
    int         channel      = 0;      // CAN通道（0或1）
    std::string joint_name;            // 对应关节名称

    // 减速比（电机端速度 / 输出端速度）
    float gear_ratio  = 1.0f;
};

class MotorManager
{
public: 
    // recv_buffer 由 HardwareInterface 持有并传入
    explicit MotorManager(int device_type, int device_index)
        : driver_(device_type, device_index),
          initialized_(false){}

    ~MotorManager()
    {
        if (initialized_)
            shutdown();
    }

    // ----- 配置 -----
    void addMotor(const MotorConfig &cfg)
    {
        motors_.push_back(cfg);
        id_to_index_[makeKey(cfg.channel, cfg.dev_id)] =
            static_cast<int>(motors_.size()) - 1;
    }

    int motorCount() const { return static_cast<int>(motors_.size()); }

    const MotorConfig &motorConfig(int idx) const 
    { 
      if (idx < 0 || idx >= static_cast<int>(motors_.size()))
        throw std::out_of_range("Motor index out of range");
      return motors_[idx]; 
    }

    int findIndex(int channel, int dev_id) const
    {
        auto it = id_to_index_.find(makeKey(channel, dev_id));
        if (it == id_to_index_.end()) return -1;
        return it->second;
    }

    // ----- 初始化（非实时，上电时调用一次）-----
    bool initialize(const std::string &lib_path,
                    int abit_baud = 1000000,
                    int dbit_baud = 5000000)
    {
        if (initialized_)
            return false;

        if (motors_.empty())
        {
            std::cerr << "[MotorManager] No motors configured\n";
            return false;
        }

        // 锁定内存页，防止实时线程换页引起抖动
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
            std::cerr << "[MotorManager] mlockall failed (non-fatal)\n";

        if (!driver_.loadLibrary(lib_path))
        {
            std::cerr << "[MotorManager] loadLibrary failed\n";
            return false;
        }

        if (!driver_.openDevice())
        {
            std::cerr << "[MotorManager] openDevice failed\n";
            return false;
        }

        // 设置 CANFD 标准 0为CANFD ISO，1为CANFD BOSCH
        if (!driver_.setCANFDStandard(0))
            std::cerr << "[MotorManager] setCANFDStandard warning\n";

        // 初始化用到的 CAN 通道
        for (int ch : usedChannels())
        {
            if (!driver_.setBaud(ch, abit_baud, dbit_baud))
            {
                std::cerr << "[MotorManager] setBaud failed ch=" << ch << "\n";
                return false;
            }
            if (!driver_.initCAN(ch))
            {
                std::cerr << "[MotorManager] initCAN failed ch=" << ch << "\n";
                return false;
            }
            if (!driver_.setFilter(ch, 0, 0x00000000, 0xFFFFFFFF))
            {
                std::cerr << "[main] setFilter failed ch=" << ch << "\n";
                return false;
            }
            if (!driver_.startCAN(ch))
            {
                std::cerr << "[MotorManager] startCAN failed ch=" << ch << "\n";
                return false;
            }
            if (!driver_.startReceiveThread(ch))    
            {
                std::cerr << "[MotorManager] receive threads failed ch=" << ch << "\n";
                return false;
            }   
        }

        initialized_ = true;
        return true;
    }

    bool shutdown()
    {
        if (!initialized_)
            return false;

        bool ok = true;
        for (int ch : usedChannels())
        {
            if (!driver_.stopReceiveThread(ch))
            {
                std::cerr << "[MotorManager] stop receive thread failed ch=" << ch << "\n";
                ok = false;
            }
            if (!driver_.resetCAN(ch))
            {
                std::cerr << "[MotorManager] resetCAN failed ch=" << ch << "\n";
                ok = false;
            }
        }

        if (!driver_.closeDevice())
        {
            std::cerr << "[MotorManager] close device failed" << "\n";
            ok = false;
        }

        munlockall();
        initialized_ = false;     
        return ok;
    }

    // =========================================================
    //  单轴 PDO 控制指令——按模式分拆的独立公共接口
    //
    //  sendCSP / sendCSV / sendCurrent / sendProfilePosition
    //  各自独立，直接编码并发送对应模式的 PDO 控制帧。
    //  底层均通过私有辅助函数 sendSingleAxisCmd_() 发送。
    // =========================================================

    // ---- CSP 周期同步位置 ----
    //  @param target_deg     目标位置（度，[-180, 180]）
    bool sendCSP(int idx, float target_deg, bool release_brake = true)
    {
        if (idx < 0 || idx >= motorCount())
        {
            std::cerr << "[MotorManager] sendCSP: idx=" << idx << " out of range\n";
            return false;
        }

        const MotorConfig &m = motors_[idx];
        SingleAxisCmd cmd;
        cmd.enable        = true;
        cmd.release_brake = release_brake;
        cmd.clear_error   = false;
        cmd.mode          = ControlMode::CSP;
        cmd.target_param1 = target_deg;
        return sendSingleAxisCmd_(m.channel, m.dev_id, cmd);
    }

    // ---- CSV 周期同步速度 ----
    //  @param target_rpm  目标速度（RPM，正值正转）
    bool sendCSV(int idx, float target_rpm, bool release_brake = true)
    {
        if (idx < 0 || idx >= motorCount())
        {
            std::cerr << "[MotorManager] sendCSV: idx=" << idx << " out of range\n";
            return false;
        }

        const MotorConfig &m = motors_[idx];
        SingleAxisCmd cmd;
        cmd.enable        = true;
        cmd.release_brake = release_brake;
        cmd.clear_error   = false;
        cmd.mode          = ControlMode::CSV;
        cmd.target_param1 = target_rpm;
        return sendSingleAxisCmd_(m.channel, m.dev_id, cmd);
    }

    // ---- 电流环 ----
    //  @param target_ma  目标 Iq 电流（mA）
    bool sendCurrent(int idx, float target_ma, bool release_brake = true)
    {
        if (idx < 0 || idx >= motorCount())
        {
            std::cerr << "[MotorManager] sendCurrent: idx=" << idx << " out of range\n";
            return false;
        }

        const MotorConfig &m = motors_[idx];
        SingleAxisCmd cmd;
        cmd.enable        = true;
        cmd.release_brake = release_brake;
        cmd.clear_error   = false;
        cmd.mode          = ControlMode::CurrentLoop;
        cmd.target_param1 = target_ma;
        return sendSingleAxisCmd_(m.channel, m.dev_id, cmd);
    }

    // ---- 轮廓位置模式（含速度/加速度规划）----
    //  @param profile_vel_rpm  轮廓速度（RPM）
    //  @param accel_rpms       加减速（RPM/s）
    //  需要连续发送才能其作用
    bool sendProfilePosition(int idx, float target_deg, float profile_vel_rpm = 100.0f, 
                                float accel_rpms = 500.0f, bool release_brake = true)
    {
        if (idx < 0 || idx >= motorCount())
        {
            std::cerr << "[MotorManager] sendProfilePosition: idx=" << idx << " out of range\n";
            return false;
        }

        const MotorConfig &m = motors_[idx];
        SingleAxisCmd cmd;
        cmd.enable               = true;
        cmd.release_brake        = release_brake;
        cmd.clear_error          = false;
        cmd.mode                 = ControlMode::ProfilePosition;
        cmd.target_param1        = target_deg;
        cmd.profile_velocity_rpm = profile_vel_rpm;
        cmd.accel_rpms           = accel_rpms;
        return sendSingleAxisCmd_(m.channel, m.dev_id, cmd);
    }

    // =========================================================
    //  多轴 CSP 模板帧控制（按通道分组）
    //
    //  设计思路：
    //    - 系统最多 2 条 CAN 通道，每条通道独立维护一个 64 字节模板帧
    //      和一张「槽位→全局电机索引」映射表。
    //    - initMultiAxisCSP(channel) 在初始化阶段对每条通道调用一次，
    //      筛选该通道下所有电机，用 MultiAxisCmd + encodeMultiAxisCmd
    //      将所有静态字段一次性编码固化到模板缓冲区。
    //    - sendMultiAxisCSP(channel, targets_rad) 在实时循环中调用，
    //      每次只按槽位更新 target1（2 字节），然后整帧发送。
    //
    //  约束：
    //    - 每条通道最多 8 轴（CANFD 64 字节 / 7 字节·轴 = 8 轴整除）。
    //    - 必须先 addMotor() 完成，再调用 initMultiAxisCSP()。
    // =========================================================
    struct ChannelState
    {
        uint8_t buf[64] = {};           // 64 字节帧模板
        int     count   = 0;            // 本通道电机数（≤8）
        int     slot_to_global[8] = {}; 
    };

    // =========================================================
    // @brief  初始化多轴 CSP 模板帧（非实时，上电时调用一次）
    //  从 motors_ 中筛选 channel 下的所有电机（按注册顺序），
    //  构造 MultiAxisCmd 并通过 encodeMultiAxisCmd 编码成 64 字节模板。
    //  同时建立 slot_to_global 映射，供 sendMultiAxisCSP 按全局索引寻址。
    //  @param channel        目标 CAN 通道（0 或 1）
    //  @return true          该通道至少有 1 个电机且初始化成功
    // =========================================================
    bool initMultiAxisCSP(int channel)
    {
        // 筛选属于该通道的电机，按注册顺序收集全局索引
        std::vector<int> indices;
        for (int i = 0; i < motorCount(); i++)
            if (motors_[i].channel == channel)
                indices.push_back(i);

        if (indices.empty())
        {
            std::cerr << "[MotorManager] initMultiAxisCSP: no motors on ch" << channel << "\n";
            return false;
        }
        if (static_cast<int>(indices.size()) > 8)
        {
            std::cerr << "[MotorManager] initMultiAxisCSP: ch" << channel
                      << " has " << indices.size() << " motors, truncating to 8\n";
            indices.resize(8);
        }

        const int n = static_cast<int>(indices.size());

        // 构造 MultiAxisCmd，所有静态字段在此一次性确定
        MultiAxisCmd multi;
        multi.count = n;
        for (int slot = 0; slot < n; slot++)
        {
            const MotorConfig &m   = motors_[indices[slot]];
            SingleAxisCmd     &c   = multi.cmds[slot];
            c.dev_id               = static_cast<uint8_t>(m.dev_id);
            c.enable               = true;
            c.release_brake        = true;
            c.clear_error          = false;
            c.mode                 = ControlMode::CSP;
            c.target_param1        = 0.0f;  // 初始目标位置：0°
            c.accel_rpms           = 0.0f;  // CSP 模式不使用
            c.profile_velocity_rpm = 0.0f;
        }

        // 通过协议层统一编码，避免在 Manager 里重复字节布局知识
        ChannelState &state = states_[channel];
        encodeMultiAxisCmd(multi, state.buf);
        state.count = n;
        for (int slot = 0; slot < n; slot++)
            state.slot_to_global[slot] = indices[slot];

        std::cout << "[MotorManager] initMultiAxisCSP: ch" << channel
                  << " " << n << " motors ready\n";
        return true;
    }

    // ============================================================
    //  @brief  发送多轴 CSP 周期帧
    // 按 slot_to_global 映射从 targets_rad 中取出各轴目标位置，
    // 只更新各槽位的 target1 字段（buf[slot*7+1..2]，int16 大端），
    // 其余字节保持模板值，随后整帧发送。
    
    // @param channel      目标 CAN 通道（须已 initMultiAxisCSP）
    // @param targets_rad  全局目标位置数组，长度须 >= motorCount()，
    //                     下标与 addMotor() 注册顺序一致
    // @return true        发送成功
    // ============================================================    
    bool sendMultiAxisCSP(int channel, const double *targets_rad)
    {
        ChannelState &state = states_[channel];

        for (int slot = 0; slot < state.count; slot++)
        {
            const double   rad = targets_rad[state.slot_to_global[slot]];
            const int16_t cnt = static_cast<int16_t>(rad * 32768.0f / static_cast<double>(M_PI));
            const int     base = slot * 7;
            state.buf[base + 1] = static_cast<uint8_t>(cnt >> 8);
            state.buf[base + 2] = static_cast<uint8_t>(cnt & 0xFF);
        }

        return driver_.sendData(CAN_ID_PDO_MULTI, 0, 1, 0,
                                channel, state.buf, 64);
        
    }

    // ============================================================
    // @brief  发送多轴 CSP 周期帧（std::vector 重载，便于上层调用）
    // ============================================================
    bool sendMultiAxisCSP(int channel, const std::vector<double> &targets_rad)
    {
        if (static_cast<int>(targets_rad.size()) < motorCount())
        {
            std::cerr << "[MotorManager] sendMultiAxisCSP: targets size "
                      << targets_rad.size() << " < motorCount " << motorCount() << "\n";
            return false;
        }
        
        return sendMultiAxisCSP(channel, targets_rad.data());
    }

    // ============================================================
    // 发送 CANopen SYNC 同步帧。
    // ============================================================
    bool sendSync(int channel)
    {
        uint8_t empty[1] = {0};
        if (!driver_.sendData(CAN_ID_SYNC, 0, 1, 0, channel, empty, 1))
            return false;
        return true;   
    }

    // =========================================================
    //  PDO 反馈读取
    //
    //  从接收队列中弹出并解析一帧 PDO 格式的电机反馈
    //  （CAN ID = 反馈帧 ID，由 parseMotorFeedback 识别）。
    //  适用于实时控制循环中的周期性状态读取。
    //
    //  @param channel  要读取的 CAN 通道
    //  @param fb       输出：解析成功的反馈帧
    //  @return true    本次调用成功解析到一帧有效 PDO 反馈
    // =========================================================
    bool popPDOFeedback(int channel, MotorFeedback &fb)
    {
        ZCAN_ReceiveFD_Data msg;
        while (driver_.popFrame(channel, msg))
        {
            if (parseMotorFeedback(msg, channel, fb))
                return true;
        }
        return false;
    }

    // =========================================================
    //  SDO 读请求发送（
    //
    //  向指定电机发送 SDO 上传请求（读对象字典条目）。
    //  非实时调用，发送后需配合 popSDOResponse() 等待应答。
    //
    //  @param idx      电机索引（motors_ 中的下标）
    //  @param index    对象字典索引（如 0x6064）
    //  @return true    发送成功
    // =========================================================
    bool sendSDORead(int idx, uint16_t index)
    {
        if (idx < 0 || idx >= motorCount())
        {
            std::cerr << "[MotorManager] sendSDORead: idx=" << idx << " out of range\n";
            return false;
        }

        const MotorConfig &m = motors_[idx];
        uint8_t buf[8];
        uint8_t subindex = 0x00;

        encodeSDORead(index, subindex, buf);
        return driver_.sendData(CAN_ID_SDO_TX_BASE | m.dev_id,
                                0, 0, 0, m.channel, buf, 8);
    }

    // =========================================================
    //  SDO 写请求发送（
    //
    //  向指定电机发送 SDO 上传请求（写对象字典条目）。
    //  非实时调用，发送后需配合 popSDOResponse() 等待应答。
    //
    //  @param idx      电机索引（motors_ 中的下标）
    //  @param index    对象字典索引（如 0x6064）
    //  @return true    发送成功
    // =========================================================
    bool sendSDOWrite(int idx, uint8_t cmd_code, uint16_t index, uint32_t value) 
    {
        if (idx < 0 || idx >= motorCount())
        {
            std::cerr << "[MotorManager] sendSDOWrite: idx=" << idx << " out of range\n";
            return false;
        }

        const MotorConfig &m = motors_[idx];
        uint8_t buf[8];
        uint8_t subindex = 0x00;

        encodeSDOWrite(cmd_code, index, subindex, value, buf);
        return driver_.sendData(CAN_ID_SDO_TX_BASE | m.dev_id,
                                0, 0, 0, m.channel, buf, 8);
    }
    
    // =========================================================
    //  SDO 应答接收
    //
    //  从接收队列中弹出并解析一帧 SDO 响应帧
    //  （CAN ID = 0x580 | dev_id，由 parseSDOResponse 识别）。
    //  适用于非实时的参数读取、使能确认、状态查询等场景。
    //
    //  与 popPDOFeedback() 的区别：
    //    - popPDOFeedback()  解析 PDO 格式周期反馈（实时状态）
    //    - popSDOResponse() 解析 SDO 格式应答帧（非实时参数读写确认）
    //
    //  @param channel  要读取的 CAN 通道
    //  @param dev_id   期望应答的电机 dev_id（用于过滤其他电机的帧）
    //  @param resp     输出：解析成功的 SDO 响应
    //  @return true    本次调用成功解析到目标电机的 SDO 响应帧
    // =========================================================
    bool popSDOResponse(int idx, SDOResponse &resp)
    {
        if (idx < 0 || idx >= motorCount()) 
            return false;

        const MotorConfig &m = motors_[idx];

        uint32_t expect_id = CAN_ID_SDO_RX_BASE | static_cast<uint32_t>(m.dev_id);
        ZCAN_ReceiveFD_Data msg;

        if (driver_.popFrameById(m.channel, expect_id, msg))
            return parseSDOResponse(msg, resp);

        return false;
    }

    // =========================================================
    //  SDO 应答等待
    //
    //  发出 SDO 读请求后阻塞等待应答，内部轮询直至超时。
    //  适合测试代码或非实时初始化场景（如读取初始位置）。
    //
    //  @param idx          电机索引
    //  @param index        对象字典索引
    //  @param subindex     子索引
    //  @param resp         输出：SDO 响应
    //  @param timeout_ms   超时时间（毫秒，默认 500ms）
    //  @return true        在超时前收到有效应答
    // =========================================================
    bool sendSDOReadAndWait(int idx, uint16_t index, SDOResponse &resp, int timeout_ms = 500)
    {
        if (!sendSDORead(idx, index))
            return false;

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);

        while (std::chrono::steady_clock::now() < deadline)
        {
            if (popSDOResponse(idx, resp))
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 1ms 轮询间隔
        }
        return false;
    }

    // =========================================================
    //  SDO 应答等待
    //
    //  发出 SDO 读请求后阻塞等待应答，内部轮询直至超时。
    //  适合测试代码或非实时初始化场景（如读取初始位置）。
    //
    //  @param idx          电机索引
    //  @param index        对象字典索引
    //  @param subindex     子索引
    //  @param resp         输出：SDO 响应
    //  @param timeout_ms   超时时间（毫秒，默认 500ms）
    //  @return true        在超时前收到有效应答
    // =========================================================
    bool sendSDOWriteAndWait(int idx, uint8_t cmd_code, uint16_t index, uint32_t value, 
                                SDOResponse &resp, int timeout_ms = 500)
    {
        if (!sendSDOWrite(idx, cmd_code, index, value))
            return false;

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);

        while (std::chrono::steady_clock::now() < deadline)
        {
            if (popSDOResponse(idx, resp))
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 1ms 轮询间隔
        }
        return false;
    }

    // ======================================================================
    // 作用：使能指定索引的电机，使电机进入 CiA402 的 Operation Enabled 状态。
    // 流程：
    // 检查电机索引是否有效。
    // 获取对应电机配置 MotorConfig。
    // 通过 SDO 向对象字典 0x6040(ControlWord) 依次写入：
    // 0x0006 (Shutdown)
    // 0x0007 (Switch On)
    // 0x000F (Enable Operation)
    // 完成 CiA402 状态机从 Switch On Disabled 到 Operation Enabled 的转换。
    // 返回值：
    // true：电机成功使能
    // false：SDO通信失败或索引非法
    // ======================================================================
    bool enableMotor(int idx)
    {
        if (idx < 0 || idx >= motorCount()) return false;

        const MotorConfig &m = motors_[idx];
        SDOResponse resp;

        if (!sendSDOWriteAndWait(idx, 0x2B, 0x6040, 0x0006, resp))
            return false;

        if (!sendSDOWriteAndWait(idx, 0x2B, 0x6040, 0x0007, resp))
            return false;

        if (!sendSDOWriteAndWait(idx, 0x2B, 0x6040, 0x000F, resp))
            return false;

        std::cout << "[MotorManager] Motor " << m.dev_id
                  << " enabled on ch" << m.channel << "\n";
        return true;
    }

    // ================================================================
    // 作用：关闭指定电机的使能，使电机退出 Operation Enabled 状态。
    // 流程：
    // 检查电机索引是否有效。
    // 通过 SDO 向对象字典 0x6040 写入 0x0006 (Shutdown)。
    // 电机进入 Ready to Switch On 状态，不再输出力矩。
    // 返回值：
    // true：成功关闭电机
    // false：通信失败或索引非法
    // ===============================================================
    bool disableMotor(int idx)
    {
        if (idx < 0 || idx >= motorCount()) return false;

        const MotorConfig &m = motors_[idx];
        SDOResponse resp;

        if (!sendSDOWriteAndWait(idx, 0x2B, 0x6040, 0x0006, resp))
            return false;

        std::cout << "[MotorManager] Motor " << m.dev_id
                  << " disabled on ch" << m.channel << "\n";
        return true;
    }

    // ============================================================================
    // 使能系统中的所有电机。
    // ============================================================================
    bool enableAll()
    {
        for (int i = 0; i < motorCount(); i++)
            if (!enableMotor(i)) return false;
        return true;
    }

    // ============================================================================
    // 失能系统中的所有电机。
    // ============================================================================
    bool disableAll()
    {
        for (int i = 0; i < motorCount(); i++)
            if (!disableMotor(i)) return false;
        return true;
    }

    // ============================================================================
    // 清除指定电机的错误状态(Fault)。
    // ============================================================================   
    bool clearError(int idx)
    {
        if (idx < 0 || idx >= motorCount()) return false;

        const MotorConfig &m = motors_[idx];
        SingleAxisCmd cmd;
        cmd.dev_id      = m.dev_id;
        cmd.enable      = false;
        cmd.release_brake = false;
        cmd.clear_error = true;
        cmd.mode        = ControlMode::CSP;
        uint8_t buf[7];
        encodeSingleAxisCmd(cmd, buf);
        if (!driver_.sendData(CAN_ID_PDO_SINGLE_BASE | m.dev_id,
                          0, 1, 0, m.channel, buf, 7))
            return false;
        return true;
    }

private:
    bool sendSingleAxisCmd_(int channel, int dev_id, SingleAxisCmd cmd)
    {
        if (findIndex(channel, dev_id) < 0)
        {
            std::cerr << "[MotorManager] sendSingleAxisCmd: motor not found"
                      << " ch=" << channel << " dev_id=" << dev_id << "\n";
            return false;
        }

        cmd.dev_id = static_cast<uint8_t>(dev_id);
        uint8_t buf[7];
        encodeSingleAxisCmd(cmd, buf);
        return driver_.sendData(CAN_ID_PDO_SINGLE_BASE | static_cast<uint32_t>(dev_id),
                                0, 1, 0, channel, buf, 7);
    }

    std::vector<int> usedChannels() const
    {
        std::unordered_set<int> unique;
        unique.reserve(motors_.size());
        for (const auto &m : motors_)
            unique.insert(m.channel);

        std::vector<int> channels;
        channels.reserve(unique.size());
        for (int ch : unique)
            channels.push_back(ch);

        return channels;
    }

    static uint64_t makeKey(int channel, int dev_id)
    {
        return (static_cast<uint64_t>(channel) << 32) | static_cast<uint64_t>(dev_id);
    }

private:
    CANFDDriver                        driver_;
    std::vector<MotorConfig>            motors_;
    std::unordered_map<uint64_t, int>   id_to_index_;
    std::unordered_map<int, ChannelState> states_;
    bool initialized_;
};

