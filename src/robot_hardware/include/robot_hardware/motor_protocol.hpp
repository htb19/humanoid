#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unistd.h>

#include "controlcanfd.h"

// ============================================================
//  CAN ID 定义
// ============================================================
static constexpr uint32_t CAN_ID_SDO_TX_BASE    = 0x600;  // 主站→电机 SDO
static constexpr uint32_t CAN_ID_SDO_RX_BASE    = 0x580;  // 电机→主站 SDO反馈
static constexpr uint32_t CAN_ID_PDO_SINGLE_BASE = 0x100;  // 单轴PDO控制
static constexpr uint32_t CAN_ID_PDO_MULTI       = 0x200;  // 多轴PDO控制（广播）
static constexpr uint32_t CAN_ID_FEEDBACK_BASE   = 0x300;  // 电机反馈帧
static constexpr uint32_t CAN_ID_SYNC            = 0x080;  // 同步帧

// ============================================================
//  控制模式枚举
// ============================================================
enum class ControlMode : uint8_t
{
    ProfilePosition = 0x01,  // 轮廓位置模式（含内部速度规划，适合低频指令）
    ProfileVelocity = 0x02,  // 轮廓速度模式
    CSP             = 0x03,  // 周期同步位置（1kHz实时控制首选）
    CSV             = 0x04,  // 周期同步速度
    CurrentLoop     = 0x05,  // 电流环模式
};

// ============================================================
//  统一 rad ↔ encoder count 转换 (int16 量程)
// ============================================================
inline int32_t radToCount(double rad)
{
    const double raw = rad * 32768.0 / M_PI;
    if (raw > 32767.0) return 32767;
    if (raw < -32768.0) return -32768;
    return static_cast<int32_t>(static_cast<int16_t>(raw));
}

inline double countToRad(int32_t count)
{
    return static_cast<double>(count) * M_PI / 32768.0;
}

// ============================================================
//  电机反馈状态结构体
// ============================================================
struct MotorFeedback
{
    int      dev_id;            // 电机ID（1~127）
    int      channel;           // CAN通道（0或1）

    float    position_deg;      // 负载端位置（度）[-180, 180]
    float    position_rad;      // 负载端位置（弧度）
    int16_t  velocity_rpm;      // 电机端速度（RPM）
    float    velocity_rad_s;    // 电机端速度（rad/s，供ros2_control使用）
    int16_t  current_ma;        // 实际Iq电流（mA）

    uint16_t error_code;        // 错误码（位字段，见下方枚举）
    float    temperature_deg;   // 电机线圈温度（°C）
    uint8_t  control_mode;      // 当前控制模式

    // 状态位
    bool enabled;               // 使能状态
    bool brake_released;        // 抱闸释放
    bool has_error;             // 报错状态
    bool in_position;           // 位置到位状态
};

// 错误码位定义
enum MotorError : uint16_t
{
    ERROR_OVERVOLTAGE     = 0x0001,
    ERROR_UNDERVOLTAGE    = 0x0002,
    ERROR_OVERTEMP        = 0x0004,
    ERROR_STALL           = 0x0008,
    ERROR_OVERLOAD        = 0x0010,
    ERROR_FLYWHEEL        = 0x0020,
    ERROR_POS_LIMIT       = 0x0040,
    ERROR_NEG_LIMIT       = 0x0080,
    ERROR_ENCODER         = 0x0100,
    ERROR_OVERSPEED       = 0x0200,
    ERROR_ANGLE_INIT_FAIL = 0x0400,
    ERROR_POS_DEVIATION   = 0x1000,
    ERROR_ENCODER_FAULT   = 0x2000,
};

// ============================================================
//  解析反馈帧（0x300 | Dev_ID），12字节 CANFD
// ============================================================
inline bool parseMotorFeedback(const ZCAN_ReceiveFD_Data &msg,
                                int channel,
                                MotorFeedback &fb)
{
    uint32_t can_id = msg.frame.can_id & 0x1FFFFFFF;

    if ((can_id & 0x700) != CAN_ID_FEEDBACK_BASE)
        return false;

    fb.dev_id  = static_cast<int>(can_id & 0x7F);
    fb.channel = channel;

    const uint8_t *d = msg.frame.data;

    // [0][1]: 负载端位置，高字节在前，int16
    //         [-32768 ~ 32767] 对应 [-180° ~ 180°]
    int16_t raw_pos   = static_cast<int16_t>((d[0] << 8) | d[1]);
    fb.position_deg   = static_cast<float>(raw_pos) * 90.0f / 16384.0f;
    fb.position_rad   = static_cast<float>(raw_pos) * static_cast<float>(M_PI) / 32768.0f;

    // [2][3]: 电机端速度，高字节在前，int16，单位 RPM
    fb.velocity_rpm   = static_cast<int16_t>((d[2] << 8) | d[3]);
    fb.velocity_rad_s = static_cast<float>(fb.velocity_rpm) * 2.0f *
                        static_cast<float>(M_PI) / 60.0f;

    // [4][5]: 实际Iq电流，高字节在前，int16，单位 mA
    fb.current_ma     = static_cast<int16_t>((d[4] << 8) | d[5]);

    // [6][7]: 错误码，高字节在前
    fb.error_code     = static_cast<uint16_t>((d[6] << 8) | d[7]);

    // [8][9]: 电机线圈温度，高字节在前，int16，单位 0.1°C
    int16_t raw_temp   = static_cast<int16_t>((d[8] << 8) | d[9]);
    fb.temperature_deg = static_cast<float>(raw_temp) / 10.0f;

    // [10]: 控制模式反馈
    fb.control_mode = d[10];

    // [11]: 状态字节
    fb.enabled        = (d[11] >> 7) & 0x01;
    fb.brake_released = (d[11] >> 6) & 0x01;
    fb.has_error      = (d[11] >> 5) & 0x01;
    fb.in_position    = (d[11] >> 4) & 0x01;

    return true;
}

// ============================================================
//  编码控制字节 [0]
// ============================================================
inline uint8_t encodeControlByte(bool enable, bool release_brake,
                                  bool clear_error, ControlMode mode)
{
    uint8_t b = 0;
    b |= (enable        ? 1u : 0u) << 7;
    b |= (release_brake ? 1u : 0u) << 6;
    b |= (clear_error   ? 1u : 0u) << 5;
    b |= (static_cast<uint8_t>(mode) & 0x0Fu) << 1;
    return b;
}

// ============================================================
//  单轴PDO控制指令结构体
// ============================================================
struct SingleAxisCmd
{
    uint8_t     dev_id           = 1;
    bool        enable           = true;
    bool        release_brake    = true;
    bool        clear_error      = false;
    ControlMode mode             = ControlMode::CSP;

    // 含义随模式变化：
    //   CSP / ProfilePosition : 目标位置（度）[-180, 180]
    //   CSV / ProfileVelocity : 目标速度（RPM）
    //   CurrentLoop           : 目标Iq电流（mA）
    float target_param1 = 0.0f;

    // 仅 ProfilePosition / ProfileVelocity 模式有效
    // 轮廓加减速（RPM/S）[0 ~ 20000]
    float accel_rpms = 0.0f;

    // 仅 ProfilePosition 模式有效
    // 输出端轮廓速度（RPM）
    float profile_velocity_rpm = 0.0f;
};

// ============================================================
//  编码单轴PDO控制帧，输出 7 字节
//  CAN ID = 0x100 | dev_id
// ============================================================
inline void encodeSingleAxisCmd(const SingleAxisCmd &cmd, uint8_t *buf7)
{
    memset(buf7, 0, 7);

    buf7[0] = encodeControlByte(cmd.enable, cmd.release_brake,
                                 cmd.clear_error, cmd.mode);

    // 目标参数1 → int16，高字节在前
    int16_t p1 = 0;
    float val = 0;
    switch (cmd.mode)
    {
    case ControlMode::CSP:
    case ControlMode::ProfilePosition:
        // 度 → cnt，[-180° ~ 180°] 对应 [-32768 ~ 32767]
        val = cmd.target_param1 * 16384.0f / 90.0f;
        p1 = static_cast<int16_t>(val >= 0 ? val + 0.5f : val - 0.5f);
        break;
    case ControlMode::CSV:
    case ControlMode::ProfileVelocity:
        p1 = static_cast<int16_t>(cmd.target_param1);
        break;
    case ControlMode::CurrentLoop:
        p1 = static_cast<int16_t>(cmd.target_param1);
        break;
    default:
        break;
    }
    buf7[1] = static_cast<uint8_t>(p1 >> 8);
    buf7[2] = static_cast<uint8_t>(p1 & 0xFF);

    // 加减速 → uint16，高字节在前
    uint16_t acc = static_cast<uint16_t>(cmd.accel_rpms);
    buf7[3] = static_cast<uint8_t>(acc >> 8);
    buf7[4] = static_cast<uint8_t>(acc & 0xFF);

    // 轮廓速度 → uint16，高字节在前
    uint16_t vel = static_cast<uint16_t>(cmd.profile_velocity_rpm);
    buf7[5] = static_cast<uint8_t>(vel >> 8);
    buf7[6] = static_cast<uint8_t>(vel & 0xFF);
}

// ============================================================
//  多轴PDO控制帧，最多 8 个电机，输出 64 字节
//  CAN ID = 0x200（广播，CANFD）
// ============================================================
struct MultiAxisCmd
{
    int           count = 0;
    SingleAxisCmd cmds[8];
};

inline void encodeMultiAxisCmd(const MultiAxisCmd &multi, uint8_t *buf64)
{
    memset(buf64, 0, 64);
    int n = (multi.count > 8) ? 8 : multi.count;
    for (int i = 0; i < n; i++)
    {
        encodeSingleAxisCmd(multi.cmds[i], buf64 + i * 7);
        buf64[56 + i] = multi.cmds[i].dev_id;
    }
}

// ============================================================
//  SDO 编码工具（配置参数，不走实时路径）
// ============================================================
// cmd_code: 0x23=写4字节, 0x2B=写2字节, 0x2F=写1字节
inline void encodeSDOWrite(uint8_t cmd_code, uint16_t index,
                            uint8_t subindex, uint32_t value,
                            uint8_t *buf8)
{
    memset(buf8, 0, 8);
    buf8[0] = cmd_code;
    buf8[1] = static_cast<uint8_t>(index & 0xFF);   // 低字节在前
    buf8[2] = static_cast<uint8_t>(index >> 8);
    buf8[3] = subindex;
    buf8[4] = static_cast<uint8_t>(value & 0xFF);   // 数据低位在前
    buf8[5] = static_cast<uint8_t>((value >> 8)  & 0xFF);
    buf8[6] = static_cast<uint8_t>((value >> 16) & 0xFF);
    buf8[7] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

inline void encodeSDORead(uint16_t index, uint8_t subindex, uint8_t *buf8)
{
    memset(buf8, 0, 8);
    buf8[0] = 0x40;
    buf8[1] = static_cast<uint8_t>(index & 0xFF);
    buf8[2] = static_cast<uint8_t>(index >> 8);
    buf8[3] = subindex;
}

// ============================================================
//  SDO 反馈解析
// ============================================================

// ── SDO 响应类型 ─────────────────────────────────────────────
enum class SDOResponseType : uint8_t
{
    ReadOk   = 0,   // 读取成功
    WriteOk  = 1,   // 写入确认
    Abort    = 2,   // 中止/错误
    Unknown  = 3,   // 无法识别
};

// ── SDO 读取反馈结构体 ────────────────────────────────────────
//   所有字段在 parseSDOResponse() 返回 true 后有效
struct SDOResponse
{
    SDOResponseType type       = SDOResponseType::Unknown;
    uint8_t         node_id    = 0;       // 节点ID（1~127）
    uint16_t        index      = 0;       // 对象索引
    uint8_t         subindex   = 0;       // 子索引
    uint8_t         cmd_code   = 0;       // 原始指令码

    // ── 读取成功时有效 ────────────────────────────────────────
    int32_t         raw_value  = 0;       // 原始整数值（按数据类型解析）

    // 常用对象直接解析好的物理量（仅对应索引时有效，否则为 0）
    float           current_ma        = 0.0f;   // 0x6078: 电流（mA）
    int32_t         velocity_rpm      = 0;      // 0x606C: 速度（RPM）
    float           position_deg      = 0.0f;      // 0x6064: 位置（cnt）
    float           position_rad      = 0.0f;      // 0x6064: 位置（cnt）
    float           temperature_deg   = 0.0f;   // 0x2662: 温度（°C）
    float           torque_nm         = 0.0f;   // 0x2663: 力矩（Nm）
    float           bus_voltage_v     = 0.0f;   // 0x6079: 母线电压（V）
    uint16_t        error_code        = 0;      // 0x603F: 错误码位字段
    uint16_t        status_word       = 0;      // 0x6041: DS402 状态字
    int8_t          operation_mode    = 0;      // 0x6060: 运行模式

    // ── 中止时有效 ────────────────────────────────────────────
    uint32_t        abort_code        = 0;      // SDO 中止码（原始值）
};

// ── SDO 指令码 → 有效数据字节数 ──────────────────────────────
//   返回 0 表示写入确认（0x60）或未知指令
static inline uint8_t sdoCmdDataSize(uint8_t cmd)
{
    switch (cmd)
    {
    case 0x4F: return 1;   // expedited, 1 byte
    case 0x4B: return 2;   // expedited, 2 bytes
    case 0x47: return 3;   // expedited, 3 bytes
    case 0x43: return 4;   // expedited, 4 bytes
    default:   return 0;
    }
}

// ── 小端序读取工具 ────────────────────────────────────────────
static inline uint32_t sdoReadU32LE(const uint8_t *p)
{
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

// ── 按类型解析 4 字节缓冲（小端序）→ int32 ───────────────────
static inline int32_t sdoDecodeTyped(const uint8_t *buf4, uint8_t data_size,
                                      uint16_t index)
{
    uint8_t b[4] = {0, 0, 0, 0};
    uint8_t n = (data_size < 4) ? data_size : 4;
    memcpy(b, buf4, n);

    // 根据已知索引选择有符号/无符号及位宽
    switch (index)
    {
    // INT8
    case 0x6060:
        { int8_t v; memcpy(&v, b, 1); return v; }

    // INT16
    case 0x6078: case 0x6071:
        { int16_t v; memcpy(&v, b, 2); return v; }

    // UINT16
    case 0x603F: case 0x6041: case 0x6040:
        { uint16_t v; memcpy(&v, b, 2); return static_cast<int32_t>(v); }

    // INT32
    case 0x606C: case 0x6064: case 0x607A:
    case 0x60FF: case 0x607D: case 0x2662: case 0x2663:
        { int32_t v; memcpy(&v, b, 4); return v; }

    // UINT32（默认）
    default:
        { uint32_t v; memcpy(&v, b, 4); return static_cast<int32_t>(v); }
    }
}

/**
 * @brief  解析 SDO 反馈帧（来自电机从站的 0x58x 帧）
 *
 * @param  msg      ZCAN_ReceiveFD_Data 接收帧（DLC 须 >= 8）
 * @param  resp     输出：解析结果，失败时仅 type/node_id/index 等基础字段有效
 * @return true     解析成功（ReadOk 或 WriteOk）
 *         false    中止帧或无法识别（resp.type 指示原因）
 *
 * @note   只处理 CAN ID 在 [0x581, 0x5FF] 范围内的帧，其余返回 false。
 *
 * 典型用法：
 * @code
 *   SDOResponse resp;
 *   if (parseSDOResponse(msg, resp)) {
 *       if (resp.index == 0x6078)
 *           printf("电流: %.1f mA\n", resp.current_ma);
 *   } else if (resp.type == SDOResponseType::Abort) {
 *       printf("SDO 中止: 0x%08X\n", resp.abort_code);
 *   }
 * @endcode
 */
inline bool parseSDOResponse(const ZCAN_ReceiveFD_Data &msg, SDOResponse &resp)
{
    resp = SDOResponse{};   // 清零

    uint32_t can_id = msg.frame.can_id & 0x1FFFFFFF;

    // 不在 SDO 反馈范围内
    if (can_id < 0x581 || can_id > 0x5FF)
        return false;

    resp.node_id  = static_cast<uint8_t>(can_id & 0x0F);
    resp.cmd_code = msg.frame.data[0];
    resp.index    = static_cast<uint16_t>(msg.frame.data[1])
                  | (static_cast<uint16_t>(msg.frame.data[2]) << 8);
    resp.subindex = msg.frame.data[3];

    const uint8_t *payload = msg.frame.data + 4;   // Data[4~7]

    // ── 中止响应 ─────────────────────────────────────────────
    if (resp.cmd_code == 0x80)
    {
        resp.type       = SDOResponseType::Abort;
        resp.abort_code = sdoReadU32LE(payload);
        return false;
    }

    // ── 写入确认（0x60） ─────────────────────────────────────
    if (resp.cmd_code == 0x60)
    {
        resp.type = SDOResponseType::WriteOk;
        return true;
    }

    // ── 读取响应（0x4x） ─────────────────────────────────────
    uint8_t data_size = sdoCmdDataSize(resp.cmd_code);
    if (data_size == 0)
    {
        resp.type = SDOResponseType::Unknown;
        return false;
    }

    resp.type      = SDOResponseType::ReadOk;
    resp.raw_value = sdoDecodeTyped(payload, data_size, resp.index);

    // ── 常用对象物理量直接解析 ────────────────────────────────
    switch (resp.index)
    {
    case 0x6078:   // 电流当前值，INT16，mA
        resp.current_ma = static_cast<float>(static_cast<int16_t>(resp.raw_value));
        break;

    case 0x606C:   // 电机端速度，INT32，RPM
        resp.velocity_rpm = resp.raw_value;
        break;

    case 0x6064:   // 电机实际位置，INT32，cnt
        resp.position_deg = static_cast<float>(resp.raw_value)*90.0f/16384.0f;
        resp.position_rad = static_cast<float>(resp.raw_value)*static_cast<float>(M_PI)/32768.0f;
        break;

    case 0x2662:   // 温度传感器，INT32，单位 0.1°C
        resp.temperature_deg = static_cast<float>(resp.raw_value) * 0.1f;
        break;

    case 0x2663:   // 力矩传感器，INT32，单位 0.001 Nm
        resp.torque_nm = static_cast<float>(resp.raw_value) * 0.001f;
        break;

    case 0x6079:   // 母线电压，UINT32，单位 0.1V
        resp.bus_voltage_v =
            static_cast<float>(static_cast<uint32_t>(resp.raw_value)) * 0.1f;
        break;

    case 0x603F:   // 错误码，UINT16，位字段（对应 MotorError 枚举）
        resp.error_code = static_cast<uint16_t>(resp.raw_value);
        break;

    case 0x6041:   // DS402 状态字，UINT16
        resp.status_word = static_cast<uint16_t>(resp.raw_value);
        break;

    case 0x6060:   // 运行模式指令，INT8
        resp.operation_mode = static_cast<int8_t>(resp.raw_value);
        break;

    default:
        break;   // 其余参数通过 raw_value 自行使用
    }

    return true;
}

/**
 * @brief  判断 SDO 反馈中的错误码是否包含指定故障位
 *
 * @param  resp   parseSDOResponse 解析结果（index 须为 0x603F）
 * @param  err    MotorError 枚举位
 * @return true   该故障位置位
 */
inline bool sdoHasError(const SDOResponse &resp, MotorError err)
{
    return (resp.index == 0x603F) && (resp.error_code & static_cast<uint16_t>(err));
}

/**
 * @brief  从 DS402 状态字中提取驱动器状态机状态
 *
 * DS402 状态机低6位定义：
 *   xxxx xx00 = Not Ready to Switch On
 *   xxxx x001 = Switch On Disabled       (0x40)
 *   xxxx x011 = Ready to Switch On       (0x21 mask 0x6F == 0x21)
 *   xxxx x111 = Switched On              (0x23)
 *   xxxx1111  = Operation Enabled        (0x27)
 *   xxxx1000  = Fault                    (0x08 mask 0x4F == 0x08)
 *   xxxx0111  = Quick Stop Active
 *   xxxx1111  = Fault Reaction Active
 */
enum class DS402State : uint8_t
{
    NotReady        = 0,
    SwitchOnDisabled,   // 准备好启动（上电初始）
    ReadyToSwitchOn,    // 松开抱闸
    SwitchedOn,         // 已上电
    OperationEnabled,   // 使能激活（正常运行）
    Fault,              // 报错状态
    QuickStopActive,
    FaultReactionActive,
    Unknown,
};

inline DS402State parseDS402State(uint16_t status_word)
{
    uint16_t s = status_word & 0x006F;   // 取关键位 [6,5,3,2,1,0]

    if      ((s & 0x4F) == 0x00) return DS402State::NotReady;
    else if ((s & 0x4F) == 0x40) return DS402State::SwitchOnDisabled;
    else if ((s & 0x6F) == 0x21) return DS402State::ReadyToSwitchOn;
    else if ((s & 0x6F) == 0x23) return DS402State::SwitchedOn;
    else if ((s & 0x6F) == 0x27) return DS402State::OperationEnabled;
    else if ((s & 0x4F) == 0x08) return DS402State::Fault;
    else if ((s & 0x6F) == 0x07) return DS402State::QuickStopActive;
    else if ((s & 0x4F) == 0x0F) return DS402State::FaultReactionActive;
    else                         return DS402State::Unknown;
}

