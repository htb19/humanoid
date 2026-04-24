#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <string.h>
#include <functional>
#include <atomic>
#include <unordered_map>   
#include <queue>          
#include <mutex>          
#include "controlcanfd.h"
#include "ring_buffer.hpp"

typedef DEVICE_HANDLE (*pZCAN_OpenDevice)(UINT deviceType, UINT deviceIndex, UINT reserved);
typedef UINT (*pZCAN_CloseDevice)(DEVICE_HANDLE device_handle);
typedef UINT (*pZCAN_GetDeviceInf)(DEVICE_HANDLE device_handle, ZCAN_DEVICE_INFO *pInfo);
typedef UINT (*pZCAN_IsDeviceOnLine)(DEVICE_HANDLE device_handle);
typedef CHANNEL_HANDLE (*pZCAN_InitCAN)(DEVICE_HANDLE device_handle, UINT can_index, ZCAN_CHANNEL_INIT_CONFIG *pInitConfig);
typedef UINT (*pZCAN_StartCAN)(CHANNEL_HANDLE channel_handle);
typedef UINT (*pZCAN_ResetCAN)(CHANNEL_HANDLE channel_handle);
typedef UINT (*pZCAN_ClearBuffer)(CHANNEL_HANDLE channel_handle);
typedef UINT (*pZCAN_GetReceiveNum)(CHANNEL_HANDLE channel_handle, BYTE type);
typedef UINT (*pZCAN_Transmit)(CHANNEL_HANDLE channel_handle, ZCAN_Transmit_Data *pTransmit, UINT len);
typedef UINT (*pZCAN_Receive)(CHANNEL_HANDLE channel_handle, ZCAN_Receive_Data *pReceive, UINT len, int wait_time);
typedef UINT (*pZCAN_TransmitFD)(CHANNEL_HANDLE channel_handle, ZCAN_TransmitFD_Data *pTransmit, UINT len);
typedef UINT (*pZCAN_ReceiveFD)(CHANNEL_HANDLE channel_handle, ZCAN_ReceiveFD_Data *pReceive, UINT len, int wait_time);
typedef IProperty *(*pGetIProperty)(DEVICE_HANDLE device_handle);
typedef UINT (*pReleaseIProperty)(IProperty *pIProperty);
typedef UINT (*pZCAN_SetAbitBaud)(DEVICE_HANDLE device_handle, UINT can_index, UINT abitbaud);
typedef UINT (*pZCAN_SetDbitBaud)(DEVICE_HANDLE device_handle, UINT can_index, UINT dbitbaud);
typedef UINT (*pZCAN_SetCANFDStandard)(DEVICE_HANDLE device_handle, UINT can_index, UINT canfd_standard);
typedef UINT (*pZCAN_SetResistanceEnable)(DEVICE_HANDLE device_handle, UINT can_index, UINT enable);
typedef UINT (*pZCAN_SetBaudRateCustom)(DEVICE_HANDLE device_handle, UINT can_index, char *RateCustom);
typedef UINT (*pZCAN_ClearFilter)(CHANNEL_HANDLE channel_handle);
typedef UINT (*pZCAN_AckFilter)(CHANNEL_HANDLE channel_handle);
typedef UINT (*pZCAN_SetFilterMode)(CHANNEL_HANDLE channel_handle, UINT mode);
typedef UINT (*pZCAN_SetFilterStartID)(CHANNEL_HANDLE channel_handle, UINT start_id);
typedef UINT (*pZCAN_SetFilterEndID)(CHANNEL_HANDLE channel_handle, UINT EndID);

class CANFDDriver
{
public:
    static constexpr int MAX_CHANNELS = 2;
    static constexpr size_t RX_QUEUE_SIZE = 4096;

public:
    CANFDDriver(int device_type, int device_index)
        : lib_handle(nullptr),
          device_type_(device_type),
          device_index_(device_index),
          device_handle_(INVALID_DEVICE_HANDLE)
    {
        for (int i = 0; i < MAX_CHANNELS; ++i)
        {
            channel_handle_[i] = INVALID_CHANNEL_HANDLE;
            recv_run_flag_[i].store(false);
        }
    }

    ~CANFDDriver()
    {
        stopAllReceiveThreads();
        closeDevice();

        if (lib_handle)
            dlclose(lib_handle);
    }

    bool loadLibrary(const std::string &path)
    {
        lib_handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!lib_handle)
        {
            std::cout << "dlopen failed: " << dlerror() << std::endl;
            return false;
        }

        bool ok = true;
        ok &= loadAPI("ZCAN_OpenDevice", (void **)&zcan_open_device);
        ok &= loadAPI("ZCAN_CloseDevice", (void **)&zcan_close_device);
        ok &= loadAPI("ZCAN_GetDeviceInf", (void **)&zcan_get_device_inf);
        ok &= loadAPI("ZCAN_IsDeviceOnLine", (void **)&zcan_is_device_online);
        ok &= loadAPI("ZCAN_InitCAN", (void **)&zcan_init_can);
        ok &= loadAPI("ZCAN_StartCAN", (void **)&zcan_start_can);
        ok &= loadAPI("ZCAN_ResetCAN", (void **)&zcan_reset_can);
        ok &= loadAPI("ZCAN_ClearBuffer", (void **)&zcan_clear_buffer);
        ok &= loadAPI("ZCAN_GetReceiveNum", (void **)&zcan_get_receive_num);
        ok &= loadAPI("ZCAN_Transmit", (void **)&zcan_transmit);
        ok &= loadAPI("ZCAN_Receive", (void **)&zcan_receive);
        ok &= loadAPI("ZCAN_TransmitFD", (void **)&zcan_transmit_fd);
        ok &= loadAPI("ZCAN_ReceiveFD", (void **)&zcan_receive_fd);
        ok &= loadAPI("GetIProperty", (void **)&get_iproperty);
        ok &= loadAPI("ReleaseIProperty", (void **)&release_iproperty);

        ok &= loadAPI("ZCAN_SetAbitBaud", (void **)&zcan_set_abit_baud);
        ok &= loadAPI("ZCAN_SetDbitBaud", (void **)&zcan_set_dbit_baud);
        ok &= loadAPI("ZCAN_SetCANFDStandard", (void **)&zcan_set_canfd_standard);
        ok &= loadAPI("ZCAN_SetResistanceEnable", (void **)&zcan_set_resistance_enable);
        ok &= loadAPI("ZCAN_SetBaudRateCustom", (void **)&zcan_set_baud_rate_custom);

        ok &= loadAPI("ZCAN_ClearFilter", (void **)&zcan_clear_filter);
        ok &= loadAPI("ZCAN_AckFilter", (void **)&zcan_ack_filter);
        ok &= loadAPI("ZCAN_SetFilterMode", (void **)&zcan_set_filter_mode);
        ok &= loadAPI("ZCAN_SetFilterStartID", (void **)&zcan_set_filter_start_id);
        ok &= loadAPI("ZCAN_SetFilterEndID", (void **)&zcan_set_filter_end_id);

        return ok;
    }

    inline bool openDevice()
    {
        if (!zcan_open_device)
            return false;
        device_handle_ = zcan_open_device(device_type_, device_index_, 0);
        return device_handle_ != INVALID_DEVICE_HANDLE;
    }

    inline bool setCANFDStandard(uint standard)
    {
        if (device_handle_ == INVALID_DEVICE_HANDLE)
            return false;
        if (!zcan_set_canfd_standard)
            return false;

        if(1 != zcan_set_canfd_standard(device_handle_, device_index_, standard))
            return false;

        return true;
    }

    inline bool setBaud(uint channel, int abit, int dbit)
    {
        if (device_handle_ == INVALID_DEVICE_HANDLE)
            return false;
        if (channel >= MAX_CHANNELS)
            return false;
        if (!zcan_set_abit_baud || !zcan_set_dbit_baud)
            return false;

        if(1 != zcan_set_abit_baud(device_handle_, channel, abit))
            return false;
        if(1 != zcan_set_dbit_baud(device_handle_, channel, dbit))
            return false;
        return true;
    }

    inline bool initCAN(uint channel)
    {
        if (device_handle_ == INVALID_DEVICE_HANDLE)
            return false;
        if (channel >= MAX_CHANNELS)
            return false;
        if (!zcan_init_can)
            return false;

        ZCAN_CHANNEL_INIT_CONFIG cfg{};
        cfg.can_type = 1;
        cfg.canfd.acc_code = 0;
        cfg.canfd.acc_mask = 0xFFFFFFFF;
        cfg.canfd.filter = 1;
        cfg.canfd.mode = 0;

        CHANNEL_HANDLE handle = zcan_init_can(device_handle_, channel, &cfg);
        if (handle == INVALID_CHANNEL_HANDLE)
            return false;

        channel_handle_[channel] = handle;
        return true;
    }

    inline bool resetCAN(uint channel)
    {
        if (channel >= MAX_CHANNELS)
            return false;
        if (channel_handle_[channel] == INVALID_CHANNEL_HANDLE)
            return false;
        if (!zcan_reset_can)
            return false;
        return zcan_reset_can(channel_handle_[channel]) == STATUS_OK;
    }

    inline bool startCAN(uint channel)
    {
        if (channel >= MAX_CHANNELS)
            return false;
        if (channel_handle_[channel] == INVALID_CHANNEL_HANDLE)
            return false;
        if (!zcan_start_can)
            return false;
        return zcan_start_can(channel_handle_[channel]) == STATUS_OK;
    }

    inline bool setFilter(uint channel, uint filterMode, uint startID, uint endID)
    {
        if (channel >= MAX_CHANNELS)
            return false;
        if (channel_handle_[channel] == INVALID_CHANNEL_HANDLE)
            return false;
        if (!zcan_clear_filter || !zcan_set_filter_mode || ! zcan_set_filter_start_id ||
            !zcan_set_filter_end_id || !zcan_ack_filter)
            return false;

        if(1 != zcan_clear_filter(channel_handle_[channel]))
            return false;
        if(1 != zcan_set_filter_mode(channel_handle_[channel],filterMode))
            return false;
        if(1 != zcan_set_filter_start_id(channel_handle_[channel], startID))
            return false;
        if(1 != zcan_set_filter_end_id(channel_handle_[channel], endID))
            return false;
        if(1 != zcan_ack_filter(channel_handle_[channel]))
            return false;
        return true;
    }

    inline uint MakeCanId(uint id, int eff, int rtr, int err)
    {
        uint ueff = (uint)((eff != 0) ? 1 : 0);
        uint urtr = (uint)((rtr != 0) ? 1 : 0);
        uint uerr = (uint)((err != 0) ? 1 : 0);
        return id | (ueff << 31) | (urtr << 30) | (uerr << 29);
    }

    bool sendData(uint id, uint frame_type_index, uint protocol_index, uint canfd_exp_index, uint channel, const uint8_t *data, uint len)
    {
        // 基本状态与边界检查
        if (device_handle_ == INVALID_DEVICE_HANDLE)
            return false;
        if (channel >= MAX_CHANNELS)
            return false;
        if (channel_handle_[channel] == INVALID_CHANNEL_HANDLE)
            return false;
        if (data == nullptr || len == 0)
            return false;

        uint result = 0;
        if (protocol_index == 0)
        {
            if (!zcan_transmit)
                return false;

            ZCAN_Transmit_Data can_data{};
            can_data.frame.can_id = MakeCanId(id, frame_type_index, 0, 0);
            memcpy(can_data.frame.data, data, len > 8 ? 8 : len);
            can_data.frame.can_dlc = len > 8 ? 8 : len;
            can_data.transmit_type = 1;
            result = zcan_transmit(channel_handle_[channel], &can_data, 1);
        }
        else
        {
            if (!zcan_transmit_fd)
                return false;

            ZCAN_TransmitFD_Data canfd_data{};
            canfd_data.frame.can_id = MakeCanId(id, frame_type_index, 0, 0);
            memcpy(canfd_data.frame.data, data, len > 64 ? 64 : len);
            canfd_data.frame.len = len > 64 ? 64 : len;
            canfd_data.transmit_type = 1;
            canfd_data.frame.flags = ((canfd_exp_index != 0) ? 1 : 0);
            result = zcan_transmit_fd(channel_handle_[channel], &canfd_data, 1);
        }

        return result == 1;
    }

    inline bool closeDevice()
    {
        if (device_handle_ != INVALID_DEVICE_HANDLE && zcan_close_device)
        {
            zcan_close_device(device_handle_);
            device_handle_ = INVALID_DEVICE_HANDLE;
            return true;
        }
        return false;
    }

    inline bool startReceiveThread(int channel)
    {
        if (channel < 0 || channel >= MAX_CHANNELS)
            return false;
        if (channel_handle_[channel] == INVALID_CHANNEL_HANDLE)
            return false;

        recv_run_flag_[channel].store(true);

        // ========== 调试版本 1：先用普通调度测试 ==========
        // 注释掉实时调度，用默认 SCHED_OTHER
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        
        // pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
        // struct sched_param param;
        // param.sched_priority = 80;//设置优先级
        // pthread_attr_setschedparam(&attr, &param);
        // pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

        auto *arg = new ThreadArg{this, channel};
        if (pthread_create(&recv_threads_[channel], &attr, recvThread, arg) != 0)
        {
            delete arg;
            recv_run_flag_[channel] = false;
            pthread_attr_destroy(&attr);
            return false;
        }

        pthread_attr_destroy(&attr);
        return true;
    }

    inline bool stopReceiveThread(int channel)
    {
        if (channel < 0 || channel >= MAX_CHANNELS)
            return false;

        if (recv_run_flag_[channel].load())
        {
            recv_run_flag_[channel].store(false);
            pthread_join(recv_threads_[channel], nullptr);
            return true;
        }

        return false;
    }

    void stopAllReceiveThreads()
    {
        for (int ch = 0; ch < MAX_CHANNELS; ch++)
            stopReceiveThread(ch);
    }

    bool popFrame(int channel, ZCAN_ReceiveFD_Data &msg)//PDO反馈数据
    {
        if (channel < 0 || channel >= MAX_CHANNELS)
            return false;
        return rx_queue_[channel].pop(msg);
    }

    bool popFrameById(int channel, uint32_t can_id, ZCAN_ReceiveFD_Data &msg)//SDO反馈数据
    {
        if (channel < 0 || channel >= MAX_CHANNELS)
            return false;

        std::lock_guard<std::mutex> lock(rx_id_mutex_[channel]);
        auto it = rx_id_queues_[channel].find(can_id);
        if (it == rx_id_queues_[channel].end() || it->second.empty())
            return false;

        msg = it->second.front();
        it->second.pop();
        return true;
    }

private:

    inline bool loadAPI(const char *name, void **func_ptr)
    {
        *func_ptr = dlsym(lib_handle, name);
        if (!*func_ptr)
        {
            std::cout << "Load API failed: " << name << std::endl;
            return false;
        }
        return true;
    }

    struct ThreadArg
    {
        CANFDDriver *self;
        int channel;
    };

    static void *recvThread(void *arg)
    {
        auto *targ = static_cast<ThreadArg *>(arg);
        CANFDDriver *self = targ->self;
        int channel = targ->channel;
        delete targ;

        if (channel < 0 || channel >= MAX_CHANNELS)
            return nullptr;
        if (self->device_handle_ == INVALID_DEVICE_HANDLE)
            return nullptr;
        if (self->channel_handle_[channel] == INVALID_CHANNEL_HANDLE)
            return nullptr;
        if (!self->zcan_receive_fd)
            return nullptr;

        constexpr int WAIT_TIME = 1;
        ZCAN_ReceiveFD_Data buf[2048];

        while (self->recv_run_flag_[channel].load())
        {
            int len = self->zcan_receive_fd(
                self->channel_handle_[channel],
                buf,
                2048,
                WAIT_TIME);
            if (len <= 0)   continue;

            for (int i = 0; i < len; i++)
            {
                uint32_t id = buf[i].frame.can_id & 0x1FFFFFFF;
                self->rx_queue_[channel].push(buf[i]);

                if (id >= 0x580 && id <= 0x5FF)
                {
                    std::lock_guard<std::mutex> lock(self->rx_id_mutex_[channel]);
                    self->rx_id_queues_[channel][id].push(buf[i]);
                }
            }
        }

        return nullptr;
    }

private:
    void *lib_handle;

    int device_type_;
    int device_index_;

    DEVICE_HANDLE device_handle_;
    CHANNEL_HANDLE channel_handle_[MAX_CHANNELS];

    pthread_t recv_threads_[MAX_CHANNELS];
    std::atomic<bool> recv_run_flag_[MAX_CHANNELS];
    RingBuffer<ZCAN_ReceiveFD_Data, RX_QUEUE_SIZE> rx_queue_[MAX_CHANNELS];
    std::unordered_map<uint32_t, std::queue<ZCAN_ReceiveFD_Data>> rx_id_queues_[MAX_CHANNELS];
    std::mutex rx_id_mutex_[MAX_CHANNELS];

    pZCAN_OpenDevice zcan_open_device;
    pZCAN_CloseDevice zcan_close_device;
    pZCAN_GetDeviceInf zcan_get_device_inf;
    pZCAN_IsDeviceOnLine zcan_is_device_online;

    pZCAN_InitCAN zcan_init_can;
    pZCAN_StartCAN zcan_start_can;
    pZCAN_ResetCAN zcan_reset_can;

    pZCAN_ClearBuffer zcan_clear_buffer;
    pZCAN_GetReceiveNum zcan_get_receive_num;

    pZCAN_Transmit zcan_transmit;
    pZCAN_Receive zcan_receive;
    pZCAN_TransmitFD zcan_transmit_fd;
    pZCAN_ReceiveFD zcan_receive_fd;

    pGetIProperty get_iproperty;
    pReleaseIProperty release_iproperty;

    pZCAN_SetAbitBaud zcan_set_abit_baud;
    pZCAN_SetDbitBaud zcan_set_dbit_baud;
    pZCAN_SetCANFDStandard zcan_set_canfd_standard;
    pZCAN_SetResistanceEnable zcan_set_resistance_enable;
    pZCAN_SetBaudRateCustom zcan_set_baud_rate_custom;

    pZCAN_ClearFilter zcan_clear_filter;
    pZCAN_AckFilter zcan_ack_filter;
    pZCAN_SetFilterMode zcan_set_filter_mode;
    pZCAN_SetFilterStartID zcan_set_filter_start_id;
    pZCAN_SetFilterEndID zcan_set_filter_end_id;

};
