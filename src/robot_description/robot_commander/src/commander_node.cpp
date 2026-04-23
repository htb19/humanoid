#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <example_interfaces/msg/bool.hpp>
#include <example_interfaces/msg/float64_multi_array.hpp>
#include <robot_interfaces/msg/pose_command.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
using Bool    = example_interfaces::msg::Bool;
using FloatArray = example_interfaces::msg::Float64MultiArray;
using PoseCmd = robot_interfaces::msg::PoseCommand;

using namespace std::placeholders;

// ─────────────────────────────────────────────
//  TaskQueue  —  线程安全的任务队列
//  存储无参 std::function<void()>，工作线程顺序消费
// ─────────────────────────────────────────────
class TaskQueue
{
public:
    // 将一个可调用对象压入队列，并通知工作线程
    void push(std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(task));
        }
        cv_.notify_one();
    }

    // 阻塞等待直到队列非空或停止信号，返回 false 表示应当退出
    bool waitAndPop(std::function<void()> &task)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });

        if (stopped_ && queue_.empty()) {
            return false;
        }

        task = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // 通知工作线程退出（在析构前调用）
    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    std::size_t size()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<std::function<void()>> queue_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    bool                              stopped_ = false;
};

// ─────────────────────────────────────────────
//  Commander
// ─────────────────────────────────────────────
class Commander
{
public:
    explicit Commander(std::shared_ptr<rclcpp::Node> node)
        : node_(node)
    {
        // ── MoveGroup 初始化 ──────────────────────────
        right_arm_     = std::make_shared<MoveGroupInterface>(node_, "right_arm");
        left_arm_      = std::make_shared<MoveGroupInterface>(node_, "left_arm");
        right_gripper_ = std::make_shared<MoveGroupInterface>(node_, "right_gripper");
        left_gripper_  = std::make_shared<MoveGroupInterface>(node_, "left_gripper");
        neck_          = std::make_shared<MoveGroupInterface>(node_, "neck");

        setScalingFactors(right_arm_,     1.0, 1.0);
        setScalingFactors(left_arm_,      1.0, 1.0);
        setScalingFactors(right_gripper_, 1.0, 1.0);
        setScalingFactors(left_gripper_,  1.0, 1.0);
        setScalingFactors(neck_,          1.0, 1.0);

        // ── 工作线程启动 ──────────────────────────────
        //   所有规划 & 执行均在此线程串行完成，不阻塞 ROS2 spin
        worker_thread_ = std::thread(&Commander::workerLoop, this);

        // ── 订阅者 ────────────────────────────────────
        open_right_gripper_sub_ = node_->create_subscription<Bool>(
            "open_right_gripper", 10,
            std::bind(&Commander::openRightGripperCallback, this, _1));

        open_left_gripper_sub_ = node_->create_subscription<Bool>(
            "open_left_gripper", 10,
            std::bind(&Commander::openLeftGripperCallback, this, _1));

        right_joint_cmd_sub_ = node_->create_subscription<FloatArray>(
            "right_joint_command", 10,
            std::bind(&Commander::rightJointCmdCallback, this, _1));

        left_joint_cmd_sub_ = node_->create_subscription<FloatArray>(
            "left_joint_command", 10,
            std::bind(&Commander::leftJointCmdCallback, this, _1));

        right_pose_cmd_sub_ = node_->create_subscription<PoseCmd>(
            "right_pose_command", 10,
            std::bind(&Commander::rightPoseCmdCallback, this, _1));

        left_pose_cmd_sub_ = node_->create_subscription<PoseCmd>(
            "left_pose_command", 10,
            std::bind(&Commander::leftPoseCmdCallback, this, _1));

        neck_joint_cmd_sub_ = node_->create_subscription<FloatArray>(
            "neck_joint_command", 10,
            std::bind(&Commander::neckJointCmdCallback, this, _1));

        RCLCPP_INFO(node_->get_logger(), "Commander 已启动，工作线程就绪");
    }

    ~Commander()
    {
        task_queue_.stop();          // 通知工作线程退出
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    // ── 公开运动接口（可在外部直接调用，也可作为任务入队）──

    void goToNamedTarget(const std::shared_ptr<MoveGroupInterface> &mg,
                         const std::string &name)
    {
        mg->setStartStateToCurrentState();
        mg->setNamedTarget(name);
        planAndExecute(mg);
    }

    void goToJointTarget(const std::shared_ptr<MoveGroupInterface> &mg,
                         const std::vector<double> &joints)
    {
        mg->setStartStateToCurrentState();
        mg->setJointValueTarget(joints);
        planAndExecute(mg);
    }

    void goToPoseTarget(const std::shared_ptr<MoveGroupInterface> &arm,
                        double x, double y, double z,
                        double roll, double pitch, double yaw,
                        bool cartesian_path = false,
                        bool relative       = false)
    {
        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        q.normalize();

        geometry_msgs::msg::PoseStamped target_pose;
        target_pose.header.frame_id = "base_link";

        if (!relative) {
            target_pose.pose.position.x    = x;
            target_pose.pose.position.y    = y;
            target_pose.pose.position.z    = z;
            target_pose.pose.orientation.x = q.getX();
            target_pose.pose.orientation.y = q.getY();
            target_pose.pose.orientation.z = q.getZ();
            target_pose.pose.orientation.w = q.getW();
        } else {
            target_pose.pose = arm->getCurrentPose().pose;
            target_pose.pose.position.x += x;
            target_pose.pose.position.y += y;
            target_pose.pose.position.z += z;

            tf2::Quaternion q_current;
            tf2::fromMsg(target_pose.pose.orientation, q_current);
            // q = (q_current * q).normalize();// 本地系
            q = (q * q_current ).normalize();// 世界系

            target_pose.pose.orientation.x = q.getX();
            target_pose.pose.orientation.y = q.getY();
            target_pose.pose.orientation.z = q.getZ();
            target_pose.pose.orientation.w = q.getW();
        }

        arm->setStartStateToCurrentState();

        if (!cartesian_path) {
            arm->setPoseTarget(target_pose);
            planAndExecute(arm);
        } else {
            std::vector<geometry_msgs::msg::Pose> waypoints;
            waypoints.push_back(target_pose.pose);

            moveit_msgs::msg::RobotTrajectory trajectory;
            moveit_msgs::msg::MoveItErrorCodes error_code;
            double fraction = arm->computeCartesianPath(
                waypoints, 0.01, 0.0, trajectory, true, &error_code);

            if (fraction >= 1.0 - 1e-6) {
                arm->execute(trajectory);
            } else {
                RCLCPP_ERROR(node_->get_logger(),
                    "笛卡尔路径规划失败，仅完成 %.2f%%", fraction * 100.0);
            }
        }
    }

    void openGripper(const std::shared_ptr<MoveGroupInterface> &gripper)
    {
        gripper->setStartStateToCurrentState();
        gripper->setNamedTarget("Gripper_open");
        planAndExecute(gripper);
    }

    void closeGripper(const std::shared_ptr<MoveGroupInterface> &gripper)
    {
        gripper->setStartStateToCurrentState();
        gripper->setNamedTarget("Gripper_closed");
        planAndExecute(gripper);
    }

private:

    // ── 工作线程主循环 ────────────────────────────────
    void workerLoop()
    {
        RCLCPP_INFO(node_->get_logger(), "工作线程启动");

        std::function<void()> task;
        while (task_queue_.waitAndPop(task)) {
            try {
                task();   // 串行执行，每次只运行一个运动任务
            } catch (const std::exception &e) {
                RCLCPP_ERROR(node_->get_logger(),
                    "任务执行异常: %s", e.what());
            }
        }

        RCLCPP_INFO(node_->get_logger(), "工作线程退出");
    }

    // ── 工具：规划并执行 ──────────────────────────────
    void planAndExecute(const std::shared_ptr<MoveGroupInterface> &mg)
    {
        MoveGroupInterface::Plan plan;
        bool ok = (mg->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (ok) {
            mg->execute(plan);
        } else {
            RCLCPP_WARN(node_->get_logger(), "运动规划失败，跳过本次执行");
        }
    }

    void setScalingFactors(const std::shared_ptr<MoveGroupInterface> &mg,
                           float vel, float acc)
    {
        mg->setMaxVelocityScalingFactor(vel);
        mg->setMaxAccelerationScalingFactor(acc);
    }

    // ── 订阅回调（仅入队，立即返回）────────────────────

    void openRightGripperCallback(const Bool &msg)
    {
        bool open = msg.data;
        // 按值捕获，避免悬空引用
        task_queue_.push([this, open]() {
            open ? openGripper(right_gripper_) : closeGripper(right_gripper_);
        });
        RCLCPP_DEBUG(node_->get_logger(),
            "右夹爪任务入队（%s），当前队列长度: %zu",
            open ? "open" : "close", task_queue_.size());
    }

    void openLeftGripperCallback(const Bool &msg)
    {
        bool open = msg.data;
        task_queue_.push([this, open]() {
            open ? openGripper(left_gripper_) : closeGripper(left_gripper_);
        });
        RCLCPP_DEBUG(node_->get_logger(),
            "左夹爪任务入队（%s），当前队列长度: %zu",
            open ? "open" : "close", task_queue_.size());
    }

    void rightJointCmdCallback(const FloatArray &msg)
    {
        if (msg.data.size() != 6) {
            RCLCPP_WARN(node_->get_logger(),
                "右臂关节指令长度应为 6，收到 %zu，已忽略", msg.data.size());
            return;
        }
        std::vector<double> joints(msg.data.begin(), msg.data.end());
        task_queue_.push([this, joints]() {
            goToJointTarget(right_arm_, joints);
        });
        RCLCPP_DEBUG(node_->get_logger(),
            "右臂关节任务入队，当前队列长度: %zu", task_queue_.size());
    }

    void leftJointCmdCallback(const FloatArray &msg)
    {
        if (msg.data.size() != 6) {
            RCLCPP_WARN(node_->get_logger(),
                "左臂关节指令长度应为 6，收到 %zu，已忽略", msg.data.size());
            return;
        }
        std::vector<double> joints(msg.data.begin(), msg.data.end());
        task_queue_.push([this, joints]() {
            goToJointTarget(left_arm_, joints);
        });
        RCLCPP_DEBUG(node_->get_logger(),
            "左臂关节任务入队，当前队列长度: %zu", task_queue_.size());
    }

    void rightPoseCmdCallback(const PoseCmd &msg)
    {
        // 将消息字段按值捕获到 lambda，避免消息生命周期问题
        double x = msg.x, y = msg.y, z = msg.z;
        double roll = msg.roll, pitch = msg.pitch, yaw = msg.yaw;
        bool cart = msg.cartesian_path, rel = msg.relative;

        task_queue_.push([this, x, y, z, roll, pitch, yaw, cart, rel]() {
            goToPoseTarget(right_arm_, x, y, z, roll, pitch, yaw, cart, rel);
        });
        RCLCPP_DEBUG(node_->get_logger(),
            "右臂位姿任务入队，当前队列长度: %zu", task_queue_.size());
    }

    void leftPoseCmdCallback(const PoseCmd &msg)
    {
        double x = msg.x, y = msg.y, z = msg.z;
        double roll = msg.roll, pitch = msg.pitch, yaw = msg.yaw;
        bool cart = msg.cartesian_path, rel = msg.relative;

        task_queue_.push([this, x, y, z, roll, pitch, yaw, cart, rel]() {
            goToPoseTarget(left_arm_, x, y, z, roll, pitch, yaw, cart, rel);
        });
        RCLCPP_DEBUG(node_->get_logger(),
            "左臂位姿任务入队，当前队列长度: %zu", task_queue_.size());
    }

    void neckJointCmdCallback(const FloatArray &msg)
    {
        if (msg.data.size() != 2) {
            RCLCPP_WARN(node_->get_logger(),
                "颈部关节指令长度应为 2，收到 %zu，已忽略", msg.data.size());
            return;
        }
        std::vector<double> joints(msg.data.begin(), msg.data.end());
        task_queue_.push([this, joints]() {
            goToJointTarget(neck_, joints);
        });
        RCLCPP_DEBUG(node_->get_logger(),
            "颈部关节任务入队，当前队列长度: %zu", task_queue_.size());
    }

private:
    // ── 成员变量 ──────────────────────────────────────

    std::shared_ptr<rclcpp::Node> node_;

    std::shared_ptr<MoveGroupInterface> right_arm_;
    std::shared_ptr<MoveGroupInterface> left_arm_;
    std::shared_ptr<MoveGroupInterface> right_gripper_;
    std::shared_ptr<MoveGroupInterface> left_gripper_;
    std::shared_ptr<MoveGroupInterface> neck_;

    // 任务队列（线程安全）
    TaskQueue   task_queue_;
    std::thread worker_thread_;

    rclcpp::Subscription<Bool>::SharedPtr       open_right_gripper_sub_;
    rclcpp::Subscription<Bool>::SharedPtr       open_left_gripper_sub_;
    rclcpp::Subscription<FloatArray>::SharedPtr right_joint_cmd_sub_;
    rclcpp::Subscription<FloatArray>::SharedPtr left_joint_cmd_sub_;
    rclcpp::Subscription<FloatArray>::SharedPtr neck_joint_cmd_sub_;
    rclcpp::Subscription<PoseCmd>::SharedPtr    right_pose_cmd_sub_;
    rclcpp::Subscription<PoseCmd>::SharedPtr    left_pose_cmd_sub_;
};

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    // MultiThreadedExecutor：让 ROS2 回调与工作线程真正并行
    auto node = std::make_shared<rclcpp::Node>("Commander");
    auto commander = std::make_shared<Commander>(node);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
