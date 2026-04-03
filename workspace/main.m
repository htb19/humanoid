%% 
%% ===================================================================
%  main.m  —— 7自由度串联机械臂 运动学与工作空间仿真 主程序
%  作者：Kiro AI Assistant
%  说明：当前模型基于"典型7DOF人形协作臂"假设转轴建立，
%        若后续拿到真实 URDF / DH 参数，可直接替换 build_robot.m
%        和 fk_dh.m 中的参数配置区。
% ====================================================================
clear; clc; close all;
fprintf('====== 7-DOF 机械臂运动学仿真 ======\n\n');

%% ==================== 参数配置区域（可手动修改） ====================
% 连杆长度 (mm)，程序内部会自动转为 m
link_lengths_mm.L_base    = 100.0;   % 基座到肩关节高度
link_lengths_mm.L_shoulder = 0.0;    % 肩关节偏移（侧向）
link_lengths_mm.L_upper   = 227.0;   % 大臂长度
link_lengths_mm.L_lower   = 267.0;   % 小臂长度
link_lengths_mm.L_wrist   = 226.5;   % 末端（腕到工具）长度
% 总长校验: 227 + 267 + 226.5 = 720.5 (厂家标称717.5，差异来自关节间隙)

% 关节限位 (单位：度)
joint_limits_deg = [
    -170,  170;   % Joint1: 肩关节俯仰 (绕Z轴)
    -105,  105;   % Joint2: 肩关节侧摆 (绕Y轴)
    -170,  170;   % Joint3: 肩关节周转 (绕Z轴)
       0,  130;   % Joint4: 肘关节屈伸 (绕Y轴)
    -170,  170;   % Joint5: 肘关节周转 (绕Z轴)
    -120,  120;   % Joint6: 腕关节俯仰 (绕Y轴)
    -170,  170;   % Joint7: 腕关节周转 (绕Z轴)
];

% 转换为 m 和 rad
link_lengths.L_base     = link_lengths_mm.L_base    / 1000;
link_lengths.L_shoulder = link_lengths_mm.L_shoulder / 1000;
link_lengths.L_upper    = link_lengths_mm.L_upper    / 1000;
link_lengths.L_lower    = link_lengths_mm.L_lower    / 1000;
link_lengths.L_wrist    = link_lengths_mm.L_wrist    / 1000;

joint_limits_rad = deg2rad(joint_limits_deg);

%% ==================== 1. 建立机器人模型 ====================
fprintf('--- 1. 建立 rigidBodyTree 模型 ---\n');
[robot, endEffectorName] = build_robot(link_lengths, joint_limits_rad);
showdetails(robot);
fprintf('\n');

%% ==================== 2. 正运动学验证 ====================
fprintf('--- 2. 正运动学验证 ---\n');
% 测试关节角 (度) -> 弧度
q_test_deg = [0, 30, 0, 60, 0, -30, 0];
q_test = deg2rad(q_test_deg);

% 方法A: rigidBodyTree FK
T_rbt = getTransform(robot, q_test, endEffectorName);
pos_rbt = T_rbt(1:3, 4);

% 方法B: 手写 DH FK
T_dh = fk_dh(q_test, link_lengths);
pos_dh = T_dh(1:3, 4);

fprintf('测试关节角 (deg): [%s]\n', num2str(q_test_deg));
fprintf('rigidBodyTree FK 末端位置 (m): [%.4f, %.4f, %.4f]\n', pos_rbt);
fprintf('手写 DH FK      末端位置 (m): [%.4f, %.4f, %.4f]\n', pos_dh);
fprintf('两种方法位置差 (m): %.6f\n', norm(pos_rbt - pos_dh));

% 欧拉角输出
eul = rotm2eul(T_rbt(1:3,1:3), 'ZYX');
fprintf('末端姿态 ZYX 欧拉角 (deg): [%.2f, %.2f, %.2f]\n\n', rad2deg(eul));

%% ==================== 3. 逆运动学验证 ====================
fprintf('--- 3. 逆运动学验证 ---\n');
% 3组可达目标点 (位置 m + 姿态用单位矩阵)
targets = {
    trvec2tform([0.40,  0.00, 0.30]);
    trvec2tform([0.30,  0.20, 0.40]);
    trvec2tform([0.20, -0.15, 0.50]);
};

q0 = zeros(1, 7);  % IK 初始猜测

for i = 1:length(targets)
    T_target = targets{i};
    [q_sol, pos_err, info] = ik_numeric(robot, endEffectorName, ...
                                         T_target, q0, joint_limits_rad);
    % FK 回代验证
    T_check = getTransform(robot, q_sol, endEffectorName);
    pos_target = T_target(1:3, 4)';
    pos_actual = T_check(1:3, 4)';
    err = norm(pos_target - pos_actual);

    fprintf('目标点 %d: [%.3f, %.3f, %.3f] m\n', i, pos_target);
    fprintf('  求解关节角 (deg): [%s]\n', num2str(rad2deg(q_sol), '%.1f '));
    fprintf('  FK回代位置 (m):   [%.4f, %.4f, %.4f]\n', pos_actual);
    fprintf('  位置误差 (m):     %.6f\n', err);
    fprintf('  求解状态:         %d\n\n', info);
end

%% ==================== 4. 工作空间分析 ====================
fprintf('--- 4. 工作空间分析 ---\n');
n_samples = 10000;
workspace_analysis(robot, endEffectorName, joint_limits_rad, n_samples, link_lengths);

%% ==================== 5. 轨迹测试 ====================
fprintf('\n--- 5. 轨迹跟踪测试 ---\n');
trajectory_test(robot, endEffectorName, joint_limits_rad, link_lengths);

%% ==================== 6. 显示机械臂姿态 ====================
fprintf('\n--- 6. 机械臂姿态可视化 ---\n');
plot_robot_pose(robot, q_test, endEffectorName, '测试姿态 q=[0,30,0,60,0,-30,0] deg');

fprintf('\n====== 仿真完成 ======\n');
