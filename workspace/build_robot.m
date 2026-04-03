function [robot, endEffectorName] = build_robot(L, jlim)
% BUILD_ROBOT  使用 rigidBodyTree 建立 7 自由度串联机械臂模型
%
% 输入:
%   L    - 结构体，包含连杆长度 (单位 m):
%          L.L_base, L.L_shoulder, L.L_upper, L.L_lower, L.L_wrist
%   jlim - 7x2 矩阵，关节上下限 (单位 rad)
%          每行 [lower, upper]
%
% 输出:
%   robot           - rigidBodyTree 对象
%   endEffectorName - 末端执行器 body 名称 (字符串)
%
% 关节轴假设说明 (典型 7DOF 人形协作臂，类似 KUKA iiwa / Franka):
%   Joint1: 肩关节俯仰  —— 绕 Z 轴旋转 (基座竖直轴)
%   Joint2: 肩关节侧摆  —— 绕 Y 轴旋转
%   Joint3: 肩关节周转  —— 绕 Z 轴旋转 (大臂轴线)
%   Joint4: 肘关节屈伸  —— 绕 Y 轴旋转
%   Joint5: 肘关节周转  —— 绕 Z 轴旋转 (小臂轴线)
%   Joint6: 腕关节俯仰  —— 绕 Y 轴旋转
%   Joint7: 腕关节周转  —— 绕 Z 轴旋转 (末端轴线)
%
% 注意: 当前模型基于假设转轴建立。若拿到真实 URDF / DH 参数，
%       可直接替换此函数中的变换矩阵和关节轴定义。

robot = rigidBodyTree('DataFormat', 'row');

% ---- 关节轴定义 ----
% Z-Y-Z-Y-Z-Y-Z 交替排列，是 7DOF 冗余臂最常见的构型
joint_axes = {
    [0 0 1];   % Joint1: Z
    [0 1 0];   % Joint2: Y
    [0 0 1];   % Joint3: Z
    [0 1 0];   % Joint4: Y
    [0 0 1];   % Joint5: Z
    [0 1 0];   % Joint6: Y
    [0 0 1];   % Joint7: Z
};

% ---- 各关节相对于上一关节的固定变换 (平移部分) ----
% 这些变换描述了当关节角为 0 时，子坐标系相对于父坐标系的位姿
joint_transforms = {
    trvec2tform([0, 0, L.L_base]);          % base -> J1: 沿Z抬高基座
    trvec2tform([0, L.L_shoulder, 0]);       % J1 -> J2: 肩部侧向偏移
    trvec2tform([0, 0, L.L_upper]);          % J2 -> J3: 沿Z方向大臂
    trvec2tform([0, 0, 0]);                  % J3 -> J4: 肘关节同位
    trvec2tform([0, 0, L.L_lower]);          % J4 -> J5: 沿Z方向小臂
    trvec2tform([0, 0, 0]);                  % J5 -> J6: 腕关节同位
    trvec2tform([0, 0, 0]);                  % J6 -> J7: 腕周转同位
};

% 末端执行器相对于 Joint7 的固定偏移
ee_transform = trvec2tform([0, 0, L.L_wrist]);

body_names = {'link1','link2','link3','link4','link5','link6','link7'};
joint_names = {'joint1','joint2','joint3','joint4','joint5','joint6','joint7'};
parent_names = {'base','link1','link2','link3','link4','link5','link6'};

% ---- 逐个添加连杆和关节 ----
for i = 1:7
    body = rigidBody(body_names{i});
    jnt  = rigidBodyJoint(joint_names{i}, 'revolute');

    % 设置关节轴
    jnt.JointAxis = joint_axes{i};

    % 设置关节限位 (rad)
    jnt.PositionLimits = jlim(i, :);

    % 设置关节到父坐标系的固定变换
    setFixedTransform(jnt, joint_transforms{i});

    body.Joint = jnt;
    addBody(robot, body, parent_names{i});
end

% ---- 添加末端执行器 (固定关节) ----
ee = rigidBody('end_effector');
ee_jnt = rigidBodyJoint('ee_fixed', 'fixed');
setFixedTransform(ee_jnt, ee_transform);
ee.Joint = ee_jnt;
addBody(robot, ee, 'link7');

endEffectorName = 'end_effector';

fprintf('rigidBodyTree 模型建立完成，共 %d 个关节。\n', 7);
end
