function T = fk_dh(q, L)
% FK_DH  基于 Modified DH (Craig) 参数的正运动学
%
% 输入:
%   q - 1x7 或 7x1 关节角向量 (单位 rad)
%   L - 结构体，包含连杆长度 (单位 m):
%       L.L_base, L.L_shoulder, L.L_upper, L.L_lower, L.L_wrist
%
% 输出:
%   T - 4x4 齐次变换矩阵，表示末端执行器相对于基座的位姿
%
% Modified DH 参数定义 (Craig 约定):
%   alpha_{i-1} : 绕 X_{i-1} 轴旋转的角度
%   a_{i-1}     : 沿 X_{i-1} 轴的平移距离
%   d_i         : 沿 Z_i 轴的平移距离
%   theta_i     : 绕 Z_i 轴旋转的角度 (= q_i + offset)
%
% 关节轴假设: Z-Y-Z-Y-Z-Y-Z (与 build_robot.m 一致)
%   为了在 MDH 中实现 Y 轴旋转，需要在相邻坐标系间加入
%   alpha = ±pi/2 来"翻转"Z轴方向。
%
% 注意: 当前 DH 参数基于假设建立，若拿到真实参数可直接替换下方表格。

q = q(:)';  % 确保行向量

% ===================== MDH 参数表 (可修改区) =====================
%        alpha_{i-1}    a_{i-1}       d_i          theta_offset
%        (rad)          (m)           (m)          (rad)
MDH = [
    0,              0,            L.L_base,     0;       % Joint1: 基座->肩(Z轴)
   -pi/2,           0,            L.L_shoulder, 0;       % Joint2: Z->Y 翻转(Y轴)
    pi/2,           0,            L.L_upper,    0;       % Joint3: Y->Z 翻转(Z轴)
   -pi/2,           0,            0,            0;       % Joint4: Z->Y 翻转(Y轴)
    pi/2,           0,            L.L_lower,    0;       % Joint5: Y->Z 翻转(Z轴)
   -pi/2,           0,            0,            0;       % Joint6: Z->Y 翻转(Y轴)
    pi/2,           0,            0,            0;       % Joint7: Y->Z 翻转(Z轴)
];
% =================================================================

T = eye(4);

for i = 1:7
    alpha = MDH(i, 1);
    a     = MDH(i, 2);
    d     = MDH(i, 3);
    theta = q(i) + MDH(i, 4);

    % Modified DH 变换矩阵 (Craig 约定)
    ct = cos(theta); st = sin(theta);
    ca = cos(alpha); sa = sin(alpha);

    A_i = [
        ct,      -st,      0,     a;
        st*ca,    ct*ca,  -sa,   -sa*d;
        st*sa,    ct*sa,   ca,    ca*d;
        0,        0,       0,     1;
    ];

    T = T * A_i;
end

% 末端执行器偏移 (沿最后一个坐标系的 Z 轴)
T_ee = trvec2tform([0, 0, L.L_wrist]);
T = T * T_ee;

end
