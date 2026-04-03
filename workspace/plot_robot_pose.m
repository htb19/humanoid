function plot_robot_pose(robot, q, eeName, titleStr)
% PLOT_ROBOT_POSE  可视化机械臂在给定关节角下的姿态
%
% 输入:
%   robot    - rigidBodyTree 对象
%   q        - 1x7 关节角向量 (rad)
%   eeName   - 末端执行器 body 名称
%   titleStr - 图标题字符串 (可选)
%
% 功能:
%   1. 使用 show() 显示机械臂 3D 姿态
%   2. 在末端执行器处标注坐标系和位置信息
%   3. 显示各关节角数值

if nargin < 4
    titleStr = '机械臂姿态';
end

% 获取末端位姿
T_ee = getTransform(robot, q, eeName);
pos_ee = T_ee(1:3, 4)';
eul_ee = rotm2eul(T_ee(1:3,1:3), 'ZYX');

% 绘制机械臂
figure('Name', titleStr, 'Position', [350 100 900 700]);
show(robot, q, 'Frames', 'on', 'PreservePlot', false);
hold on;

% 在末端标注位置
plot3(pos_ee(1), pos_ee(2), pos_ee(3), 'rp', 'MarkerSize', 15, ...
      'MarkerFaceColor', 'r', 'DisplayName', '末端位置');

% 添加文字标注
text(pos_ee(1)+0.02, pos_ee(2)+0.02, pos_ee(3)+0.02, ...
     sprintf('EE: [%.3f, %.3f, %.3f] m', pos_ee), ...
     'FontSize', 9, 'Color', 'r');

hold off;

% 设置图形属性
title(sprintf('%s\n末端位置: [%.3f, %.3f, %.3f] m  |  ZYX欧拉角: [%.1f, %.1f, %.1f] deg', ...
      titleStr, pos_ee, rad2deg(eul_ee)));
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; grid on; view(135, 25);
light('Position', [1 1 1]);

% 命令行输出关节角信息
fprintf('姿态信息:\n');
fprintf('  关节角 (deg): [%s]\n', num2str(rad2deg(q), '%.1f '));
fprintf('  末端位置 (m): [%.4f, %.4f, %.4f]\n', pos_ee);
fprintf('  ZYX欧拉角 (deg): [%.2f, %.2f, %.2f]\n', rad2deg(eul_ee));

end
