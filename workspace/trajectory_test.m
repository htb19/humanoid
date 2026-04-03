function trajectory_test(robot, eeName, jlim, L)
% TRAJECTORY_TEST  轨迹跟踪测试：圆形轨迹 + IK 求解 + 误差分析
%
% 输入:
%   robot  - rigidBodyTree 对象
%   eeName - 末端执行器 body 名称
%   jlim   - 7x2 关节限位矩阵 (rad)
%   L      - 连杆长度结构体
%
% 功能:
%   1. 在机械臂前方生成一个圆形轨迹
%   2. 对每个轨迹点做 IK 求解
%   3. 绘制期望轨迹与实际轨迹对比
%   4. 绘制各关节角随轨迹点变化曲线
%   5. 统计平均误差和最大误差

% ---- 轨迹参数 ----
n_points = 50;                    % 轨迹点数
center = [0.35, 0.0, 0.35];      % 圆心位置 (m)，位于机械臂前方
radius = 0.08;                    % 圆半径 (m)
theta_traj = linspace(0, 2*pi, n_points);

% 生成圆形轨迹点 (在 YZ 平面上的圆，X 固定)
desired_pos = zeros(n_points, 3);
for k = 1:n_points
    desired_pos(k, :) = center + [0, radius*cos(theta_traj(k)), radius*sin(theta_traj(k))];
end

% ---- IK 求解 ----
fprintf('正在对 %d 个轨迹点进行 IK 求解...\n', n_points);
actual_pos = zeros(n_points, 3);
q_traj = zeros(n_points, 7);
pos_errors = zeros(n_points, 1);
ik_flags = zeros(n_points, 1);

q_current = zeros(1, 7);  % 初始猜测，后续用上一步解作为初值

for k = 1:n_points
    % 构造目标位姿 (位置 + 单位姿态)
    T_target = trvec2tform(desired_pos(k, :));

    % IK 求解 (用上一步的解作为初始值，提高连续性)
    [q_sol, err, flag] = ik_numeric(robot, eeName, T_target, q_current, jlim);

    q_traj(k, :) = q_sol;
    pos_errors(k) = err;
    ik_flags(k) = flag;

    % FK 回代
    T_actual = getTransform(robot, q_sol, eeName);
    actual_pos(k, :) = T_actual(1:3, 4)';

    % 更新初始猜测
    q_current = q_sol;
end

% ---- 误差统计 ----
mean_err = mean(pos_errors);
max_err  = max(pos_errors);
fprintf('轨迹跟踪误差统计:\n');
fprintf('  平均位置误差: %.6f m (%.4f mm)\n', mean_err, mean_err*1000);
fprintf('  最大位置误差: %.6f m (%.4f mm)\n', max_err, max_err*1000);
fprintf('  IK 成功率:    %d / %d\n', sum(ik_flags == 1), n_points);

% ---- 图1: 期望轨迹 vs 实际轨迹 (3D) ----
figure('Name', '轨迹跟踪对比', 'Position', [200 200 800 600]);
plot3(desired_pos(:,1), desired_pos(:,2), desired_pos(:,3), ...
      'b-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '期望轨迹');
hold on;
plot3(actual_pos(:,1), actual_pos(:,2), actual_pos(:,3), ...
      'r--x', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '实际轨迹');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('末端轨迹跟踪: 期望 vs 实际');
legend('Location', 'best');
axis equal; grid on; view(30, 25);
hold off;

% ---- 图2: 位置误差曲线 ----
figure('Name', '轨迹跟踪误差', 'Position', [250 250 800 400]);
subplot(2,1,1);
plot(1:n_points, pos_errors * 1000, 'r-', 'LineWidth', 1.5);
xlabel('轨迹点序号'); ylabel('位置误差 (mm)');
title('各轨迹点位置误差');
grid on;
yline(mean_err*1000, 'b--', sprintf('平均: %.4f mm', mean_err*1000), ...
      'LineWidth', 1, 'LabelHorizontalAlignment', 'left');

% ---- 图3: 各关节角变化曲线 ----
subplot(2,1,2);
joint_labels = {'J1','J2','J3','J4','J5','J6','J7'};
hold on;
colors = lines(7);
for j = 1:7
    plot(1:n_points, rad2deg(q_traj(:,j)), '-', ...
         'Color', colors(j,:), 'LineWidth', 1.2, 'DisplayName', joint_labels{j});
end
hold off;
xlabel('轨迹点序号'); ylabel('关节角 (deg)');
title('各关节角随轨迹点变化');
legend('Location', 'eastoutside');
grid on;

% ---- 图4: XYZ 分量对比 ----
figure('Name', '轨迹 XYZ 分量', 'Position', [300 300 800 600]);
axis_labels = {'X', 'Y', 'Z'};
for dim = 1:3
    subplot(3,1,dim);
    plot(1:n_points, desired_pos(:,dim)*1000, 'b-', 'LineWidth', 1.5, ...
         'DisplayName', '期望');
    hold on;
    plot(1:n_points, actual_pos(:,dim)*1000, 'r--', 'LineWidth', 1.5, ...
         'DisplayName', '实际');
    hold off;
    xlabel('轨迹点序号');
    ylabel(sprintf('%s (mm)', axis_labels{dim}));
    title(sprintf('%s 方向轨迹对比', axis_labels{dim}));
    legend('Location', 'best');
    grid on;
end
sgtitle('末端位置各分量对比');

fprintf('轨迹测试完成。\n');
end
