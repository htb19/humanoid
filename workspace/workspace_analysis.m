function workspace_analysis(robot, eeName, jlim, n_samples, L)
% WORKSPACE_ANALYSIS  Monte Carlo 随机采样法分析 7DOF 机械臂工作空间
%
% 输入:
%   robot     - rigidBodyTree 对象
%   eeName    - 末端执行器 body 名称
%   jlim      - 7x2 关节限位矩阵 (rad)
%   n_samples - 采样数量 (建议 >= 10000)
%   L         - 连杆长度结构体 (用于标题显示)
%
% 输出:
%   在命令行打印工作空间范围
%   绘制 3D 散点图及 XY/XZ/YZ 投影
%
% 扩展接口:
%   若需按姿态约束筛选可达工作空间，可在采样循环中加入
%   姿态判断条件，过滤不满足约束的点。

fprintf('正在进行工作空间采样 (%d 组)...\n', n_samples);

% 预分配
positions = zeros(n_samples, 3);

% 随机采样各关节角 (均匀分布，严格遵守上下限)
rng(42);  % 固定随机种子，便于复现
q_samples = zeros(n_samples, 7);
for j = 1:7
    q_samples(:, j) = jlim(j,1) + (jlim(j,2) - jlim(j,1)) * rand(n_samples, 1);
end

% 计算末端位置
for i = 1:n_samples
    T = getTransform(robot, q_samples(i,:), eeName);
    positions(i, :) = T(1:3, 4)';
end

% ---- 统计工作空间范围 ----
x_range = [min(positions(:,1)), max(positions(:,1))];
y_range = [min(positions(:,2)), max(positions(:,2))];
z_range = [min(positions(:,3)), max(positions(:,3))];

fprintf('工作空间范围 (m):\n');
fprintf('  X: [%.4f, %.4f]  跨度: %.4f\n', x_range, diff(x_range));
fprintf('  Y: [%.4f, %.4f]  跨度: %.4f\n', y_range, diff(y_range));
fprintf('  Z: [%.4f, %.4f]  跨度: %.4f\n', z_range, diff(z_range));

% 计算距原点最远距离
max_reach = max(vecnorm(positions, 2, 2));
fprintf('  最大可达距离: %.4f m\n', max_reach);

% ---- 绘图 ----
% 图1: 3D 工作空间散点图
figure('Name', '工作空间 3D 视图', 'Position', [100 100 800 600]);
scatter3(positions(:,1), positions(:,2), positions(:,3), ...
         1, positions(:,3), 'filled');
colormap(jet);
colorbar;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('7-DOF 机械臂工作空间 (Monte Carlo, N=%d)', n_samples));
axis equal; grid on; view(30, 25);

% 图2: 三视图投影
figure('Name', '工作空间投影视图', 'Position', [150 150 1200 400]);

subplot(1,3,1);
scatter(positions(:,1), positions(:,2), 1, positions(:,3), 'filled');
colormap(jet);
xlabel('X (m)'); ylabel('Y (m)');
title('XY 平面投影 (俯视)');
axis equal; grid on;

subplot(1,3,2);
scatter(positions(:,1), positions(:,3), 1, positions(:,2), 'filled');
colormap(jet);
xlabel('X (m)'); ylabel('Z (m)');
title('XZ 平面投影 (侧视)');
axis equal; grid on;

subplot(1,3,3);
scatter(positions(:,2), positions(:,3), 1, positions(:,1), 'filled');
colormap(jet);
xlabel('Y (m)'); ylabel('Z (m)');
title('YZ 平面投影 (正视)');
axis equal; grid on;

sgtitle('工作空间三视图投影');

fprintf('工作空间分析完成。\n');
end
