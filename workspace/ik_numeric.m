function [q_sol, pos_err, exitFlag] = ik_numeric(robot, eeName, T_target, q0, jlim)
% IK_NUMERIC  基于 MATLAB Robotics System Toolbox 的数值逆运动学求解
%
% 输入:
%   robot    - rigidBodyTree 对象
%   eeName   - 末端执行器 body 名称 (字符串)
%   T_target - 4x4 目标齐次变换矩阵
%   q0       - 1x7 初始关节角猜测 (rad)
%   jlim     - 7x2 关节限位矩阵 (rad), 每行 [lower, upper]
%
% 输出:
%   q_sol    - 1x7 求解得到的关节角 (rad)
%   pos_err  - 末端位置误差范数 (m)
%   exitFlag - 求解状态标志 (1=成功, 其他=可能未收敛)
%
% 说明:
%   使用 inverseKinematics 对象，权重向量 [0.1 0.1 0.1 1 1 1]
%   表示姿态权重较低、位置权重较高。可根据需要调整。
%   关节限位已在 build_robot 中设置，IK 求解器会自动遵守。

try
    % 创建 IK 求解器
    ik = inverseKinematics('RigidBodyTree', robot);
    ik.SolverParameters.MaxIterations = 1500;
    ik.SolverParameters.MaxTime = 10;       % 最大求解时间 (秒)

    % 权重: [orientX orientY orientZ posX posY posZ]
    weights = [0.1, 0.1, 0.1, 1, 1, 1];

    % 求解
    [q_sol, solInfo] = ik(eeName, T_target, weights, q0);

    % 计算位置误差
    T_actual = getTransform(robot, q_sol, eeName);
    pos_err = norm(T_target(1:3,4) - T_actual(1:3,4));

    % 判断求解状态
    exitFlag = solInfo.ExitFlag;

    if exitFlag ~= 1
        warning('IK 求解可能未完全收敛 (ExitFlag=%d)，位置误差=%.6f m', ...
                exitFlag, pos_err);
    end

    % 检查关节限位 (额外安全检查)
    for i = 1:7
        if q_sol(i) < jlim(i,1) || q_sol(i) > jlim(i,2)
            warning('Joint%d = %.2f deg 超出限位 [%.2f, %.2f] deg', ...
                    i, rad2deg(q_sol(i)), rad2deg(jlim(i,1)), rad2deg(jlim(i,2)));
        end
    end

catch ME
    warning('IK 求解失败: %s', ME.message);
    q_sol = q0;
    pos_err = Inf;
    exitFlag = -1;
end

end
