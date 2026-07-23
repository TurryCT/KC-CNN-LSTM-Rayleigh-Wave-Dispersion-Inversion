
clear; close all; clc;
StartTime = clock;
% =================== 基本参数 ===================
fmin = 2; df = 0.5; fmax = 81.5;
freq = fmin:df:fmax;

%VS_true = [200 250 300 400]; 
%VP_true = [431 532 681 943]; 
%H_true  = [2 3 5]; 
%den = [2000 2000 2000 2000]; 

pv = calcbase(freq, VS_true, H_true, VP_true, den);  % 理论频散曲线

% =================== 反演参数 ===================
Npop   = 100;                   % 种群个体数
Max_it = 100;                   % 最大迭代次数
%lb     = [150 187.5 225 300 1.5 2.5 4.5 323.25 399 510.75 707.25];
%ub     = [250 312.5 375 500 2.5 3.5 5.5 538.75 665 851.25 1178.75];

Nrun   = 20;                     % 多次运行次数
nD = length(lb);                 % 搜索维度

% =================== 存储多次结果 ===================
Leader_pos_all   = zeros(Nrun, nD);
Leader_score_all = zeros(Nrun,1);
Convergence_all  = zeros(Nrun, Max_it);   % 保存每次收敛曲线

% =================== 多次运行 WOA ===================
for run_idx = 1:Nrun
    fprintf('第 %d 次 WOA 运行...\n', run_idx);
    [Leader_pos, Leader_score, ~, Convergence_curve] = ...
        WOA_inversion(freq, VS_true, VP_true, H_true, den, Npop, Max_it, lb, ub);
    
    Leader_pos_all(run_idx,:)   = Leader_pos;
    Leader_score_all(run_idx)   = Leader_score;
    Convergence_all(run_idx,:)  = Convergence_curve;
end

% =================== 取平均结果 ===================
Leader_pos_mean   = mean(Leader_pos_all, 1);
Leader_score_mean = mean(Leader_score_all);
Convergence_mean  = mean(Convergence_all, 1);  

% =================== 绘制平均收敛曲线 ===================
figure;
plot(1:Max_it, Convergence_all', 'k--', 'LineWidth', 0.5); hold on; 
plot(1:Max_it, Convergence_mean, 'r-', 'LineWidth', 2);              
xlabel('迭代次数'); ylabel('最优适应度');
title(sprintf('WOA 多次运行平均收敛曲线 (%d 次运行)', Nrun));
legend('单次运行','平均曲线','Location','northeast');
grid on;

% =================== 绘制平均频散曲线 ===================
pv_mean = calcbase(freq, Leader_pos_mean(1:4), Leader_pos_mean(5:7), Leader_pos_mean(8:11), den);

figure; 
plot(freq, pv, 'k-', 'LineWidth', 2); hold on;
plot(freq, pv_mean, 'r--', 'LineWidth', 2);
xlabel('频率 (Hz)'); ylabel('相速度 (m/s)');
legend('理论频散曲线', '平均WOA结果');
title(sprintf('频散曲线对比 (%d次平均)', Nrun));
grid on;

EndTime = clock;
fprintf('总耗时: %.2f 秒\n', etime(EndTime, StartTime));
