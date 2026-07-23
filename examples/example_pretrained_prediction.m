function example_pretrained_prediction

clc;
close all;

modelFile = 'pretrained_KC_CNN_LSTM_multimodal.mat';
dataFile  = 'example_test_samples_checked.mat';

%% ================= 加载模型和数据 =================
if ~isfile(modelFile)
    error('找不到模型文件：%s', modelFile);
end

if ~isfile(dataFile)
    error('找不到测试数据文件：%s', dataFile);
end

M = load(modelFile);
D = load(dataFile);

%% ================= 检查必要变量 =================
requiredModelVars = { ...
    'net', ...
    'Vs_min', 'Vs_max', ...
    'H_min', 'H_max', ...
    'outputVsDim', 'outputHDim'};

for k = 1:numel(requiredModelVars)
    if ~isfield(M, requiredModelVars{k})
        error('模型文件缺少变量：%s', requiredModelVars{k});
    end
end

requiredDataVars = { ...
    'X_public', ...
    'Vs_true_public', ...
    'H_true_public'};

for k = 1:numel(requiredDataVars)
    if ~isfield(D, requiredDataVars{k})
        error('测试数据文件缺少变量：%s', requiredDataVars{k});
    end
end

%% ================= 构造网络输入 =================
nSample = numel(D.X_public);
outputTotal = M.outputVsDim + M.outputHDim;

XInput = cat(3, D.X_public{:});
dlX = dlarray(XInput, 'CTB');

%% ================= 网络预测 =================
dlYPred = forward(M.net, dlX);
YPredRaw = gather(extractdata(dlYPred));

if size(YPredRaw,1) == outputTotal && ...
        size(YPredRaw,2) == nSample

    YPredTest_norm = double(YPredRaw.');

elseif size(YPredRaw,1) == nSample && ...
        size(YPredRaw,2) == outputTotal

    YPredTest_norm = double(YPredRaw);

else
    error(['网络输出尺寸无法识别。当前尺寸为%s，' ...
           '期望为%d×%d或%d×%d。'], ...
        mat2str(size(YPredRaw)), ...
        outputTotal, nSample, ...
        nSample, outputTotal);
end

%% ================= 反归一化 =================
YPredTest_Vs = ...
    YPredTest_norm(:,1:M.outputVsDim) .* ...
    (M.Vs_max - M.Vs_min) + M.Vs_min;

YPredTest_H = ...
    YPredTest_norm(:, ...
    M.outputVsDim+1:M.outputVsDim+M.outputHDim) .* ...
    (M.H_max - M.H_min) + M.H_min;

%% ================= 真实参数 =================
YTrueTest_Vs = double(D.Vs_true_public);
YTrueTest_H  = double(D.H_true_public);

assert(isequal(size(YPredTest_Vs),size(YTrueTest_Vs)), ...
    '预测Vs与真实Vs尺寸不一致');

assert(isequal(size(YPredTest_H),size(YTrueTest_H)), ...
    '预测H与真实H尺寸不一致');

%% ================= RMSE =================
diff_Vs = double(YPredTest_Vs) - double(YTrueTest_Vs);
diff_H  = double(YPredTest_H)  - double(YTrueTest_H);

rmse_Vs_perSample = sqrt( ...
    sum(diff_Vs.^2,2) ./ size(diff_Vs,2));

rmse_H_perSample = sqrt( ...
    sum(diff_H.^2,2) ./ size(diff_H,2));

overall_rmse_Vs = mean(rmse_Vs_perSample,'omitnan');
overall_rmse_H  = mean(rmse_H_perSample,'omitnan');

%% ================= 只输出结果 =================
fprintf('\nVs平均逐样本RMSE = %.6f m/s\n',overall_rmse_Vs);
fprintf('H 平均逐样本RMSE = %.6f m\n',overall_rmse_H);

%% ================= 保存结果 =================
save('prediction_samples_results.mat', ...
    'YPredTest_norm', ...
    'YPredTest_Vs', ...
    'YPredTest_H', ...
    'YTrueTest_Vs', ...
    'YTrueTest_H', ...
    'rmse_Vs_perSample', ...
    'rmse_H_perSample', ...
    'overall_rmse_Vs', ...
    'overall_rmse_H');

end