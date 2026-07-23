clear; close all; clc;
StartTime = clock;

%% ================= 参数设置 =================
N_total = 30000;          
tau = 0.9;              

freq = [];
flen = length(freq);

outputVsDim = 4;
outputHDim  = 4; 
outputTotal = outputVsDim + outputHDim;

Nfreq_base = 17;  
Nfreq_2nd  = flen - Nfreq_base; 


%% ================= 生成速度模型 =================
[Vs_all, Vp_all, den_all, H_all] = shijigenerate_velocity_models(N_total, tau);
fprintf('✅ 全量速度模型生成耗时 %.2f 秒\n', etime(clock,StartTime));

%% ================= 正演计算 =================
StartTime = clock;
pv_all  = zeros(N_total, flen);  
pv_all2 = zeros(N_total, flen); 

parfor i = 1:N_total
    vs_i  = Vs_all(i,:);         
    vp_i  = Vp_all(i,:);         
    rho_i = den_all(i,:);       
    h_i   = H_all(i,:);           

    pv_i  = calcmulti(freq, vs_i, h_i, vp_i, rho_i);  
    pv_all(i,:) = pv_i(:,1)';

    pv_i2 = calcmulti2(freq, vs_i, h_i, vp_i, rho_i);  
    pv_all2(i,:) = pv_i2(:,1)';
end

fprintf('✅ 全量频散正演完成，耗时 %.2f 秒\n', etime(clock,StartTime));
 Vs_half=600; rho_half=2000; rho_all=den_all;
  nPlot = 10;  
    idx = randperm(N_total, nPlot);  
    
    % 绘制对应的Vs剖面对比
    draw_vs_stairs_comparison(Vs_all, H_all, nPlot, true);
    
    % 分开绘制基阶和一阶频散曲线
    draw_dispersion_curves(freq, pv_all, pv_all2, idx);
    
%% ================= 剔除基阶含0样本 =================
invalid_idx = any(pv_all==0,2);
num_invalid = sum(invalid_idx);
fprintf('⚠️ 剔除 %d 个基阶含 0 的样本，占总数 %.2f%%\n', num_invalid, num_invalid/N_total*100);

valid_idx = ~invalid_idx;
pv_all  = pv_all(valid_idx,:);
pv_all2 = pv_all2(valid_idx,:);
Vs_all  = Vs_all(valid_idx,:);
Vp_all  = Vp_all(valid_idx,:);
den_all = den_all(valid_idx,:);
H_all   = H_all(valid_idx,:);
N_valid = sum(valid_idx);
fprintf('✅ 剩余有效样本数: %d\n', N_valid);

%% ================= 数据准备 =================
rng('shuffle');

% 划分训练/验证集
idx_perm = randperm(N_valid);
nTrainFull = floor(0.7 * N_valid);
nValFull   = N_valid - nTrainFull;

trainIdxFull = idx_perm(1:nTrainFull);
valIdxFull   = idx_perm(nTrainFull+1:end);

fprintf('✅ 样本划分完成：训练 %d，验证 %d\n', nTrainFull, nValFull);

% 定义归一化函数
normalize   = @(x, xmin, xmax) (x - xmin)./(xmax - xmin + eps);
denormalize = @(x, xmin, xmax) x.*(xmax - xmin) + xmin;

% ================= 归一化参数 =================
Vs_min = min(Vs_all(:,1:outputVsDim),[],'all'); 
Vs_max = max(Vs_all(:,1:outputVsDim),[],'all'); 
H_min  = min(H_all(:,1:outputHDim),[],'all');  
H_max  = max(H_all(:,1:outputHDim),[],'all');

pv_min  = min(pv_all,[],1);
pv_max  = max(pv_all,[],1);
pv2_min = min(pv_all2,[],1);
pv2_max = max(pv_all2,[],1);

% ================= 构造基阶/二阶掩码 =================
mask_base = zeros(1, flen);
mask_base(1:Nfreq_base) = 1;

mask_2nd = zeros(1, flen);
mask_2nd(Nfreq_base+1:end) = 1;

% ================= 构造网络输入 =================
pv_norm_all  = normalize(pv_all, pv_min, pv_max);
pv2_norm_all = normalize(pv_all2, pv2_min, pv2_max);

XFull = cell(N_valid,1);
for i = 1:N_valid
    XFull{i} = [pv_norm_all(i,:) .* mask_base;     
                pv2_norm_all(i,:) .* mask_2nd];   
end

% ================= 构造归一化输出 =================
YNormFull = [normalize(Vs_all(:,1:outputVsDim), Vs_min, Vs_max), ...
             normalize(H_all(:,1:outputHDim), H_min, H_max)];

% 划分训练/验证集
XTrainFull = XFull(trainIdxFull);
YTrainFull = YNormFull(trainIdxFull,:);
XValFull   = XFull(valIdxFull);
YValFull   = YNormFull(valIdxFull,:);

%% ===== 构造自定义损失层 =====
lossLayer = customRegressionLayer( ...
    'custom_loss', ...
    Vs_min, Vs_max, ...
    H_min,  H_max, ...
    rho_all, ...
    Vs_half, rho_half, ...
    false);         % debug
%% ================= CNN-LSTM 网络 =================
layers = [
    sequenceInputLayer(2, ...
        'Name','input', ...
        'Normalization','none', ...
        'MinLength',flen)

    convolution1dLayer(3,64,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling1dLayer(2,'Stride',2,'Name','pool1')

    convolution1dLayer(3,128,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling1dLayer(2,'Stride',2,'Name','pool2')

    convolution1dLayer(3,256,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    maxPooling1dLayer(2,'Stride',2,'Name','pool3')

    lstmLayer(128,'OutputMode','last','Name','lstm')

    fullyConnectedLayer(outputTotal,'Name','fc')

    lossLayer         
];



%% ================= 训练网络 =================
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64, ...
    'ExecutionEnvironment','cpu', ...
    'Shuffle','never', ...
    'ValidationData',{XValFull,YValFull}, ...
    'ValidationFrequency',50, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'InitialLearnRate',1e-3);

[net, info] = trainNetwork(XTrainFull, YTrainFull, layers, options);

% 保存训练和验证 loss
save('loss_historysedj.mat','info');
fprintf('训练和验证 loss 已保存到 loss_history.mat\n');
