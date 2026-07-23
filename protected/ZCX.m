    clear; close all; clc;
    StartTime = clock;
    %% === 参数设置 ===
    N_total = 100000;    
    N_sample = 30000;   
    tau = 0.9;          
    fmin = 2; df = 0.5; fmax = 81.5;
    freq = fmin:df:fmax;
    flen = length(freq);
    outputVsDim = 10;
    outputHDim  = 10;
    outputTotal = outputVsDim + outputHDim;
    %% === 生成速度模型（全量样本） ===
    [Vs_all, Vp_all, den_all, H_all] = XQDHgenerate_velocity_models(N_total, tau);
    %% === 正演计算（基阶+一阶） ===
    StartTime = clock;
    pv_all  = zeros(N_total, flen);  
    pv_all2 = zeros(N_total, flen); 
    parfor i = 1:N_total
        vs_i = Vs_all(i,:);         
        vp_i = Vp_all(i,:);         
        rho_i = den_all(i,:);       
        h_i = H_all(i,:);           
    
        pv_i = calcmulti(freq, vs_i, h_i, vp_i, rho_i);  
        pv_all(i,:) = pv_i(:,1)';
    
        pv_i2 = calcmulti2(freq, vs_i, h_i, vp_i, rho_i);  
        pv_all2(i,:) = pv_i2(:,1)';
    end
    
    mask = ones(size(pv_all2));
    mask(pv_all2==0 | isnan(pv_all2)) = 0;
    fprintf('✅ 全量频散正演完成，耗时 %.2f 秒\n', etime(clock,StartTime));
    nPlot = 10;  
    idx = randperm(N_total, nPlot);  
    % 绘制对应的Vs剖面对比
    draw_vs_stairs_comparison(Vs_all, H_all, nPlot, true);
    % 分开绘制基阶和一阶频散曲线
    draw_dispersion_curves(freq, pv_all, pv_all2, idx);
    
    
    %% === 剔除基阶含0样本 ===
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
    mask    = mask(valid_idx,:);
    N_valid = sum(valid_idx);
    fprintf('✅ 剩余有效样本数: %d\n', N_valid);
     %noise = 0.1;

% 对基阶频散添加 ±5% 随机噪音
%pv_all  = pv_all + 2*(0.5 - rand(size(pv_all))) .* pv_all .* noise;

% 对一阶频散添加 ±5% 随机噪音
%pv_all2 = pv_all2 + 2*(0.5 - rand(size(pv_all2))) .* pv_all2 .* noise;


    %% === 随机挑选样本用于QPSO寻优 ===
    rng('shuffle');
    sampleIdx = randperm(N_valid, N_sample);
    
    X_sample  = pv_all(sampleIdx,:);
    X2_sample = pv_all2(sampleIdx,:);
    Vs_sample = Vs_all(sampleIdx,:);
    H_sample  = H_all(sampleIdx,:);
    den_sample = den_all(sampleIdx,:);
    mask_sample = mask(sampleIdx,:);
    
    %% === 划分训练/验证集（寻优样本） ===
    nTrain = floor(0.8*N_sample);
    trainIdx = 1:nTrain;
    valIdx   = nTrain+1:N_sample;
    
    rho_all_train = den_sample(trainIdx,:);
  
    %% === 归一化函数 ===
    normalize = @(x, xmin, xmax) (x - xmin)./(xmax - xmin + eps);
    denormalize = @(x, xmin, xmax) x.*(xmax - xmin) + xmin;
    
    %% === 归一化参数 ===
    Vs_use = Vs_sample(:,1:outputVsDim);
    Vs_min = min(Vs_use(trainIdx,:),[],'all'); 
    Vs_max = max(Vs_use(trainIdx,:),[],'all'); 
    
    H_use  = H_sample(:,1:outputHDim);
    H_min  = min(H_use(trainIdx,:),[],'all');  
    H_max  = max(H_use(trainIdx,:),[],'all');
    
    pv_min  = min(X_sample(trainIdx,:),[],1);
    pv_max  = max(X_sample(trainIdx,:),[],1);
    pv2_min = min(X2_sample(trainIdx,:),[],1);
    pv2_max = max(X2_sample(trainIdx,:),[],1);
    
    %% === 构造输入/输出（寻优样本） ===
    XTrain = cell(nTrain,1); XVal = cell(length(valIdx),1);
    for i = 1:nTrain
        base = normalize(X_sample(trainIdx(i),:), pv_min, pv_max);
        first = normalize(X2_sample(trainIdx(i),:), pv2_min, pv2_max);
        first_masked = first .* mask_sample(trainIdx(i),:);
        XTrain{i} = [base; first_masked];
    end
    for i = 1:length(valIdx)
        base = normalize(X_sample(valIdx(i),:), pv_min, pv_max);
        first = normalize(X2_sample(valIdx(i),:), pv2_min, pv2_max);
        first_masked = first .* mask_sample(valIdx(i),:);
        XVal{i} = [base; first_masked];
    end
    
    YTrain = [normalize(Vs_use(trainIdx,:), Vs_min, Vs_max), normalize(H_use(trainIdx,:), H_min, H_max)];
    YVal   = [normalize(Vs_use(valIdx,:), Vs_min, Vs_max), normalize(H_use(valIdx,:), H_min, H_max)];
    
    vs_half = 1200; rho_half = 2000;
    
    %% === 调用 QPSO 寻优超参数 ===
    customLayer = customRegressionLayer('normMSEWeighted',Vs_min,Vs_max,H_min,H_max,rho_all_train,vs_half, rho_half,true);
    [best_params, ~, ~] = runQPSO(XTrain, YTrain, XVal, YVal, freq, outputTotal, customLayer);
 %% ==========================================================================
 fprintf('--- 重置 GPU ---\n');
%gpuDeviceCount       % 可选，查看可用 GPU 
%gpuDevice(1);        % 重置并启用第1个 GPU
%parallel.gpu.enableCUDAForwardCompatibility(true)
  rho_all=den_all; 
rng('shuffle');
idx_perm = 1:N_valid; 
nTrainFull = floor(0.7 * N_valid);
nValFull   = floor(0.2 * N_valid);
nTestFull  = N_valid - nTrainFull - nValFull;
trainIdxFull = idx_perm(1:nTrainFull);
valIdxFull   = idx_perm(nTrainFull+1 : nTrainFull+nValFull);
testIdxFull  = idx_perm(nTrainFull+nValFull+1 : end);
    % 定义归一化函数
    normalize = @(x, xmin, xmax) (x - xmin)./(xmax - xmin + eps);
    denormalize = @(x, xmin, xmax) x.*(xmax - xmin) + xmin;
    
    % 全量归一化参数
    Vs_min = min(Vs_all(:,1:outputVsDim),[],'all'); 
    Vs_max = max(Vs_all(:,1:outputVsDim),[],'all'); 
    H_min  = min(H_all(:,1:outputHDim),[],'all');  
    H_max  = max(H_all(:,1:outputHDim),[],'all');
    
    pv_min  = min(pv_all,[],1);
    pv_max  = max(pv_all,[],1);
    pv2_min = min(pv_all2,[],1);
    pv2_max = max(pv_all2,[],1);
    
    % 构造全量归一化输入
    pv_norm_all  = normalize(pv_all, pv_min, pv_max);
    pv2_norm_all = normalize(pv_all2, pv2_min, pv2_max);
    
    XFull = cell(N_valid,1);
    for i = 1:N_valid
        XFull{i} = [pv_norm_all(i,:); pv2_norm_all(i,:) .* mask(i,:)];
    end
    
    % 全量归一化输出
    YNormFull = [normalize(Vs_all(:,1:outputVsDim), Vs_min, Vs_max), ...
                 normalize(H_all(:,1:outputHDim), H_min, H_max)];
    
    % 划分训练/验证集
    XTrainFull = XFull(trainIdxFull);
    YTrainFull = YNormFull(trainIdxFull,:);
    XValFull   = XFull(valIdxFull);
    YValFull   = YNormFull(valIdxFull,:);
  XTestFull  = XFull(testIdxFull);
    YTestFull  = YNormFull(testIdxFull,:);
    % 密度信息按训练/验证集划分
  rho_train_full = den_all(trainIdxFull,:);
rho_val_full   = den_all(valIdxFull,:);
rho_test_full  = den_all(testIdxFull,:);
    rho_all = [rho_train_full; rho_val_full];
    %% === 构建网络结构 ===
    lr = best_params(1);
    % CNN参数
    nCNN = round(best_params(2));
    kernelSize = round(best_params(3));
    baseChannel = round(best_params(4)/32)*32;  
    % LSTM参数
    nLSTM = round(best_params(5));               
    baseHidden = round(best_params(6)/32)*32;   
    maxHidden = 1024;                            
    layers = [
        sequenceInputLayer(2,'Name','input','Normalization','none','MinLength',flen)
    ];
    % === CNN层，通道按倍数递增4===
    channel = baseChannel;
    for c = 1:nCNN
        layers = [layers
            convolution1dLayer(kernelSize, channel,'Padding','same','Name',sprintf('conv%d',c))
            batchNormalizationLayer('Name',sprintf('bn%d',c))
            reluLayer('Name',sprintf('relu%d',c))
            maxPooling1dLayer(2,'Stride',2,'Name',sprintf('pool%d',c))];
        channel = min(channel*2, 1024);
    end
    
    % === LSTM层，隐藏单元按倍数递增 ===
    hidden = baseHidden;
    for l = 1:nLSTM
        layers = [layers
            lstmLayer(hidden,'OutputMode','last','Name',sprintf('lstm%d',l))];
        hidden = min(hidden*2, maxHidden);
    end
    
    % FC + 自定义回归层
   
customLayerFull = customRegressionLayer('normMSEWeighted', Vs_min,Vs_max,H_min,H_max,rho_all,vs_half,rho_half,true);

    layers = [layers
        fullyConnectedLayer(outputTotal,'Name','fc')
        customLayerFull
    ];
    %% === 训练网络（全量样本） ===
    options = trainingOptions('adam', ...
        'MaxEpochs',100, ...
        'MiniBatchSize',64, ...
        'ExecutionEnvironment','gpu', ... 
        'Shuffle','never', ...
        'ValidationData',{XValFull,YValFull}, ...
        'ValidationFrequency', 50, ...   
        'Plots','training-progress', ...
        'Verbose',true, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateDropPeriod',10, ...
        'InitialLearnRate',best_params(1));

 
    StartTime = clock;
   [net, info] = trainNetwork(XTrainFull, YTrainFull, layers, options);

    