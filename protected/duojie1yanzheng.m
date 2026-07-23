%% ================= 手动输入基阶频散曲线 =================
StartTime = clock;
pv_manual_base = [ ...];
pv_manual_2nd = [ ...];  

%% ================= 构造掩码 =================
mask_base   = zeros(1, flen);
mask_base(1:Nfreq_base) = 1;

mask_2nd    = zeros(1, flen);
mask_2nd(Nfreq_base+1:end) = 1;

%% ================= 构造归一化输入 =================
pv_manual = zeros(1, flen);  
pv_manual(1:Nfreq_base)       = pv_manual_base;
pv_manual(Nfreq_base+1:end)   = 0; 

pv2_manual = zeros(1, flen);
pv2_manual(Nfreq_base+1:end)  = pv_manual_2nd; 

% 归一化
pv_manual_norm  = normalize(pv_manual,  pv_min,  pv_max);
pv2_manual_norm = normalize(pv2_manual, pv2_min, pv2_max);

%% ================= 构造网络输入 =================
X_manual = cell(1,1);
X_manual{1} = [
    pv_manual_norm .* mask_base;     
    pv2_manual_norm .* mask_2nd      
];
Y_pred_norm = predict(net, X_manual);
Vs_pred = denormalize(Y_pred_norm(1:outputVsDim), Vs_min, Vs_max);
H_pred  = denormalize(Y_pred_norm(outputVsDim+1:end), H_min, H_max);

Vs_full = [Vs_pred, 600];        
H_full  = H_pred;
Vp_full = zeros(1,length(Vs_full));
den_full = [2000*ones(1,length(Vs_pred)), 2000];  

% 使用泊松比计算纵波速度
nu_layers = [0.38, 0.38, 0.35, 0.35, 0.3 0.3]; 
for k = 1:length(Vs_full)
    nu = nu_layers(k);
    Vp_full(k) = Vs_full(k) * sqrt(2*(1-nu)/(1-2*nu));
end

% 正演计算预测频散
pv_pred_all = calcmulti(freq, Vs_full, H_full, Vp_full, den_full);

%% ================= 绘制预测 vs 实测 =================
figure('Color','w', 'Position',[100 100 500 400]); 
hold on;

plot(freq(1:Nfreq_base), pv_manual_base, 'ko', 'LineWidth',2.2, 'MarkerFaceColor','k');

c_red = [214, 39, 24] / 255;    

plot(freq(1:Nfreq_base), pv_pred_all(1:Nfreq_base,1), ...
    '-', ...
    'Color', c_red, ...
    'LineWidth', 2.2, ...
    'MarkerFaceColor', c_red);

plot(freq(Nfreq_base+1:end), pv_manual_2nd, 'ko', 'LineWidth',2.2, 'MarkerFaceColor','k');

c_red = [214, 39, 24] / 255;   

plot(freq(Nfreq_base+1:end), pv_pred_all(Nfreq_base+1:end,2), ...
    '-', ...
    'Color', c_red, ...
    'LineWidth', 2.2, ...
    'MarkerFaceColor', c_red);

xlabel('$f$ (Hz)', 'Interpreter','latex', ...
       'FontName','Times New Roman', 'FontSize',16);
ylabel('Phase Velocity (m/s)', 'FontName','Times New Roman', 'FontSize',16);

lgd = legend('Measured data','Multimodal',  'Location','northeast');
set(lgd,'FontName','Times New Roman','FontSize',16,'Box','off');

grid on;
set(gca,'FontName','Times New Roman','FontSize',16);

box on;  
set(gca, ...
    'FontName','Times New Roman', ...
    'FontSize',16, ...
    'LineWidth',1, ...
    'Box','on');
  xticks([ 10 20 30 40 50 60]);
xlim([8 61])
  yticks([ 150 200 250 300 350 400 450 ]);
ylim([140 450]);  