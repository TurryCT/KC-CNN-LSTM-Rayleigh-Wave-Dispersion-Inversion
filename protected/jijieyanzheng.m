
%% ================= 手动输入基阶频散曲线 =================
pv_manual = [ ];

if length(pv_manual) ~= flen
    error('❌ 手动输入 PV 长度 (%d) 与 flen (%d) 不一致', ...
        length(pv_manual), flen);
end

%% ================= 构造一阶频散（关闭） =================
pv2_manual  = zeros(1, flen);
mask_manual = zeros(1, flen);

%% ================= 归一化函数 =================
normalize   = @(x,xmin,xmax) (x-xmin)./(xmax-xmin+eps);
denormalize = @(x,xmin,xmax) x.*(xmax-xmin)+xmin;

%% ================= 归一化输入 =================
pv_manual_norm  = normalize(pv_manual,  pv_min,  pv_max);
pv2_manual_norm = normalize(pv2_manual, pv2_min, pv2_max);

%% ================= 构造网络输入 =================
X_manual = cell(1,1);
X_manual{1} = [
    pv_manual_norm;
    pv2_manual_norm .* mask_manual
];

%% ================= 网络预测 =================
Y_pred_norm = predict(net, X_manual);

if numel(Y_pred_norm) ~= (outputVsDim + outputHDim)
    error('❌ 网络输出维度错误');
end

Vs_pred = denormalize( ...
    Y_pred_norm(1:outputVsDim), Vs_min, Vs_max);

H_pred  = denormalize( ...
    Y_pred_norm(outputVsDim+1:end), H_min, H_max);

nu_layers = [0.38, 0.38, 0.35, 0.35, 0.3]; 
Vs_full = [Vs_pred, 600];                  
Vp_full = zeros(1,length(Vs_full));

for k = 1:length(Vs_full)
    nu = nu_layers(k);
    Vp_full(k) = Vs_full(k) * sqrt(2*(1-nu)/(1-2*nu));
end

den_pred = 2000*ones(1,length(Vs_pred));      
den_full = [den_pred, 2000];                   


pv_pred_all = calcmulti(freq, Vs_full, H_pred, Vp_full, den_full);

fig2 = figure('Color','w', 'Position', [100 100 500 400]); 
hold on;

plot(freq, pv_manual, 'ko', ...
    'LineWidth',2.2, 'MarkerFaceColor','k'); 
c_pred = [0, 158, 115] / 255;  
plot(freq, pv_pred_all(:,1), '-', ...
    'Color', c_pred, ...
    'LineWidth', 2.2, ...
    'MarkerFaceColor', c_pred);  

xticks([ 10 15 20 25 30]);
xlim([9 30.5])
yticks([ 150 200 250 300 350 400 450 ]);
ylim([140 450]);   
xlabel('$f$ (Hz)', 'Interpreter','latex', ...
       'FontName','Times New Roman', 'FontSize',16);

ylabel('Phase Velocity (m/s)', 'FontName','Times New Roman', 'FontSize',16);

lgd = legend('Measured data','Fundamental-mode', 'Location','northeast');
set(lgd, 'FontName','Times New Roman', 'FontSize',16, 'Box','off');

grid on;
set(gca, 'FontName','Times New Roman', 'FontSize',16);
box on;

ax = gca;
ax.LineWidth = 1;


