function idx = draw_vs_stairs_comparison(Vs, H, num_models, random_sample)
    if nargin < 3
        num_models = 15;
    end
    if nargin < 4
        random_sample = false;
    end

    total_models = size(Vs, 1);
    if num_models > total_models
        error('请求绘制的模型数超过总样本数');
    end

    if random_sample
        idx = randperm(total_models, num_models);
    else
        idx = 1:num_models;
    end
base_colors = [
    220, 132,  38;   
     46, 137, 196;   
    198, 154,  36;   
     35, 168, 175;   
     45, 158,  88;   
    215,  82,  67;   
    125,  95, 190;   
    198, 134,  70;   
    200,  82, 122;   
     96, 125, 145;   
     80, 154, 145;   
    170, 155,  63;   
    110, 176, 130;   
    180, 126,  92;  
     83, 133, 160    
] / 255;
    cmap = repmat(base_colors, ceil(num_models / size(base_colors,1)), 1);
    cmap = cmap(1:num_models, :);

    % === 绘制图形 ===
    fig = figure('Color','w', 'Position', [100, 100, 300, 600]); hold on;
    for i = 1:num_models
        vs = Vs(idx(i), :);
        h  = H(idx(i), :);

        if size(vs,2) > size(h,2)
            vs = vs(1:end-1);
        end

        z = [0, cumsum(h)];
        vs_plot = zeros(1, 2*length(vs));
        z_plot  = zeros(1, 2*length(h));
        for k = 1:length(vs)
            vs_plot(2*k-1:2*k) = vs(k);
        end
        for k = 1:length(h)
            z_plot(2*k-1) = z(k);
            z_plot(2*k)   = z(k+1);
        end
        stairs(vs_plot, z_plot, 'Color', cmap(i,:), 'LineWidth', 1.8);
    end

    set(gca, 'YDir', 'reverse');
    set(gca, 'XAxisLocation', 'top');  
    set(gca, 'FontName', 'Times', 'FontSize', 16, 'LineWidth', 1);
    xticks([0 400 800 1200]);
    xlim([0 1300]);
    box on
  xlabel('$V_{\mathrm{s}}$ (m/s)', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('Depth (m)', 'FontSize', 16);
    yticks(0:5:30);
ylim([0 30.5]);
    grid off;
    hold off;
