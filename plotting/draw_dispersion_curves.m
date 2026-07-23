function draw_dispersion_curves(freq, pv_all, pv_all2, idx)

    num_models = length(idx);

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
    %% --- 绘制基阶频散曲线 ---
    fig1 = figure('Color','w', 'Position', [100, 100, 300, 600]); hold on;
    for i = 1:num_models
        plot(pv_all(idx(i),:), freq, 'Color', cmap(i,:), ...
             'LineWidth', 1.8, 'LineStyle', '-');
    end
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'top', ...
             'FontName', 'Times New Roman', 'FontSize', 16, 'LineWidth', 1);
    xlabel('Phase velocity (m/s)', 'FontSize', 16, 'FontWeight', 'normal');
    ylabel('Frequency (Hz)', 'FontSize', 16, 'FontWeight', 'normal');
    xlim([0, 1300]);   xticks([0 400 800 1200]);
    ylim([min(freq), max(freq)]);
    box on; grid off;
    hold off;
    %% --- 绘制一阶频散曲线 ---
    fig2 = figure('Color','w', 'Position', [100, 100, 300, 600]); hold on;
    for i = 1:num_models
        plot(pv_all2(idx(i),:), freq, 'Color', cmap(i,:), ...
             'LineWidth', 1.5, 'LineStyle', '--');
    end
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'top', ...
             'FontName', 'Times New Roman', 'FontSize', 16, 'LineWidth', 1);
    xlabel('Phase velocity (m/s)', 'FontSize', 16, 'FontWeight', 'normal');
    ylabel('Frequency (Hz)', 'FontSize', 16, 'FontWeight', 'normal');
    xlim([0, 1300]);  xticks([0 400 800 1200]);
    ylim([min(freq), max(freq)]);
    box on; grid off;
    hold off;
end
