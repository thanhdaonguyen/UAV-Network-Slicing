function fig = compare_agents(results_dir)
    % Generate comparison plots for multiple agents
    % Input: results_dir - directory containing CSV files
    % Output: fig - figure handle
    
    % Find all CSV files in the directory
    csv_files = dir(fullfile(results_dir, '*.csv'));
    
    if isempty(csv_files)
        error('No CSV files found in %s', results_dir);
    end

    % bring MADRL_Trained.csv to the end of the list if it exists
    madrl_index = find(strcmp({csv_files.name}, 'MADRL_Trained.csv'));
    if ~isempty(madrl_index) && madrl_index ~= length(csv_files)
        madrl_file = csv_files(madrl_index);
        csv_files(madrl_index) = [];
        csv_files(end+1) = madrl_file;
    end

    fprintf('Found %d agent result files\n', length(csv_files));
    
    % Load data from each CSV file
    results_list = struct('agent_name', {}, 'metrics', {}, 'summary', {});
    
    for i = 1:length(csv_files)
        csv_path = fullfile(csv_files(i).folder, csv_files(i).name);
        
        % Extract agent name from filename
        [~, filename, ~] = fileparts(csv_files(i).name);
        agent_name = strrep(filename, '_', ' ');
        
        fprintf('  Loading: %s\n', agent_name);
        
        try
            % Read CSV file
            data = readtable(csv_path);
            
            % Extract metrics from table
            metrics = struct();
            metrics.steps = data.step;
            metrics.rewards = data.reward;
            metrics.cumulative_rewards = data.cumulative_reward;
            metrics.qos = data.qos;
            metrics.energy = data.energy;
            metrics.fairness = data.fairness;
            metrics.active_ues = data.active_ues;
            
            % Calculate summary statistics
            summary = struct();
            if ~isempty(metrics.cumulative_rewards)
                summary.total_reward = metrics.cumulative_rewards(end);
            else
                summary.total_reward = 0.0;
            end
            summary.mean_qos = mean(metrics.qos);
            summary.mean_energy = mean(metrics.energy);
            summary.mean_fairness = mean(metrics.fairness);
            summary.mean_active_ues = mean(metrics.active_ues);
            
            % Create result structure
            result = struct();
            result.agent_name = agent_name;
            result.metrics = metrics;
            result.summary = summary;
            
            results_list(end+1) = result;
            
            fprintf('    ✓ Loaded %d steps\n', length(metrics.steps));
            fprintf('    Total reward: %.2f\n', summary.total_reward);
            
        catch ME
            fprintf('    ⚠️  Error loading %s: %s\n', csv_files(i).name, ME.message);
            continue;
        end
    end

    if isempty(results_list)
        error('No valid agent results could be loaded');
    end
    
    % Sort by agent name for consistent ordering
    [~, idx] = sort({results_list.agent_name});
    results_list = results_list(idx);
    
    fprintf('\n✓ Successfully loaded %d agents\n', length(results_list));
    fprintf('Generating comparison plots...\n');
    
    % Create figure with subplots
    fig = figure('Position', [100, 100, 1600, 880]);
    
    % Standard highlighting colors (matching matplotlib defaults)
    standard_colors = [
        0.1216, 0.4667, 0.7059;  % blue
        0.8392, 0.1529, 0.1569;  % red
        0.1725, 0.6275, 0.1725;  % green
        1.0000, 0.4980, 0.0549;  % orange
        0.5804, 0.4039, 0.7412;  % purple
        0.5490, 0.3373, 0.2941;  % brown
        0.8902, 0.4667, 0.7608;  % pink
        0.4980, 0.4980, 0.4980   % gray
    ];

    standard_markers = ['o'; 's'; 'd'; '^'; 'v'; '>'; '<'; 'p'; 'h'; '+'; '*'; 'x'];

    num_agents = length(results_list);
    colors = standard_colors(1:min(num_agents, size(standard_colors, 1)), :);
    if num_agents > size(standard_colors, 1)
        % Generate additional colors if needed
        extra_colors = lines(num_agents - size(standard_colors, 1));
        colors = [colors; extra_colors];
    end
    
    % Main title
    % sgtitle('Agent Performance Comparison', ...
    %         'FontSize', 16, 'FontWeight', 'bold');
    
    % 1. Step Rewards
    subplot(2, 3, 1);
    hold on;
    for i = 1:num_agents
        result = results_list(i);
        steps = result.metrics.steps;
        rewards = result.metrics.rewards;
        
        % Smooth with moving average
        window = min(40, floor(length(rewards) / 10));
        if window < 1, window = 1; end
        
        if window > 1
            smoothed = movmean(rewards, window);
            plot(steps, smoothed, 'Color', colors(i,:), ...
                 'LineWidth', 2.5, 'DisplayName', result.agent_name);
            % Plot raw data with transparency
            plot(steps, rewards, 'Color', [colors(i,:), 0.2], 'LineWidth', 1, ...
                 'HandleVisibility', 'off');
        else
            plot(steps, rewards, 'Color', colors(i,:), ...
                 'LineWidth', 2.5, 'DisplayName', result.agent_name);
        end
    end
    title('Step Rewards', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Step');
    ylabel('Reward');
    legend('Location', 'best');
    grid on;
    alpha(gca, 0.3);
    hold off;
    
    % 2. Cumulative Rewards
    subplot(2, 3, 2);
    hold on;
    for i = 1:num_agents
        result = results_list(i);
        steps = result.metrics.steps;
        cumulative_rewards = result.metrics.cumulative_rewards;
        plot(steps, cumulative_rewards, 'Color', colors(i,:), ...
             'LineWidth', 2, 'DisplayName', result.agent_name);
    end
    title('Cumulative Rewards', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Step');
    ylabel('Cumulative Reward');
    legend('Location', 'best');
    grid on;
    alpha(gca, 0.3);
    hold off;
    
    % 3. QoS Satisfaction
    subplot(2, 3, 3);
    hold on;
    for i = 1:num_agents
        result = results_list(i);
        steps = result.metrics.steps;
        qos = result.metrics.qos;
        
        % Smooth
        window = min(200, floor(length(qos) / 10));
        if window < 1, window = 1; end
        
        if window > 1
            smoothed = movmean(qos, window);
            plot(steps, smoothed, 'Color', colors(i,:), ...
                 'LineWidth', 1, 'DisplayName', result.agent_name, 'Marker', standard_markers(i,:), ...
                 'MarkerSize', 4, 'MarkerIndices', 1:round(length(steps)/10):length(steps));
            % plot(steps, qos, 'Color', [colors(i,:), 0.2], 'LineWidth', 1, ...
            %      'HandleVisibility', 'off');
        else
            plot(steps, qos, 'Color', colors(i,:), ...
                 'LineWidth', 1, 'DisplayName', result.agent_name);
        end
    end
    title('QoS Satisfaction', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Step');
    ylabel('QoS Satisfaction');
    ylim([0, 1.05]);
    legend('Location', 'best');
    grid on;
    alpha(gca, 0.3);
    hold off;
    
    % 4. Energy Efficiency
    subplot(2, 3, 4);
    hold on;
    for i = 1:num_agents
        result = results_list(i);
        steps = result.metrics.steps;
        energy = result.metrics.energy;
        
        % Smooth
        window = min(40, floor(length(energy) / 10));
        if window < 1, window = 1; end
        
        if window > 1
            smoothed = movmean(energy, window);
            plot(steps, smoothed, 'Color', colors(i,:), ...
                 'LineWidth', 2, 'DisplayName', result.agent_name);
            plot(steps, energy, 'Color', [colors(i,:), 0.2], 'LineWidth', 1, ...
                 'HandleVisibility', 'off');
        else
            plot(steps, energy, 'Color', colors(i,:), ...
                 'LineWidth', 2, 'DisplayName', result.agent_name);
        end
    end
    title('Energy Consumption Rate', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Step');
    ylabel('Energy Consumption (Lower is Better)');
    ylim([0, 1])
    legend('Location', 'best');
    grid on;
    alpha(gca, 0.3);
    hold off;
    
    % 5. Fairness Index
    subplot(2, 3, 5);
    hold on;
    for i = 1:num_agents
        result = results_list(i);
        steps = result.metrics.steps;
        fairness = result.metrics.fairness;
        
        % Smooth
        window = min(80, floor(length(fairness) / 10));
        if window < 1, window = 1; end
        
        if window > 1
            smoothed = movmean(fairness, window);
            plot(steps, smoothed, 'Color', colors(i,:), ...
                 'LineWidth', 2, 'DisplayName', result.agent_name);
            % plot(steps, fairness, 'Color', [colors(i,:), 0.15], 'LineWidth', 1, ...
            %      'HandleVisibility', 'off');
        else
            plot(steps, fairness, 'Color', colors(i,:), ...
                 'LineWidth', 2, 'DisplayName', result.agent_name);
        end
    end
    title('Fairness Index', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Step');
    ylabel('Fairness');
    ylim([0, 0.8]);
    legend('Location', 'best');
    grid on;
    alpha(gca, 0.3);
    hold off;
    
    % 6. Summary Bar Chart
    subplot(2, 3, 6);
    metrics_names = {'Reward'; 'QoS'; 'Energy'; 'Fairness'};
    x = 1:4;
    width = 0.8 / num_agents;
    
    % Calculate normalized metrics
    total_rewards = arrayfun(@(r) r.summary.total_reward, results_list);
    max_reward = max(total_rewards);
    if max_reward == 0, max_reward = 1; end

    
    hold on;
    for i = 1:num_agents
        result = results_list(i);
        
        values = [
            result.summary.total_reward / max_reward;
            result.summary.mean_qos;
            result.summary.mean_energy;  
            result.summary.mean_fairness
        ];
        
        offset = (i - num_agents/2 - 0.5) * width;
        bar(x + offset, values, width, 'FaceColor', colors(i,:), ...
            'EdgeColor', 'none', 'FaceAlpha', 1, 'DisplayName', result.agent_name);
    end
    
    ylabel('Normalized Performance');
    title('Performance Summary (Normalized)', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', metrics_names);
    xlim([0.3, 4.7])
    ylim([0, 1.1]);
    legend('Location', 'best');
    grid on;
    alpha(gca, 0.8);
    hold off;
    
    % Save figure
    plot_path = fullfile(results_dir, 'comparison.png');
    saveas(fig, plot_path);
    fprintf('Saved comparison plot to %s\n', plot_path);
end