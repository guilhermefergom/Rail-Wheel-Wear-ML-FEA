clear all 
close all
clc

%% Plot Settings
set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 14) % Reduced for better table viewing
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 14)

%% Load Database
% Make sure x.mat and OutPut.mat are in the same folder
load x.mat
load OutPut.mat
rng(2);

% X = [RailA RailB WhlA WhlB WhlC]
% Y = [ConArea MaxPress MaxStress]
X = x';
Y = OutPut';
varname = {'Rail_A (mm)',' Rail_B (mm)', 'Wheel_A (mm)', 'Wheel_B (mm)', 'Wheel_C (mm)'};
respname = {'Contact Area (mm^2)',' Max. Pressure (MPa)', 'Max. Stress (MPa)'};
interpolator_name = {'GPR','SVM','DT','LR','NLR','ANN'};

%% Train/Test Split (70% - 30%)
[train_idx, ~, test_idx] = dividerand(length(X), 0.7, 0, 0.3);
x_train = X(train_idx, :);
y_train = Y(train_idx, :);
x_test = X(test_idx, :);
y_test = Y(test_idx, :);

%% Preallocate metrics matrices and store predictions
MSE_results = zeros(6, 3);
RMSE_results = zeros(6, 3);
R2_results = zeros(6, 3);
Y_PRED_STORE = cell(6, 3); % Cell array to store predictions for plotting later

%% Master Loop for all Models and Responses
fprintf('Training models. Please wait, Bayesian optimization may take some time...\n\n');

for N = 1:6
    fprintf('Training Model %d/6: %s...\n', N, interpolator_name{N});
    
    if N == 6
        % ==========================================
        % Artificial Neural Networks (Trains all 3 outputs at once)
        % ==========================================
        [x_train_norm, x_ps] = mapstd(x_train');
        [y_train_norm, y_ps] = mapstd(y_train');
        
        net = feedforwardnet(40);
        net.trainFcn = 'trainlm'; 
        net.trainParam.goal = 1e-2; 
        net.trainParam.min_grad = 1e-20;
        net.trainParam.mu_max = 1e20;    
        net.trainParam.epochs = 10000;
        net.trainParam.lr = 0.15;
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'purelin';
        net.trainParam.max_fail = net.trainParam.epochs;
        net.trainParam.showWindow = false; % Hide UI to keep script fast
        net.DivideFcn = '';    
        
        net = train(net, x_train_norm, y_train_norm);
        
        % Test Data evaluation
        x_test_norm = mapstd('apply', x_test', x_ps);
        y_norm_pred = sim(net, x_test_norm);
        y_pred_all = mapstd('reverse', y_norm_pred, y_ps)'; % Transpose back to N x 3
        
        for K = 1:3
            y_pred = y_pred_all(:, K);
            y_true = y_test(:, K);
            
            % Calculate Metrics
            MSE = mean((y_true - y_pred).^2);
            RMSE = sqrt(MSE);
            SS_res = sum((y_true - y_pred).^2);
            SS_tot = sum((y_true - mean(y_true)).^2);
            R2 = 1 - (SS_res/SS_tot);
            
            % Store Results
            MSE_results(N, K) = MSE;
            RMSE_results(N, K) = RMSE;
            R2_results(N, K) = R2;
            Y_PRED_STORE{N, K} = y_pred; % Store prediction
        end
        
    else
        % ==========================================
        % Regression Models (Trains each output individually)
        % ==========================================
        for K = 1:3
            y_true = y_test(:, K);
            
            % Optimization options (ShowPlots = false to avoid 100s of popups)
            opt = struct('AcquisitionFunctionName', 'expected-improvement-plus', 'ShowPlots', false);
            
            switch N
                case 1 % GPR
                    mdl = fitrgp(x_train, y_train(:,K), 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
                    y_pred = predict(mdl, x_test);
                    
                case 2 % SVM
                    mdl = fitrsvm(x_train, y_train(:,K), 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
                    y_pred = predict(mdl, x_test);
                    
                case 3 % Decision Tree
                    mdl = fitrtree(x_train, y_train(:,K), 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
                    y_pred = predict(mdl, x_test);
                    
                case 4 % Linear Regression
                    mdl = fitrlinear(x_train, y_train(:,K), 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opt);
                    y_pred = predict(mdl, x_test);
                    
                case 5 % Non-Linear Regression (Quadratic RSM)
                    % fitlm with 'quadratic' automatically handles the 5 inputs
                    mdl = fitlm(x_train, y_train(:,K), 'quadratic');
                    y_pred = predict(mdl, x_test);
            end
            
            % Calculate Metrics
            MSE = mean((y_true - y_pred).^2);
            RMSE = sqrt(MSE);
            SS_res = sum((y_true - y_pred).^2);
            SS_tot = sum((y_true - mean(y_true)).^2);
            R2 = 1 - (SS_res/SS_tot);
            
            % Store Results
            MSE_results(N, K) = MSE;
            RMSE_results(N, K) = RMSE;
            R2_results(N, K) = R2;
            Y_PRED_STORE{N, K} = y_pred; % Store prediction
        end
    end
end
fprintf('Training Complete!\n\n');

%% Generate Output Tables for the Article
Model_Names = categorical(interpolator_name');
for K = 1:3
    fprintf('=====================================================\n');
    fprintf(' PERFORMANCE METRICS FOR: %s\n', respname{K});
    fprintf('=====================================================\n');
    
    MSE  = MSE_results(:, K);
    RMSE = RMSE_results(:, K);
    R_Squared = R2_results(:, K);
    
    % Create Table
    ResultsTable = table(Model_Names, MSE, RMSE, R_Squared, ...
        'VariableNames', {'Algorithm', 'MSE', 'RMSE', 'R2'});
    
    disp(ResultsTable);
    fprintf('\n');
end

%% ========================================================================
%% PLOT DE TODOS OS 6 REGRESSORES E EXPORTAÇÃO PARA PDF
%% ========================================================================
fprintf('Generating and saving regression plots for all models...\n');

for N = 1:6
    % Cria uma nova figura para cada modelo
    fig = figure('Name', sprintf('Regression Plots - %s', interpolator_name{N}), ...
           'Position', [100+(N*20), 100+(N*20), 1200, 400]);

    for K = 1:3
        subplot(1, 3, K);
        
        % Resgata os dados reais e os dados preditos pelo modelo N
        y_true_plot = y_test(:, K);
        y_pred_plot = Y_PRED_STORE{N, K};
        
        % Scatter plot
        scatter(y_true_plot, y_pred_plot, 40, 'MarkerEdgeColor', [0 0.4470 0.7410], ...
            'MarkerFaceColor', [0.3010 0.7450 0.9330], 'MarkerFaceAlpha', 0.6);
        hold on;
        
        % Linha de identidade (y = x)
        min_val = min(y_true_plot);
        max_val = max(y_true_plot);
        plot([min_val, max_val], [min_val, max_val], 'r-', 'LineWidth', 2);
        
        % Text box com R2
        str_r2 = sprintf('R^2 = %.4f', R2_results(N, K));
        text(min_val + 0.05*(max_val-min_val), max_val - 0.1*(max_val-min_val), ...
            str_r2, 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k');
        
        % Labels e Títulos
        xlabel('Actual / FEA');
        ylabel(sprintf('Predicted / %s', interpolator_name{N}));
        title(respname{K});
        axis square;
        grid on;
    end
    
    % Título geral da figura
    sgtitle(sprintf('Regression Performance of %s', interpolator_name{N}), 'FontWeight', 'bold');
    
    % Exportar para PDF
    filename = sprintf('Regression_Plot_%s.pdf', interpolator_name{N});
    exportgraphics(fig, filename, 'ContentType', 'vector');
    fprintf('Saved %s\n', filename);
end

fprintf('All plots generated and saved successfully!\n');