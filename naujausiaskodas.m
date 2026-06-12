% Heston model portfolio simulation - bachelor thesis
% Compares Merton, myopic-only and full Heston strategies

clear all; close all; clc;

%% 1. Load data

data = readtable('stock_data.csv');
dates = data.Date(2:end);
returns_aapl = data.AAPL_Return(2:end);
returns_jnj = data.JNJ_Return(2:end);
returns_xom = data.XOM_Return(2:end);
prices_aapl = data.AAPL_Price(2:end);
prices_jnj = data.JNJ_Price(2:end);
prices_xom = data.XOM_Price(2:end);

returns = [returns_aapl, returns_jnj, returns_xom];
returns = returns(~any(isnan(returns), 2), :);

fprintf('Loaded %d daily returns\n', size(returns, 1));

%% 2. Parameters

r = 0.025;                          % risk-free rate
mu_aapl = mean(returns_aapl) * 252;
mu_jnj = mean(returns_jnj) * 252;
mu_xom = mean(returns_xom) * 252;
mu = (mu_aapl + mu_jnj + mu_xom) / 3;
gamma = 5;                          % risk aversion

% equal-weighted portfolio
returns_portfolio = mean(returns, 2);
v0 = var(returns_portfolio) * 252;  % initial variance
theta = v0;                         % long-run variance
kappa = 3;                          % mean reversion speed
xi = 0.3;                           % vol of vol
rho = -0.5;                         % correlation (leverage effect)

% Feller condition check
if 2*kappa*theta >= xi^2
    fprintf('Feller condition OK: %.4f >= %.4f\n', 2*kappa*theta, xi^2);
else
    fprintf('Warning: Feller condition violated\n');
end

fprintf('mu = %.4f, v0 = %.4f, theta = %.4f\n', mu, v0, theta);

% hedging coefficient, formula (3.28): pi_hedge = rho*xi/(gamma*theta) * (v-theta)/theta
C_hedge = (rho * xi) / (gamma * theta);
fprintf('Hedging coefficient = %.4f\n', C_hedge);

%% 3. Simulation setup

T = 1;          % horizon, years
dt = 1/252;     % daily step
N = T/dt;
t = 0:dt:T;
M = 1000;       % number of paths
W0 = 100;       % initial wealth

%% 4. Monte Carlo simulation, three strategies

V_paths = zeros(M, N+1);
S_paths = zeros(M, N+1);
W_full = zeros(M, N+1);     % full Heston (myopic + hedging)
W_myopic = zeros(M, N+1);   % myopic only
W_merton = zeros(M, N+1);   % Merton constant vol

pi_full = zeros(M, N+1);
pi_myopic = zeros(M, N+1);
pi_hedge = zeros(M, N+1);

V_paths(:, 1) = v0;
S_paths(:, 1) = 100;
W_full(:, 1) = W0;
W_myopic(:, 1) = W0;
W_merton(:, 1) = W0;

% Merton constant allocation
pi_merton = (mu - r) / (gamma * v0);
fprintf('Merton allocation = %.4f\n', pi_merton);

% correlated shocks (Cholesky)
rng(42);
Z1 = randn(M, N);
Z2 = rho * Z1 + sqrt(1-rho^2) * randn(M, N);

for i = 1:N
    v_curr = max(V_paths(:, i), 1e-6);

    % variance, CIR process
    V_paths(:, i+1) = v_curr + kappa*(theta - v_curr)*dt + ...
                      xi*sqrt(v_curr*dt).*Z2(:, i);
    V_paths(:, i+1) = max(V_paths(:, i+1), 1e-6);

    % stock price
    S_paths(:, i+1) = S_paths(:, i) .* exp((mu - 0.5*v_curr)*dt + ...
                      sqrt(v_curr*dt).*Z1(:, i));

    % myopic demand
    pi_myopic(:, i) = (mu - r) ./ (gamma * v_curr);
    pi_myopic(:, i) = min(max(pi_myopic(:, i), 0), 2);

    % hedging demand, eq (3.28)
    pi_hedge(:, i) = C_hedge * (v_curr - theta) / theta;

    % full allocation
    pi_full(:, i) = pi_myopic(:, i) + pi_hedge(:, i);
    pi_full(:, i) = min(max(pi_full(:, i), 0), 2);

    % wealth, full Heston
    dW = W_full(:, i).*(r + pi_full(:, i).*(mu - r))*dt + ...
         W_full(:, i).*pi_full(:, i).*sqrt(v_curr*dt).*Z1(:, i);
    W_full(:, i+1) = W_full(:, i) + dW;

    % wealth, myopic only
    dW = W_myopic(:, i).*(r + pi_myopic(:, i).*(mu - r))*dt + ...
         W_myopic(:, i).*pi_myopic(:, i).*sqrt(v_curr*dt).*Z1(:, i);
    W_myopic(:, i+1) = W_myopic(:, i) + dW;

    % wealth, Merton
    dW = W_merton(:, i).*(r + pi_merton*(mu - r))*dt + ...
         W_merton(:, i).*pi_merton.*sqrt(v_curr*dt).*Z1(:, i);
    W_merton(:, i+1) = W_merton(:, i) + dW;
end

%% 5. Results

W_full_final = W_full(:, end);
W_myopic_final = W_myopic(:, end);
W_merton_final = W_merton(:, end);

sharpe_full = mean(W_full_final - W0) / std(W_full_final);
sharpe_myopic = mean(W_myopic_final - W0) / std(W_myopic_final);
sharpe_merton = mean(W_merton_final - W0) / std(W_merton_final);

fprintf('\nTerminal wealth (T = 1 year):\n');
fprintf('%-12s %10s %10s %10s\n', '', 'Full', 'Myopic', 'Merton');
fprintf('%-12s %10.2f %10.2f %10.2f\n', 'Mean', mean(W_full_final), mean(W_myopic_final), mean(W_merton_final));
fprintf('%-12s %10.2f %10.2f %10.2f\n', 'Median', median(W_full_final), median(W_myopic_final), median(W_merton_final));
fprintf('%-12s %10.2f %10.2f %10.2f\n', 'Std', std(W_full_final), std(W_myopic_final), std(W_merton_final));
fprintf('%-12s %10.2f %10.2f %10.2f\n', 'Min', min(W_full_final), min(W_myopic_final), min(W_merton_final));
fprintf('%-12s %10.2f %10.2f %10.2f\n', 'Max', max(W_full_final), max(W_myopic_final), max(W_merton_final));
fprintf('%-12s %10.4f %10.4f %10.4f\n', 'Sharpe', sharpe_full, sharpe_myopic, sharpe_merton);

gain_vs_merton = (mean(W_full_final) - mean(W_merton_final)) / mean(W_merton_final) * 100;
gain_vs_myopic = (mean(W_full_final) - mean(W_myopic_final)) / mean(W_myopic_final) * 100;
fprintf('Gain vs Merton: %.2f%%, gain vs Myopic: %.2f%%\n\n', gain_vs_merton, gain_vs_myopic);

%% 6. Empirical data graphs (section 4.2)

% normalized prices
figure('Position', [100, 100, 800, 500]);
plot(dates, prices_aapl/prices_aapl(1)*100, 'b-', 'LineWidth', 1.5); hold on;
plot(dates, prices_jnj/prices_jnj(1)*100, 'r-', 'LineWidth', 1.5);
plot(dates, prices_xom/prices_xom(1)*100, 'g-', 'LineWidth', 1.5);
xlabel('Date'); ylabel('Normalized Price (Jan 2015 = 100)');
title('Stock Price Evolution (2015-2024)');
legend('AAPL', 'JNJ', 'XOM', 'Location', 'best');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'price_paths.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% 30-day rolling volatility
figure('Position', [100, 100, 900, 500]);
window = 30;
vol_aapl = movstd(returns_aapl, window) * sqrt(252) * 100;
vol_jnj = movstd(returns_jnj, window) * sqrt(252) * 100;
vol_xom = movstd(returns_xom, window) * sqrt(252) * 100;
plot(dates(window:end), vol_aapl(window:end), 'b-', 'LineWidth', 1.5); hold on;
plot(dates(window:end), vol_jnj(window:end), 'r-', 'LineWidth', 1.5);
plot(dates(window:end), vol_xom(window:end), 'g-', 'LineWidth', 1.5);
xlabel('Date'); ylabel('Annualized Volatility (%)');
title('30-Day Rolling Realized Volatility');
legend('AAPL', 'JNJ', 'XOM', 'Location', 'best');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'realized_volatility.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% return distributions
figure('Position', [100, 100, 900, 600]);
ret_all = {returns_aapl, returns_jnj, returns_xom, returns_portfolio};
names = {'AAPL', 'JNJ', 'XOM', 'Portfolio'};
for k = 1:4
    subplot(2,2,k);
    histogram(ret_all{k}, 50, 'Normalization', 'pdf', 'FaceAlpha', 0.7);
    hold on;
    x = linspace(min(ret_all{k}), max(ret_all{k}), 100);
    plot(x, normpdf(x, mean(ret_all{k}), std(ret_all{k})), 'r--', 'LineWidth', 2);
    title([names{k} ' Returns Distribution']);
    xlabel('Daily Return'); ylabel('Probability Density');
    legend('Empirical', 'Normal Fit');
    grid on;
end
fixwhite(gcf);
exportgraphics(gcf, 'returns_distribution.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% ACF of squared returns
figure('Position', [100, 100, 900, 500]);
subplot(1,3,1); autocorr(returns_aapl.^2, 'NumLags', 20); title('ACF of r^2 (AAPL)');
subplot(1,3,2); autocorr(returns_jnj.^2, 'NumLags', 20); title('ACF of r^2 (JNJ)');
subplot(1,3,3); autocorr(returns_xom.^2, 'NumLags', 20); title('ACF of r^2 (XOM)');
fixwhite(gcf);
exportgraphics(gcf, 'autocorrelation_squared.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% QQ plots
figure('Position', [100, 100, 900, 600]);
for k = 1:4
    subplot(2,2,k);
    qqplot(ret_all{k});
    title(['QQ Plot: ' names{k}]);
end
fixwhite(gcf);
exportgraphics(gcf, 'qq_plot.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

%% 7. Simulation graphs (sections 4.4 and 4.5)

% Figure 4.2: variance paths
% legend fixed: handles saved so the legend shows the right lines
figure('Position', [100, 100, 800, 500]);
h_paths = plot(t, V_paths(1:min(50,M), :)', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
hold on;
h_mean = plot(t, mean(V_paths), 'k-', 'LineWidth', 2.5);
h_theta = plot(t, theta*ones(size(t)), 'r--', 'LineWidth', 2);
xlabel('Time (years)'); ylabel('Variance v_t');
title('Variance Paths (Heston CIR Process)');
legend([h_paths(1), h_mean, h_theta], ...
       {'Sample paths', 'Mean path', 'Long-run mean \theta'}, ...
       'Location', 'northeast');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'variance_paths.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% terminal wealth distributions
figure('Position', [100, 100, 800, 500]);
histogram(W_full_final, 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
histogram(W_myopic_final, 30, 'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
histogram(W_merton_final, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
xlabel('Terminal Wealth'); ylabel('Frequency');
title('Terminal Wealth Distributions');
legend('Full Heston', 'Myopic Only', 'Merton', 'Location', 'best');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'terminal_distributions.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% allocation decomposition (one sample path)
figure('Position', [100, 100, 900, 500]);
sp = 1;
plot(t(1:end-1), pi_full(sp, 1:end-1)*100, 'b-', 'LineWidth', 2); hold on;
plot(t(1:end-1), pi_myopic(sp, 1:end-1)*100, 'g--', 'LineWidth', 1.5);
plot(t(1:end-1), pi_hedge(sp, 1:end-1)*100, 'm:', 'LineWidth', 1.5);
yline(pi_merton*100, 'r--', 'LineWidth', 2);
xlabel('Time (years)'); ylabel('Portfolio Allocation (%)');
title('Portfolio Allocation Decomposition');
legend('Full Heston (\pi^*)', 'Myopic Component', 'Hedging Component', ...
       'Merton (constant)', 'Location', 'best');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'allocation_decomposition.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% mean wealth comparison
figure('Position', [100, 100, 800, 500]);
plot(t, mean(W_full), 'b-', 'LineWidth', 2.5); hold on;
plot(t, mean(W_myopic), 'g--', 'LineWidth', 2);
plot(t, mean(W_merton), 'r:', 'LineWidth', 2);
xlabel('Time (years)'); ylabel('Mean Wealth');
title('Mean Wealth Comparison');
legend('Full Heston', 'Myopic Only', 'Merton', 'Location', 'best');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'mean_wealth_comparison.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% Figure 4.6: hedging demand vs variance
figure('Position', [100, 100, 800, 500]);
scatter(V_paths(:), pi_hedge(:)*100, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
v_range = linspace(min(V_paths(:)), max(V_paths(:)), 100);
plot(v_range, C_hedge*(v_range - theta)/theta*100, 'r-', 'LineWidth', 2.5);
h_sim = scatter(NaN, NaN, 40, 'b', 'filled');      % dummy for legend
h_theo = plot(NaN, NaN, 'r-', 'LineWidth', 2.5);   % dummy for legend
xlabel('Variance v_t'); ylabel('Hedging Component (%)');
title('Hedging Demand vs Variance State');
legend([h_sim, h_theo], {'Simulated', 'Theoretical'}, 'Location', 'southwest');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'hedging_vs_variance.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

% Figure 4.7: total allocation vs variance
figure('Position', [100, 100, 800, 500]);
scatter(V_paths(:), pi_full(:)*100, 3, 'b', 'filled', 'MarkerFaceAlpha', 0.2);
hold on;
yline(pi_merton*100, 'r-', 'LineWidth', 2.5);
h_sim = scatter(NaN, NaN, 40, 'b', 'filled');      % dummy for legend
h_mert = plot(NaN, NaN, 'r-', 'LineWidth', 2.5);   % dummy for legend
xlabel('Variance v_t'); ylabel('Total Allocation (%)');
title('Portfolio Allocation vs Variance State');
legend([h_sim, h_mert], {'Full Heston \pi^*(v)', 'Merton (constant)'}, ...
       'Location', 'northeast');
grid on;
fixwhite(gcf);
exportgraphics(gcf, 'allocation_vs_variance.png', 'Resolution', 200, 'BackgroundColor', 'white');
close;

%% 8. Save results table

results_table = table(...
    {'Mean'; 'Median'; 'Std Dev'; 'Sharpe'; 'Min'; 'Max'}, ...
    [mean(W_full_final); median(W_full_final); std(W_full_final); sharpe_full; min(W_full_final); max(W_full_final)], ...
    [mean(W_myopic_final); median(W_myopic_final); std(W_myopic_final); sharpe_myopic; min(W_myopic_final); max(W_myopic_final)], ...
    [mean(W_merton_final); median(W_merton_final); std(W_merton_final); sharpe_merton; min(W_merton_final); max(W_merton_final)], ...
    'VariableNames', {'Metric', 'Full_Heston', 'Myopic_Only', 'Merton'});

writetable(results_table, 'simulation_results.csv');
disp('Done. Results and figures saved.');

%% helper: force white background
function fixwhite(fig)
    set(fig, 'Color', 'white');
    ax = findall(fig, 'Type', 'axes');
    set(ax, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
        'GridColor', [0.15 0.15 0.15]);
    set(findall(fig, 'Type', 'text'), 'Color', 'black');
    lg = findall(fig, 'Type', 'legend');
    set(lg, 'Color', 'white', 'TextColor', 'black', 'EdgeColor', 'black');
end
