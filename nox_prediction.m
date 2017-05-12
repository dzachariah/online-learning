%% Load data
clear all;
load('data.mat');

dataMatrix = zeros(24, 7, 260, 5);
dataMatrix(:, :, :, 1) = reshape(nox(24*4:43775, 1), 24, 7, 260);
dataMatrix(:, :, :, 2) = reshape(temperature(24*4:43775), 24, 7, 260);
dataMatrix(:, :, :, 3) = reshape(windSpeed(24*4:43775), 24, 7, 260);
dataMatrix(:, :, :, 4) = reshape(pressure(24*4:43775), 24, 7, 260);
dataMatrix(:, :, :, 5) = reshape(radiation(24*4:43775), 24, 7, 260);

% Regression model
% y(t) = phi(t)'*w, where y is NOx concentration, t is time, phi(t) is the
% vector of regressor, and w is the vector of weight

% The regressors: 
% phi = [sin(2*pi*f1*t) cos(2*pi*f1*t) ... sin(2*pi*fk*t) cos(2*pi*fk*t)]

% Frequencies (Hz) (weekly, daily, 12 hours, 8 hours)
f = [1/(7*24*60*60) 1/(24*60*60) 1/(12*60*60) 1/(8*60*60)];

% Select the day to predict NOx (test inputs)
t_tst_begin = '2014-03-12 00:00:00';
t_tst_end = '2014-03-12 23:00:00';
t_tst = [time(find(time == t_tst_begin)):1/24:time(find(time == t_tst_end))]';
t_tst_sec = posixtime(t_tst);

y_tst = nox(find(time == t_tst_begin):find(time == t_tst_end), 1);

% Training data for online learning
% POSIX time (second)
t = [time(find(time == t_tst_begin) - 1):-1/24:time(find(time == t_tst_begin) - 24*365*5)]';
t_sec = posixtime(t); 
% NOx value (mg/m^3)
y = [nox((find(time == t_tst_begin) - 1):-1:(find(time == t_tst_begin) - 24*365*5), 1)];
% Training data set D
D = [y, t_sec]; 

% Average daily NOx:
% mu_tst = func_mean(t_tst, dataMatrix);
% y_ave = mu_tst(1:24);
y_ave = mean(reshape(flip(y), 24, 365*5), 2);

%Input:
% y   - nx1 outputs from training set

% Phi - nxp matrix of regressors vectors (row-wise)
Phi = zeros(length(y), 2*length(f)); 
for row = 1:length(y)
    Phi(row, :) = [sin(2*pi*f(1)*D(row, 2)) cos(2*pi*f(1)*D(row, 2)) ...
        sin(2*pi*f(2)*D(row, 2)) cos(2*pi*f(2)*D(row, 2)) ... 
        sin(2*pi*f(3)*D(row, 2)) cos(2*pi*f(3)*D(row, 2)) ... 
        sin(2*pi*f(4)*D(row, 2)) cos(2*pi*f(4)*D(row, 2))];
end
% U   - dimension of mean parameters
U = 1; 
% L   - number of iterations per dimension
L = 30;

%Output
% w_hat_spice - px1 weights of linear regression predictor
[ w_hat_spice ] = compute_spicepredictor( y, Phi, U, L ); 

%Usage:
% Given phi(x_test) of a test point x_test, compute prediction as
% y_prediction = phi(x_test)' * w_hat_spice 

y_prediction = zeros(24, 1); 
for i = 1:24
    phi = [sin(2*pi*f(1)*t_tst_sec(i)) cos(2*pi*f(1)*t_tst_sec(i)) ...
        sin(2*pi*f(2)*t_tst_sec(i)) cos(2*pi*f(2)*t_tst_sec(i)) ... 
        sin(2*pi*f(3)*t_tst_sec(i)) cos(2*pi*f(3)*t_tst_sec(i)) ... 
        sin(2*pi*f(4)*t_tst_sec(i)) cos(2*pi*f(4)*t_tst_sec(i))];
    y_prediction(i) = phi*w_hat_spice;
end
y_prediction_plusave = y_prediction + y_ave;

%% Plot
figure; grid on;
hold on; 
plot(t_tst, y_prediction, 'or'); 
plot(t_tst, y_prediction_plusave, '-+r'); 
plot(t_tst, y_tst, 'xb'); 
xlabel('Time'); ylabel('NOx (\mug/m^3)'); 
legend('y_{prediction}', 'y_{prediction} + y_{ave}', 'y_{tst}');