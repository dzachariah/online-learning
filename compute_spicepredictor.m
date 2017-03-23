function [ w_hat_spice ] = compute_spicepredictor( y, Phi, U, L )
%Compute the weights of SPICE predictor
%Dave Zachariah 2017-03-22

%Input:
% y   - nx1 outputs from training set
% Phi - nxp matrix of regressors vectors (row-wise)
% U   - dimension of mean parameters
% L   - number of iterations per dimension

%Output
% w_hat_spice - px1 weights of linear regression predictor

%Usage:
% Given phi(x_test) of a test point x_test, compute prediction as
% y_prediction = phi(x_test)' * w_hat_spice 


%% Initialize
%Dimensions
[N,P] = size(Phi);

%Variables
w_hat_spice = zeros(P,1);
Gamma_spice = zeros(P,P);
rho_spice   = zeros(P,1);
kappa_spice = zeros(1,1);


%% Online learning and computation of weights
for i = 1:N
    [w_hat_spice, Gamma_spice, rho_spice, kappa_spice] = func_newsample_covlearn( y(i), Phi(i,:), Gamma_spice, rho_spice, kappa_spice, w_hat_spice, i, L, P, U );    
end



end

