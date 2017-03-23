function [ w_hat, Gamma, rho, kappa  ] = func_newsample_covlearn( yn, phi_row_n, Gamma, rho, kappa, w_hat, n, L, P, U )
%Online learning and resulting weights
%Dave Zachariah 

%% Recursive update
Gamma = Gamma + phi_row_n'*phi_row_n;
rho   = rho + phi_row_n'*yn;
kappa = kappa + abs(yn)^2;


%% Common variable
eta  = kappa + (w_hat'*Gamma*w_hat) - 2*( w_hat'*rho );
zeta = rho - Gamma*w_hat;


%% Cycle
for rep = 1:L
    for i = 1:P

        %Compute argument
        psi   = zeta(i) + Gamma(i,i)*w_hat(i);
        if psi >= 0
            s_hat = 1;
        else
            s_hat = -1;
        end
        
        %Compute alpha, beta, gamma
        alpha = eta + Gamma(i,i)*(w_hat(i))^2 + 2*( w_hat(i)'*zeta(i) );
        beta  = Gamma(i,i);
        gamma = abs( psi );

        %Update estimate
        w_hat_i_new = 0;
        
        if (i>=1) && (i<=U)
            if beta > 0
                w_hat_i_new = psi/beta; 
            end
            
        else 
            if (n-1)*gamma^2 > ( alpha*beta - gamma^2 )
                r_star      = real( (gamma/beta) - (1/beta) * sqrt( (alpha*beta - gamma^2)/(n-1) ) ); %ensure numerically real-valued
                w_hat_i_new = r_star * s_hat;
            end
            
        end
        
        %Update common variables
        eta  = eta + Gamma(i,i) * (w_hat(i) - w_hat_i_new )^2 + 2*( (w_hat(i) - w_hat_i_new )'*zeta(i) );
        zeta = zeta + Gamma(:,i)*(w_hat(i) - w_hat_i_new ); 

        %Store update
        w_hat(i) = w_hat_i_new;

    end
end

%% Exit

end