% This function estimates a proportional hazard rate model using maximum 
% likelihood.
%
% This code can be used for both time varying and time non-varying 
% covariates.  The nature of the covariates is determined by the type of 
% data that is used.  If the data contains a single observation for each 
% individual, i.e. values of the covariates at termination, then the model 
% will be estimated using time non-varying covariates.  However if the 
% data used is constructed using eposide splitting, so that there is an 
% observation for each individial in each time segment of the observation 
% period, then by construction the model will be estimated using time 
% varying covariates.
%
% Constrained optimization is used to facilitate convergence.  We are mainly 
% concerned with constraining the parameters that determine the shape of the
% baseline hazard, but due to the syntax of fmincon we must also provide 
% bounds for the coefficients of the covariates.  
%
% Inputs:
% 	It is assumed that each observation represents an episode so the following
%   inputs should be vectors (or a matrix in the case of covar) where each
%   row represents an observation/episode.
%
%       - id is the index for each individual; individuals may have multiple 
%         observations.
%		- ts is the starting time of the episode.
%		- tf is the ending time of the episode.
%		- event is a dummy variable that is equal to one if termination occurs
%         during the episode.
%		- covars are the covariates that influence the behavior during the 
%         episode.
%       - se_flag is a flag that calculates standard errors if it is equal to 1
%         this calculation is skipped otherwise.
%
function [haz_param, param_se] = prop_haz(bhaz, ts, tf, event, covars)
tic

% Get the number of covariates 
[nobs, ncovars] = size(covars); 

% Set the default options for maximum likelihood. The tolerance function
% is set to a more stringent value. This ensures that parameter values coverge
% to, at least, the third significant digit. 
options = optimset('Display', 'off', 'GradObj','on', 'LargeScale', 'on', 'TolFun', 1.0E-10);

% Calculate the Kaplan-Meier hazard rate to use for the starting values
KM = sum(event)/sum(tf);

% Estimate parameters using maximum likelihood
switch bhaz
    case 'log'
        disp('Performing Maximum Likelihood with log-logistic Distribution...');

        % Set the start parameters.  The first parameter is the magnitude of the 
        % baseline hazard, and the second is the shape parameter for the baseline 
        % hazard.  All other parameters are coefficients on the covariates.
        param = [KM; 1; zeros(ncovars,1)];
        
        % Set upper and lower bounds for the coefficients
        lb = [0; 0; -100*ones(ncovars, 1)];
        ub = [1; 8;  100*ones(ncovars, 1)];

        % Maximization is performed by minimizing the negative log likelihood
        [haz_param, logL, exit_flag, output, lambda, grad, hessian] = ...
             fmincon(@log_log_like, param, [], [], [], [], lb, ub, [], options, ts, tf, event, covars);

    case 'exp'
        disp('Performing Maximum Likelihood with exponentail Distribution...');

        % Set the start parameters.  The first parameter is the magnitude of the 
        % baseline hazard, and the second is the shape parameter for the baseline 
        % hazard.  All other parameters are coefficients on the covariates.
        param = [KM; zeros(ncovars,1)];
        
        % Set upper and lower bounds for the coefficients
        lb = [0; -100*ones(ncovars, 1)];
        ub = [1;  100*ones(ncovars, 1)];

        % Maximization is performed by minimizing the negative log likelihood
        [haz_param, logL, exit_flag, output, lambda, grad, hessian] = ...
             fmincon(@exp_like, param, [], [], [], [], lb, ub, [], options, ts, tf, event, covars);
         
    otherwise
        disp('Performing Maximum Likelihood with Weibull Distribution...');

        % Set the start parameters.  The first parameter is the magnitude of the 
        % baseline hazard, and the second is the shape parameter for the baseline 
        % hazard.  All other parameters are coefficients on the covariates.
        param = [KM; 1; zeros(ncovars,1)];
        
        % Set upper and lower bounds for the coefficients
        lb = [0; 0; -100*ones(ncovars, 1)];
        ub = [1; 8;  100*ones(ncovars, 1)];

        % Maximization is performed by minimizing the negative log likelihood
        [haz_param, logL, exit_flag, output, lambda, grad, hessian] = ...
             fmincon(@wei_like, param, [], [], [], [], lb, ub, [], options, ts, tf, event, covars);

end
         
% Calcualte the standard error of the coefficient estimates
param_se = sqrt(diag(hessian^-1));
t_stats  = haz_param./param_se;

disp(' ');
disp(strcat(['LnLike      ', num2str(logL,       '% 7.5f ')]));
disp(strcat(['haz_params  ', num2str(haz_param', '% 7.5f ')]));
disp(strcat(['param_se    ', num2str(param_se',  '% 7.5f ')]));
disp(strcat(['t_stats     ', num2str(t_stats',   '% 7.4f ')]));
disp(strcat(['gradient    ', num2str(grad',      '% 7.5f ')]));
disp(' ');
toc

end
