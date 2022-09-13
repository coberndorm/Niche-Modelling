function [lambda_opt, minCVE, lambdas, CVError] = k_fCV(Y,X,nlamb,parallel,plotting)
% DESCRIPTION:
% 'k_fCV' computes a k-Fold Cross-Validation procedure (k = 10 default) to
% detect the optimal shrinking parameter of a ridge regression for a
% predefined dataset. Optimal shrinking parameter is based on the minimum
% CV Error value.
%
% REQUIRED INPUTS:
%   Y : Vector of size 'n x 1' with the response variable.
%   X : Matrix of size 'n x p' with the predictor variables;
%       observations by rows and variables by columns.
%   nlamb (optional) : scalar, the number of lambdas to test in a
%                      predefined grid (default is 500).
%   plotting (optional) : Binary (0-1), make the 'CV_Error vs Log-Lambda'
%                         plot of the resultant k-Fold Cross Validation
%                         procedure (default is 0 - No plot).
%
% OUTPUTS:
%   lambda_opt : scalar with the optimal lambda by k-Fold Cross-Validation.
%   minCVE : scalar with the minimum 'CV Error' achieve with 'lambda_opt'.
%   lambdas: Vector of size 'nlamb x 1' with the sequence of lambdas.
%   CVError: Vector of size 'nlamb x 1' with the sequence of CV Errors,
%            according to each lambda.
%
% Author: Santiago Ortiz (sortiza2@eafit.edu.co)
% Date: 05/2021
%%
    if nargin < 5
      plotting = 0;
    end
    if nargin < 4
      parallel = false;
    end
    if nargin < 3
      nlamb = 500;
    end
    
    parforArg = 0;
    if parallel
        parforArg=inf;
    end
    
    lambdas = (10.^linspace(10,-4,nlamb))'; % Sequence of lambdas
    M = length(lambdas);
    k = 10; % 10-fold Cross Validation
    n = size(X,1);
    fold_size = floor(n/k); % Number of individuals per fold
    if fold_size < 6 % Conditional to guarantee homogeneity in folds when the sample size of the dataset is small
        k = 5;
        fold_size = floor(n/k);
        if fold_size < 3 % Conditional to guarantee homogeneity in folds when the sample size of the dataset is too small
            k = 3;
            fold_size = floor(n/k);
        end
    end
    A = repelem(1:(k-1),fold_size);
    B = repelem(k,n-((k-1)*fold_size));
    folds = [A,B]'; % Each individual belongs to a one fold
    CVError = zeros(M,1);
    parfor (j=1:M,parforArg) % Loop for each lambda
        MSE = zeros(k,1);
        for i=1:k % Loop for CV Error
            x_train = X(folds~=i,:); % Train predictors
            y_train = Y(folds~=i,:); % Train response
            x_test =  X(folds==i,:); % Test predictors
            y_test = Y(folds==i,:); % Test response
            Betas_R = ridge(y_train,x_train,lambdas(j),0); % Ridge Betas
            y_hat = x_test*Betas_R(2:end) + Betas_R(1); % Predicted response
            MSE(i) = mean((y_hat-y_test).^2); % MSE for the iteration
        end
        CVError(j) = mean(MSE); % Cross Validation Error measure
    end
    minCVE = min(CVError); % Minimum CV Error
    lambda_opt = lambdas(CVError==minCVE); % Optimal lambda*
    if plotting == 1
        figure;
        plot(log(lambdas),CVError,'.-r','markersize',15)
        hold on
        plot(log(lambda_opt),minCVE,'.b','markersize',25)
        plot([log(lambda_opt),log(lambda_opt)],[0,max(CVError)],'--c','linewidth',2)
        hold off
        title('k-fold Cross Validation - Lambda Selection')
        xlim([min(log(lambdas)),max(log(lambdas))])
        xlabel('Log-Lambda')
        ylabel('Cross-Validation Error (CVE)')
    end
end

