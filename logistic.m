function [theta,J_history] = logistic(X, y, alpha, iterations)
n = size(X,2)-1;    % Number of features
theta = ones(n+1, 1); % Initialising theta values to ones
m = length(y);      % Number of rows in dataset

for i = 1:iterations
    z = X * theta;          
    hypothesis = 1./(1+exp(-z));      % Sigmoid function
    theta = theta - (alpha/m)*X'*(hypothesis-y);
    J_history(i,1) = (1/m)*((-y)'*log(hypothesis)-(1-y)'*log(1-hypothesis));    % Cost function
end

figure;
plot(1:iterations,J_history);
xlabel('Number of iterations');
ylabel('Cost');
title('Cost function vs Iteration');
grid on;
end