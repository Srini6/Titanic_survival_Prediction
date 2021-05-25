function [J,grad] = neuralNetwork(theta,input_layer_size,hidden_layer_size,output_layer_size,X,y,lambda)

theta1 = reshape(theta(1:(hidden_layer_size*(1+input_layer_size))),hidden_layer_size,(input_layer_size+1));
theta2 = reshape(theta(1+(hidden_layer_size*(1+input_layer_size)):end),output_layer_size,(1+hidden_layer_size));

% Theta gradient
theta1_grad = zeros(size(theta1)); 
theta2_grad = zeros(size(theta2));

J = 0; % Initialising cost value
m = size(X,1);

%Seperating y into 3 column vector - each column containing 1 for true
%value of each Iris variety
yVec = zeros(m,output_layer_size);
for i = 1:m
    yVec(i,y(i)+1) = 1;
end

for t = 1:m
    
    % For the input layer, where l=1:
    a1 = X(t,:)';
    
    % For the hidden layers, where l=2:
    z2 = theta1 * a1;
    a2 = [1;sigmoid(z2)];
    
    % For the output layer, where l=3:
    z3 = theta2 * a2;
    a3 = sigmoid(z3);
    
    % For the delta values:
    delta3 = a3- yVec(t,:)';
    
    delta2 = (theta2' * delta3) .* [1; sigmoidGradient(z2)];
    delta2 = delta2(2:end); % Taking of the bias row
    
    % delta_1 is not calculated because we do not associate error with the input
    
    % Big delta update
    theta1_grad = theta1_grad + delta2 * a1';
    theta2_grad = theta2_grad + delta3 * a2';
end

theta1_grad = (1/m) * theta1_grad + (lambda/m) * [zeros(size(theta1, 1), 1) theta1(:,2:end)];
theta2_grad = (1/m) * theta2_grad + (lambda/m) * [zeros(size(theta2, 1), 1) theta2(:,2:end)];
grad = [theta1_grad(:) ; theta2_grad(:)];

% Feedforward for cost function
% For the input layer, where l=1:
a1 = X;

% For the hidden layers, where l=2:  
z2 = a1 * theta1';
a2 = sigmoid(z2); 
a2 = [ones(size(a2,1), 1) a2];

% For the output layer, where l=3:
z3 = a2 * theta2';
a3 = sigmoid(z3);

J = (1/m) * sum(sum(-1 * yVec .* log(a3)-(1-yVec) .* log(1-a3)));
regulator = (sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2))) * (lambda/(2*m));
J = J + regulator;

end

