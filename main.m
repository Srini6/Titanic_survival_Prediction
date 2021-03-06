clear all;
close all;
clc;

%% Loading data from csv
train_data = readtable('train.csv');
test_data = readtable('test.csv');
gender_data = table2array(readtable('gender_submission.csv')); % Contains y_test 

%% Droping unnecessary columns in training and test data
% 1. Following information does not provide any relevant information to the
% classification algorithm about the survival of the passenger. 
% PassengerId ,Name, Ticket, Embarked
train_data.PassengerId = [];
train_data.Name = [];
train_data.Ticket = [];
train_data.Embarked = [];

test_data.PassengerId = [];
test_data.Name = [];
test_data.Ticket = [];
test_data.Embarked = [];
% 2. Following parameters are eventually represented by a single column
% which is a companion or co-passenger.
% SibSp, Parch

train_data.SibSp = [];
train_data.Parch = [];

test_data.SibSp = [];
test_data.Parch = [];
% 3. Following varables are a consolidated representation of Pclass. So
% do not contribute additional information to the ML algorithm.
% Fare, Cabin

train_data.Fare = [];
train_data.Cabin = [];
test_data.Fare = [];
test_data.Cabin = [];


train_data = table2array(train_data);
test_data = table2array(test_data);

%% Normalising data
train_data(:,2) = normalizeFeatures(train_data(:,2)); %Class
train_data(:,4) = normalizeFeatures(train_data(:,4)); %Age
train_data(:,5) = normalizeFeatures(train_data(:,5)); %Companion

test_data(:,1) = normalizeFeatures(test_data(:,1)); %Class
test_data(:,3) = normalizeFeatures(test_data(:,3)); %Age
test_data(:,4) = normalizeFeatures(test_data(:,4)); %Companion

%% Input and output
X_train=train_data(:,2:5); % Class(N) - Gender - Age(N) - Companion(N)
X_train = [ones(size(X_train, 1), 1) X_train]; % Adding bias
y_train=train_data(:,1); % Survival

X_test=test_data(:,:);
X_test = [ones(size(X_test, 1), 1) X_test];
y_test = gender_data(:,2);

%% Logistic regression
alpha = 2;
iterations = 500;

% Training the algorithm
[final_theta,J_history] = logistic(X_train,y_train,alpha,iterations);

% Classification of test data
y_prediction = classify(final_theta,X_test);
accuracy = mean(double(y_prediction == y_test)) * 100;

fprintf('\n Test Accuracy from Classification using Logistic regression : %f\n',accuracy);

%% Neural network
input_layer_size = 4;  % Input features
hidden_layer_size = 1; % Hidden layer with 1 neuron unit
output_layer_size = 2; % Survival

% Initialising theta with random initialization function
theta1 = randInitializeWeights(input_layer_size,hidden_layer_size); 
theta2 = randInitializeWeights(hidden_layer_size,output_layer_size);

nn_params = [theta1(:) ; theta2(:)]; % Making a column vector with theta values

%% Training neural network
options = optimset('MaxIter', 500); % Number of iteration
lambda = 1.5;

% Neural network - Feedforward and back propagation
costFunction = @(p) neuralNetwork(p, ...
    input_layer_size, ...
    hidden_layer_size, ...
    output_layer_size, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Trained Theta values
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    output_layer_size, (hidden_layer_size + 1));


%% %% Accuracy of Testing data
test_data = [gender_data(:,2) test_data]; % 
test_data = test_data(randperm(size(test_data, 1)), :); % Random shuffling
X_test = test_data(:,2:end);
y_test = test_data(:,1);

% Predicting y
prediction = predict(Theta1,Theta2,X_test);
prediction = prediction-1;
Accuracy_test = mean(double(prediction == y_test)) * 100;
fprintf('\n Test Accuracy from Classification using Neural Network: %f\n',Accuracy_test);
