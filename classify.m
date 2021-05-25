function c = classify(theta, X)
hypothesis = X*theta;
m = size(X,1);
c = zeros(m,1);
for i=1:size(hypothesis,1)
    if hypothesis(i,1)>=0.5     % Threshold of 0.5 for classification
        c(i,1)=1;
    end
end
end
