function norm_X = normalizeFeatures(X)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
mu = mean(X);
sigma = max(X)-min(X);
norm_X = (X-mu)./sigma;
end

