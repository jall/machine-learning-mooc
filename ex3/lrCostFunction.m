function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
h = sigmoid(X * theta);
baseCost = (1 / m) * sum((-y' * log(h)) - ((1 - y)' * log(1 - h)));
costRegularisation = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);
J = baseCost + costRegularisation;

baseGrad = (1/m) * (X' * (h - y));
gradRegularisation = zeros(size(theta), 1);
gradRegularisation(2:end) = (lambda / m) * theta(2:end);
grad = baseGrad + gradRegularisation;

end
