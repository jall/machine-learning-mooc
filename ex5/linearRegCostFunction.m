function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

[J_unregularized, grad_unregularized] = unregularisedCostFunction(theta, X, y);
theta_sans_0 = theta(2:end, :);

J = J_unregularized + ((lambda / (2 * m)) * sum(theta_sans_0 .^ 2));

grad = grad_unregularized;
% Only modify grad values for j > 1, as j = 1 does not need a lambda alteration.
grad(2:end, :) = grad(2:end, :) + ((lambda / m) * theta_sans_0);

function [J, grad] = unregularisedCostFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

h = X * theta;
J = sum((h - y) .^2) * (1 / (2 * m));
grad = (1/m) * (X' * (h - y));

end











% =========================================================================

grad = grad(:);

end
