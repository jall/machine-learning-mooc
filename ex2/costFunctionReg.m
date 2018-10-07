function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

[J_unregularized, grad_unregularized] = costFunction(theta, X, y);
theta_sans_0 = theta(2:end, :);

J = J_unregularized + ((lambda / (2 * m)) * sum(theta_sans_0 .^ 2));

grad = grad_unregularized;
% Only modify grad values for j > 1, as j = 1 does not need a lambda alteration.
grad(2:end, :) = grad(2:end, :) + ((lambda / m) * theta_sans_0);

end
