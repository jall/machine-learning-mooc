function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

unregularised_cost = (1/2) * sum(sum(((X * Theta' - Y) .^ 2) .* R));
theta_regularisation = (lambda / 2) * sum(sum(Theta .^ 2));
x_regularisation = (lambda / 2) * sum(sum(X .^ 2));
J = unregularised_cost + theta_regularisation + x_regularisation;

for i = 1:num_movies
  idx = find(R(i, :) == 1);
  Theta_i = Theta(idx, :);
  Y_i =  Y(i, idx);
  X_i = X(i, :);
  X_grad_row = (X_i * Theta_i' - Y_i) * Theta_i;
  X_grad(i, :) = X_grad_row' + (lambda * X_i');
endfor

for j = 1:num_users
  idx = find(R(:, j) == 1);
  Theta_j = Theta(j, :);
  Y_j =  Y(idx, j);
  X_j = X(idx, :);
  Theta_grad_row =  X_j' * (X_j * Theta_j' - Y_j);
  Theta_grad(j, :) = Theta_grad_row' + (lambda * Theta_j);
endfor

grad = [X_grad(:); Theta_grad(:)];

end
