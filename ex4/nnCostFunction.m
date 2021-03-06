function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
y = convertToUnitColumnVectors(y', num_labels);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

m = size(X, 1);

% Feed forward
A_1 = [ones(1, m); X'];
Z_2 = Theta1 * A_1;
A_2 = [ones(1, m); sigmoid(Z_2)];
Z_3 = Theta2 * A_2;
A_3 = sigmoid(Z_3);

H = A_3;

unregularisedCost = 0;
for i = 1:m
  y_i = y(:, i);
  h_i = H(:, i);
  costOfExample = sum((-y_i' * log(h_i)) - ((1 - y_i)' * log(1 - h_i)));
  unregularisedCost += costOfExample;
endfor
unregularisedCost *= (1 / m);

regularisationCost = (lambda / (2 * m)) * (elementWiseSumOfSquares(Theta1(:, 2:end)) + elementWiseSumOfSquares(Theta2(:, 2:end)));

J = unregularisedCost + regularisationCost;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

for t = 1:m
  a_1 = A_1(:, t);
  a_2 = A_2(:, t);
  a_3 = A_3(:, t);

  z_2 = Z_2(:, t);

  delta_3 = a_3 - y(:, t);
  delta_2 = (Theta2' * delta_3) .* a_2 .* (1 - a_2);

  Theta2_grad += delta_3 * a_2';
  Theta1_grad += delta_2(2:end) * a_1';
endfor

Theta2_grad /= m;
Theta1_grad /= m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

Theta1_regularised = zeros(size(Theta1));
Theta1_regularised(:, 2:end) = (lambda / m) * Theta1(:, 2:end);
Theta1_grad += Theta1_regularised;

Theta2_regularised = zeros(size(Theta2));
Theta2_regularised(:, 2:end) = (lambda / m) * Theta2(:, 2:end);
Theta2_grad += Theta2_regularised;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


function A = convertToUnitColumnVectors(X, column_vector_length)

A_but_1D = cell2mat(arrayfun(@(x) toUnitColumnVector(x, column_vector_length), X, "UniformOutput", false));
[width, height] = size(A_but_1D);
A = reshape(A_but_1D(:)(:), width, height);

end


function v = toUnitColumnVector(idx, len)

v = zeros(len, 1);
v(idx) = 1;

end


function total = elementWiseSumOfSquares(X)

X_squared = X .^ 2;
total = sum(X_squared(:));

end
