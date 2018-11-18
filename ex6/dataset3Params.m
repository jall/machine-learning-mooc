function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% Test code to check mean error for various values of C and sigma

% Cs = [0.01 0.03 0.1 0.3 1 3 10]
% sigmas = [0.01 0.03 0.1 0.3 1 3 10]

% for sigma = sigmas
%   for C = Cs
%     model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%     predictions = svmPredict(model, Xval);
%     sigma
%     C
%     mean_error = mean(double(predictions ~= yval))
%   end
% end

C = 1;
sigma = 0.1;

end
