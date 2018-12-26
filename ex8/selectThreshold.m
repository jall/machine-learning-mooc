function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    predictions = (pval < epsilon);

    truePositives = sum((predictions == 1) & (yval == 1));
    falsePositives = sum((predictions == 1) & (yval == 0));
    trueNegatives = sum((predictions == 0) & (yval == 0));
    falseNegatives = sum((predictions == 0) & (yval == 1));

    if (truePositives + falsePositives) ~= 0
      precision = truePositives / (truePositives + falsePositives);
    else
      precision = 0;
    endif

    if (truePositives + falseNegatives) ~= 0
      recall = truePositives / (truePositives + falseNegatives);
    else
      recall = 0;
    endif

    if (precision + recall) ~= 0
      F1 = (2 * precision * recall) / (precision + recall);
    else
      F1 = 0;
    endif

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
