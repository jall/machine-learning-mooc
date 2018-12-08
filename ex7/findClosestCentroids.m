function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);
m = size(X,1);

idx = zeros(m, 1);

for i = 1:size(X,1)
  shortestDistance = Inf;
  for k = 1:K
    distance = norm(X(i,:) - centroids(k,:), 2);
    if distance < shortestDistance
      shortestDistance = distance;
      idx(i) = k;
    endif
  endfor
endfor

end
