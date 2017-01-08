function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% Initialize the centroids to be random unique examples

% Use unique examples to prevent duplicate centroids
X_unique = unique(X, 'rows');

% Randomly reorder the indices of unique examples
randidx = randperm(size(X_unique, 1));

% Take the first K unique examples as centroids
centroids = X_unique(randidx(1:K), :);

% =============================================================

end

