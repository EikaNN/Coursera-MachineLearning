function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Number of training examples
m = size(X, 1);

for i = 1:m

    [closest_centroid_index, lowest_centroid_distance] = deal(Inf);
    
    for k = 1:K
        
        centroid_distance = norm(X(i,:) - centroids(k,:))^2;

        if centroid_distance < lowest_centroid_distance
            closest_centroid_index = k;
            lowest_centroid_distance = centroid_distance;
        end

    end

    idx(i) = closest_centroid_index;

end

% =============================================================

end

