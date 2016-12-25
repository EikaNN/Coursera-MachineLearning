function [error_train, error_val] = ...
    learningCurveWithRandomExamples(X, y, Xval, yval, lambda, steps)
%LEARNINGCURVEWITHRANDOMEXAMPLES Generates the train and cross validation
%set errors needed to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVEWITHRANDOMEXAMPLES(X, y, Xval, yval, lambda, steps)
%       returns the train and cross validation set errors for a learning curve.
%       In particular, it returns two vectors of the same length - error_train
%       and error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%       In this implementation we select i random examples without replacement.
%       We repeat the process a given number of steps and take the averaged error
%       of both training and validation examples.
%
%       NOTE: If you run this function it is better to disable prints from fmincg.m
%             since printing slows down execution a lot.
%             A lookup table has been implemented to avoid recomputing theta for
%             the same subset of examples.
%

% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

lookup_theta = containers.Map('KeyType','char','ValueType','any');
lookup_calls = 0;
theta_calls = 0;

for s = 1:steps

    for i = 1:m

        [subsetX, idx] = datasample(X, i, 'Replace', false);
        subsety = y(idx);
        [subsetXval, idxval] = datasample(Xval, i, 'Replace', false);
        subsetyval = yval(idxval);

        key = mat2str(sort(idx));
        subsetAlreadyTrained = isKey(lookup_theta, key);
        if subsetAlreadyTrained
            theta = lookup_theta(key);
            lookup_calls = lookup_calls + 1;
        else
            theta = trainLinearReg(subsetX, subsety, lambda);
            lookup_theta(key) = theta;
            theta_calls = theta_calls + 1;
        end

        error_train(i) = error_train(i) + linearRegCostFunction(subsetX, subsety, theta, 0);
        error_val(i) = error_val(i) + linearRegCostFunction(Xval, yval, theta, 0);

    end

end

% Average the results
error_train = error_train / steps;
error_val = error_val / steps;

fprintf('Number of calls to trainLinearReg without lookup table: %d\n', steps*m);
fprintf('Number of calls to trainLinearReg with lookup table: %d\n', theta_calls);
fprintf('Number of calls to trainLinearReg saved by lookup table: %d\n\n', lookup_calls);

end
