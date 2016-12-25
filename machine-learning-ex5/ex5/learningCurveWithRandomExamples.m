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

% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for s = 1:steps

    for i = 1:m

        [subsetX, idx] = datasample(X, i, 'Replace', false);
        subsety = y(idx);
        [subsetXval, idxval] = datasample(Xval, i, 'Replace', false);
        subsetyval = yval(idxval);
        theta = trainLinearReg(subsetX, subsety, lambda);

        error_train(i) = error_train(i) + linearRegCostFunction(subsetX, subsety, theta, 0);
        error_val(i) = error_val(i) + linearRegCostFunction(Xval, yval, theta, 0);

    end

end

% Average the results
error_train = error_train / steps;
error_val = error_val / steps;

end
