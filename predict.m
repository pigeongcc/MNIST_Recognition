function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% since size(X) = 5000x400, we should add a column of ones to the beginning of X,
% so that X became actual a1

% Adding a bias unit to X = a1
% have to add a column
a1 = [ones(size(X, 1), 1) X];
% 5000x(400+1) = 5000x401

a2 = sigmoid(Theta1 * a1');
% 25x401 * 401x5000 = 25x5000

% Adding a bias unit to a2
% have to add a row
a2 = [ones(1,size(a2, 2)); a2];
% (25+1)x5000 = 26x5000

h = sigmoid(Theta2 * a2);
% 10x26 * 26x5000 = 10x5000 - OK

h = h';
% size(h) = 5000x10

% Making a prediction vector p(5000x1) based on hypothesis vector h(5000x10)

% assign p the index of the maximum in a col
for i=1:size(p, 1),
    [val p(i)] = max(h(i,:));
    % p(i) = ind;
end;

% =========================================================================


end
