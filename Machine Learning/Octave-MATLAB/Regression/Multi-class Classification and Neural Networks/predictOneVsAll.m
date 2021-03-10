function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

%hamza! the X matric is a 5000by400 so we have 5000 examples and 400 features for each
%hamza! the all_theta is a k*numberOffeatures in our case it is 10by400 10 is k and 400 is number of feautres equal to number of features of each example
%hamza! X is a 5000by400 matrix where we have 5000 examples aka rows and each example have 400 features aka columns and all_theta is a 10by 400 matrix where  where we have 10 hypotheses each of 400 features  
%hamza! hence when we do X*all_theta' we get matrix of 400*10 where 400 is the number examples and 10 represent each example calulated with all 10 hypotheses so each row is the calcultion of single example with all 10 hypotheses
%hamza! ironically [~,p]=max(a, [], 2) outputs the number of column that has max value and number of column is our label number genius

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
result=sigmoid(X*all_theta');

[~, p]=max(result, [], 2);




%[probability indices] = max(sigmoid(X*all_theta'));
%p = indices';
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       







% =========================================================================


end
