function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y


m = length(y); % number of training examples


eqns=X*theta;
sqrOfDiff=(eqns-y).^2;

% You need to return the following variables correctly 
J = 1/(2*m)*sum(sqrOfDiff);
%J = 1/(2*m)*transpose(sqrOfDiff)*(sqrOfDiff); this thing is just like x^2=x^Tx
%we are caculting costfuntin just to see it is convex we are not gonna use this in gradient descent
% =========================================================================

end
