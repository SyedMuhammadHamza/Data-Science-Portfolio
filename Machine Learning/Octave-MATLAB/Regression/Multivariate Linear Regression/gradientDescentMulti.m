function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

J_history = zeros(num_iters, 1);
Dim=size(X);%X is a design vector so first column is of all ones
n=Dim(2);%number of features or columns
disp(n);
disp(theta)
pause;
sizeoftheta=size(theta);
if n==2
	   for iter = 1:num_iters
	       temp1=theta(1)-alpha*(1/m*(sum(X*theta-y)));
	       temp2=theta(2)-alpha*(1/m*(sum((X*theta-y).*X(:,2))));
	       theta(1)=temp1;
	       theta(2)=temp2;
		   J_history(iter) = computeCost(X, y, theta);
	    end
elseif n>2
	   for iter = 1:num_iters
	        temptheta=theta;
			for i=1:sizeoftheta
			    temptheta(i)=theta(i)-alpha*(1/m*(sum((X*theta-y).*X(:,i))));
			end 
			theta=temptheta;
			J_history(iter) = computeCost(X, y, theta);
	    end
else 
       disp(0);
end;

%----------------------------------------vectorized implementation--------------------------
%for iter = 1:num_iters
    %theta = theta - alpha * (1/m) * (((X*theta) - y)' * X)'; % Vectorized  here (')is transpose
    %J_history(iter) = computeCostMulti(X, y, theta);
%end
    % ============================================================

    % Save the cost J in every iteration    

end
