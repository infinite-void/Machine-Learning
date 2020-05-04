function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
H = X*theta;
J = sum (((H-y).^2))/(2*m)+ lambda * (sum(theta.^2)-theta(1,1)^2)/(2*m);
theta(1,1)=0;
grad = X'*(H-y)/m + (lambda*theta)/m ;

grad = grad(:);



end
