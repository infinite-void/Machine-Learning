function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J=0;
if size(y,2)==m,
  y=y';
endif
if size(theta,1)==1,
  theta=theta';
endif
if size(X,1)==m,
  X=X';
endif
 H = theta' * X;
 temp = H' - y;
 temp = temp.^2;
 J = sum(temp)/(2*m);
end
