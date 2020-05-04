function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
if size(y,2)==m,
  y=y';
endif
if size(theta,1)==1,
  theta=theta';
endif
if size(X,1)==m,
  X=X';
endif
f = size(X,1);
for iter = 1:num_iters
H = theta' * X;
t = alpha/m;
H=H';
L=X';
     for i=1:f,
       theta(i,1) = theta(i,1) - (alpha*(1/m)*sum((H-y).*L(:,i:i)));
     endfor
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
