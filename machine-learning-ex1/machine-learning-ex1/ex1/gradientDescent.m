function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
L = X;
X = X';
for iter = 1:num_iters
    H = theta' * X;
    H = H';
    theta(1,1) = theta(1,1) - (alpha*(1/m)*sum(H-y));
    theta(2,1) = theta(2,1) - (alpha*(1/m)*sum((H-y).*L(:,2:2)));
    J_history(iter) = computeCost(X, y, theta);

end

end
