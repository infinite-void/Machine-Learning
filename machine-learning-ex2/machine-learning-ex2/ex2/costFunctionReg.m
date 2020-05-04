function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); 
J = 0;
grad = zeros(size(theta));
X = X';
H = theta' * X;
H = sigmoid(H);
H = H';
th = theta(2:length(theta),:) .^2;

J = (sum( (y .* log(H)) + ((1-y) .* log(1-H)) ) / (-1*m))+((lambda/(2*m))*(sum(th)));
X = X';
c = X(:,1:1);
  grad(1) = sum( (H-y).*c )/m;
for i = 2:length(theta),
  c = X(:,i:i);
  grad(i) = (sum( (H-y).*c )/m)+((lambda/m)*theta(i,1)) ;

end
