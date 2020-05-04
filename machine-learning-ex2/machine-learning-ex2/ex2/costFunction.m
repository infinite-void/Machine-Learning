function [J, grad] = costFunction(theta, X, y)

m = length(y); 
 
J = 0;
grad = zeros(size(theta));

X = X';

H = theta' * X;
H = sigmoid(H);
H = H';
J = sum( (y .* log(H)) + ((1-y) .* log(1-H)) ) / (-1*m);
X = X';
for i = 1:length(theta),
  c = X(:,i:i);
  grad(i) = sum( (H-y).*c )/m;
 
end
