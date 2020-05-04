function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
g = g + e;
g = g .^ (-1.*z);
g = 1 + g;
g = 1 ./ g;

end
