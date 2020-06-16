function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples
n = size(X, 2);
% You need to return the following variables correctly 
J = costFunction(theta, X, y) + lambda*(1/(2*m))*(sum(theta.^2)-theta(1,1)^2) ;
grad = zeros(n, 1);
[temp0, grad] = costFunction(theta,X, y);

for i = 2:n
    grad(i, 1) = grad(i, 1) + (lambda*(1/m))*theta(i,1);
end


end
