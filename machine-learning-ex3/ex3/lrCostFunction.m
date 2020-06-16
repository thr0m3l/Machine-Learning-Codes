function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);
% You need to return the following variables correctly 
h = sigmoid(X*theta);

J = (1/m)*(-y'*log(h) - (1-y)'*log(1-h)) + lambda*(1/(2*m))*(sum(theta.^2)-theta(1,1)^2);

grad = (1/m)*X'*(h-y);

for i = 2:n
    grad(i, 1) = grad(i, 1) + (lambda*(1/m))*theta(i,1);
end



end
