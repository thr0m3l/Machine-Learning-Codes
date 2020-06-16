function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m, 1) X];
a2 = ones(m, hidden_layer_size + 1);

ynew = zeros(m, num_labels);

for i = 1:m
    ynew(i, y(i, 1)) = 1;
end

%Forward prop and Back Prop
for i = 1:m
    %Forward
    z2 = X(i,:)*Theta1'; %(1x401)*(401x25)
    a2(i, 2:end) = sigmoid(z2);
    z3 = a2(i, :)*Theta2'; %(1x26)*(26x10)
    h(i, :) = sigmoid(z3); %(1x10)
    %Back
    del3 = h(i, :) - ynew(i, :); %(1x10)
    Theta2_grad += del3'*a2(i, :); %(10x26)
    z2 = z2'; z2 = [1; z2]; %(26x1)
    del2 = (Theta2'*del3').*sigmoidGradient(z2); %(26x10)*(10x1)
    del2 = del2(2:end, :);
    Theta1_grad += del2()*X(i, :);
end
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;


%Cost Function
for i = 1:m
    for k = 1:num_labels
        J += -ynew(i, k)*log(h(i, k)) - (1-ynew(i, k))*log(1-h(i,k));
    end
end

J = J/m;

%Regularization 
for j = 1:hidden_layer_size
    for k = 2:input_layer_size+1
        J += (lambda/(2*m))*(Theta1(j,k)^2);
    end
end



for j = 1:num_labels
    for k = 2:hidden_layer_size+1
        J += (lambda/(2*m))*(Theta2(j,k)^2);
    end
end

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
