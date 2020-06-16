function h = predict(theta, X)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
h = sigmoid(X*theta);
for i = 1:length(h)
    if  (h(i, 1) >= 0.5)
        h(i,1) = 1;
    else
        h(i,1) = 0;
    endif
end


end
