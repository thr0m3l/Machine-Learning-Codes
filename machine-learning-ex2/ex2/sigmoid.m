function g = sigmoid(z)
g = zeros(size(z));
for i = 1:rows(z)
    for j = 1:columns(z)
        g(i,j) = (1/(1+exp(-z(i,j))));
    end
end

end
