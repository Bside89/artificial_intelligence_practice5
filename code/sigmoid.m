function g = sigmoid(z)
%SIGMOID Calcula a fun��o sigmoid
%   J = SIGMOID(z) calcula o resultado de sigmoid de z.

g = 1.0 ./ (1.0 + exp(-z));

end
