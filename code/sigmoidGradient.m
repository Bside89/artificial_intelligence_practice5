function g = sigmoidGradient(z)
%SIGMOIDGRADIENT devolve o gradiente da função sigmoid no ponto z
% z pode ser uma matriz, um vetor ou um número real

s = sigmoid(z);

g = s.*(1 - s);

end
