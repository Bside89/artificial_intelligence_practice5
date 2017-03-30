function g = sigmoidGradient(z)
%SIGMOIDGRADIENT devolve o gradiente da fun��o sigmoid no ponto z
% z pode ser uma matriz, um vetor ou um n�mero real

s = sigmoid(z);

g = s.*(1 - s);

end
