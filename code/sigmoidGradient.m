function g = sigmoidGradient(z)
%SIGMOIDGRADIENT devolve o gradiente da fun��o sigmoid no ponto z
% z pode ser uma matriz, um vetor ou um n�mero real

g = sigmoid(z).*(1 - sigmoid(z));

end
