function [ A3, Z3, Z2 ] = hyp( Theta1, Theta2, X )
%HYP Hypothesis calcula e devolve a saída da camada a3 da rede neural,
%utilizando a arquitetura dada.
%   Sendo:  a1 -> 400 saídas
%           a2 -> 25 saídas
%           a3 -> 10 saídas
%   Theta1 e Theta2 são os parâmetros da rede e X é uma matriz contendo, em
%   cada uma de suas linhas, um vetor contendo a camada de entrada x para
%   cada exemplo.

A1 = [ones(1, size(X, 1)) ; X']; % camada de entrada (a.k.a. x) + bias 1
Z2 = Theta1*A1;
A2 = [ones(1, size(Z2, 2)) ; sigmoid(Z2)];  % camada intermediária + bias 1
Z3 = Theta2*A2;
A3 = sigmoid(Z3);        % camada de saída = h

end

