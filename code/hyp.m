function [ A3, A2, A1, Z3, Z2 ] = hyp( Theta1, Theta2, X )
%HYP Hypothesis calcula e devolve a saída da camada A3 da rede neural,
%utilizando a arquitetura dada.
%   Sendo:  A1 -> 400 saídas
%           A2 -> 25 saídas
%           A3 -> 10 saídas
%   Theta1 e Theta2 são os parâmetros da rede e X é uma matriz contendo, em
%   cada uma de suas linhas, um vetor contendo a camada de entrada x para
%   cada exemplo.

% Camada de entrada (a.k.a. x) + bias 1
A1 = [ones(1, size(X, 1)) ; X'];

Z2 = Theta1*A1;

% Camada intermediária + bias 1
A2 = [ones(1, size(Z2, 2)) ; sigmoid(Z2)];  

Z3 = Theta2*A2;

% Camada de saída = h
A3 = sigmoid(Z3);

end
