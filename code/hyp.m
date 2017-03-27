function [ a3 ] = hyp( Theta1, Theta2, x )
%HYP Hypothesis calcula e devolve a saída da camada a3 da rede neural,
%utilizando a arquitetura dada.
%   Sendo:  a1 -> 400 saídas
%           a2 -> 25 saídas
%           a3 -> 10 saídas
%   Theta1 e Theta2 são os parâmetros da rede e x é a camada de entrada,
%   isto é, o a1.

a1 = [1 ; x'];                  % camada de entrada (a.k.a. x) + bias 1
a2 = [1 ; sigmoid(Theta1*a1)];  % camada intermediária + bias 1
a3 = sigmoid(Theta2*a2);        % camada de saída = h

end

