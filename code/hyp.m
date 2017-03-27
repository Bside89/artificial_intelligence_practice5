function [ a3 ] = hyp( Theta1, Theta2, x )
%HYP Hypothesis calcula e devolve a sa�da da camada a3 da rede neural,
%utilizando a arquitetura dada.
%   Sendo:  a1 -> 400 sa�das
%           a2 -> 25 sa�das
%           a3 -> 10 sa�das
%   Theta1 e Theta2 s�o os par�metros da rede e x � a camada de entrada,
%   isto �, o a1.

a1 = [1 ; x'];                  % camada de entrada (a.k.a. x) + bias 1
a2 = [1 ; sigmoid(Theta1*a1)];  % camada intermedi�ria + bias 1
a3 = sigmoid(Theta2*a2);        % camada de sa�da = h

end

