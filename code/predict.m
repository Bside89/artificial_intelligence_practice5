function p = predict(Theta1, Theta2, X)
%PREDICT classifica um conjunto de amostras utilizando os par�metros da 
%rede (Theta1 e Theta2)
%   p = PREDICT(Theta1, Theta2, X) devolve um vetor com a classse de 
%   cada amostra do conjunto X.

% Variaveis �teis
% m = size(X, 1);
% num_classes = size(Theta2, 1);

% Voc� deve preencher o seguinte vetor corretamente com a classe de cada
% amostra
% p = zeros(m, 1);

% Utiliza o valor com maior probabilidade de ser 1
% Note que isto � o mesmo que calcular max(a3)

[~, p] = max(hyp(Theta1, Theta2, X), [], 1);

p = p';

end
