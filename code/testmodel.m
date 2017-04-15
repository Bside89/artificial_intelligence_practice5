% Universidade Federal de Mato Grosso
% Instituto de Engenharia
% Inteligência Artificial - 2016/2
%
% Script Octave/MATLAB que deve ser utilizado para avaliar a acurácia da
% rede neural utilizando validação cruzada k-fold.
%
% Você DEVE alterar os valores da variável lambda (linha 16).
% 

clear ; close all; clc;

%% Parâmetros de ajuste do script

max_iter = 1000;
lambda = 3;                 % Should also try different values of lambda
k = 3;                      % Parâmetro k na validação cruzada

%% Parâmetros da rede
input_layer_size  = 400;    % imagens de 20x20 
hidden_layer_size = 25;     % 25 unidades na camada intermediária
num_labels = 10;            % 10 classe, de 1 a 10
                            % (note que o dígito "0" foi mapeado para a classe 10)

%% =========== Carregando os Dados  =============

fprintf('Carregando os Dados ...\n')

load('ex5data.mat');
m = size(X, 1);

%% =================== Treinando e testando rede neural ===================
%
%

options = optimset('MaxIter', max_iter);

c = cvpartition(y, 'k', k);

k_acc = zeros(1, k);

for i=1:k,
    
    % =========================
    % === Treinamento da NN ===
    % =========================
    
    fprintf('\nTreinando a rede neural... \n');
    fprintf('K-fold CV: k = %i\n\n', i);
    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    
    costFunction = @(p) cost_function(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, X(c.training(i), :), y(c.training(i)), lambda);
    
    % Função de otimização
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    
    % =======================
    % === Avaliação da NN ===
    % =======================
    
    pred = predict(Theta1, Theta2, X(c.test(i), :));
    
    fprintf('\nTestando a rede neural... \n');
    fprintf('\nk: %4i | ', i);
    k_acc(i) = mean(double(pred == y(c.test(i)))) * 100;
    fprintf('Acurácia: %f\n', k_acc(i));
    
end

accuracy = mean(k_acc);

fprintf('\nAcurácia média final do modelo: %f\n', accuracy);
