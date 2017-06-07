% Universidade Federal de Mato Grosso
% Instituto de Engenharia
% Inteligência Artificial - 2016/2
%
% Script Octave/MATLAB que deve ser utilizado para avaliar a acurácia da
% rede neural utilizando validação cruzada k-fold.
% 

clear; close all; clc;

if is_octave(), # Carrega os pacotes necessários
    pkg load statistics;
end

%% Parâmetros de ajuste do script

max_iter = 50;
lambda = 3;                 % Should also try different values of lambda
k = 3;                      % Parâmetro k na validação cruzada

%% Parâmetros da rede
input_layer_size  = 400;    % imagens de 20x20 
hidden_layer_size = 25;     % 25 unidades na camada intermediária
num_labels = 10;            % 10 classe, de 1 a 10 (note que o dígito "0" foi 
                            % mapeado para a classe 10)

%% === Carregando os dados =====================================================

fprintf('Carregando os Dados ...\n')

load('ex5data.mat');
m = size(X, 1);

%% === Treinando e testando rede neural ========================================
%
%
options = optimset('MaxIter', max_iter);

if isOctave,
    optcv = 'KFold';
else
    optcv = 'k';
end

c = cvpartition(y, optcv, k);

k_acc = zeros(1, k);

cost = zeros(max_iter, k);

for i=1:k,
    
    % Treinamento da NN
    
    fprintf('\nTreinando a rede neural... \n');
    fprintf('K-fold CV: k = %i\n\n', i);
    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    
    if isOctave,
        idx1 = training(c, i);
        idx2 = test(c, i);
    else
        idx1 = c.training(i);
        idx2 = c.test(i);
    end
    
    costFunction = @(p) cost_function(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, X(idx1, :), y(idx1), lambda);
    
    % Função de otimização
    [nn_params, cost(:, i)] = fmincg(costFunction, initial_nn_params, options);
    
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    
    % Avaliação da NN
    
    pred = predict(Theta1, Theta2, X(idx2, :));
    
    fprintf('\nTestando a rede neural... \n');
    fprintf('\nk: %4i | ', i);
    k_acc(i) = mean(double(pred == y(idx2))) * 100;
    fprintf('Acurácia: %f\n', k_acc(i));
    
end

accuracy = mean(k_acc);

fprintf('\nAcurácia média final do modelo: %f\n', accuracy);

if k == 3, % Pré-definido para este experimento
    x = 1:max_iter;
    f = figure;
    for i=1:k,
        p = plot(x, cost(:, i));
        if ~isOctave,
            p(1).LineWidth = 1.2;
        end
        hold on;
    end
    title('fmincg - Custo por iteração');
    xlabel('Iteração');
    ylabel('Custo');
    hleg = legend('Custo 1', 'Custo 2', 'Custo 3');
    if ~isOctave,
        set(hleg,'Location','best');
    end
end
