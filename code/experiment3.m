% Universidade Federal de Mato Grosso
% Instituto de Engenharia
% Intelig�ncia Artificial - 2016/2
%
% Script Octave/MATLAB que deve ser utilizado para testar diferentes
% valores de lambda e avaliar as respectivas acur�cias.
% 

clear ; close all; clc;

%% Par�metros de ajuste do script

max_iter = 50;             % N�mero de itera��es (fmincg)
k = 3;                      % Par�metro k na valida��o cruzada

% Amostra contendo valores de lambda
lambda = [0 10^(-6) 1 2 3 4 10 10^3 10^6];

%% Par�metros da rede
input_layer_size  = 400;    % imagens de 20x20 
hidden_layer_size = 25;     % 25 unidades na camada intermedi�ria
num_labels = 10;            % 10 classe, de 1 a 10
                            % (note que o d�gito "0" foi mapeado para a classe 10)

%% =========== Carregando os Dados  =============

fprintf('Carregando os Dados ...\n')

load('ex5data.mat');
m = size(X, 1);

%% =================== Treinando e testando rede neural ===================
%
%

ml = size(lambda, 2);

options = optimset('MaxIter', max_iter);

c = cvpartition(y, 'k', k);

k_acc = zeros(ml, k);

list_costs = cell(ml, k);

accuracy = zeros(1, ml);

for i=1:ml,
    for j=1:k,

        % =========================
        % === Treinamento da NN ===
        % =========================

        fprintf('\nTreinando a rede neural... \n');
        fprintf('K-fold CV: k = %i\n', j);
        fprintf('lambda = %f\n\n', lambda(i));

        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

        initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

        costFunction = @(p) cost_function(p, ...
            input_layer_size, ...
            hidden_layer_size, ...
            num_labels, X(c.training(j), :), y(c.training(j)), lambda(i));

        % Fun��o de otimiza��o
        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

        Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
            hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
            num_labels, (hidden_layer_size + 1));

        list_costs{i, j} = cost; % Armazena o custo numa lista de matrizes
        
        % =======================
        % === Avalia��o da NN ===
        % =======================

        pred = predict(Theta1, Theta2, X(c.test(j), :));

        fprintf('\nTestando a rede neural... \n');
        fprintf('\nk: %4i | ', j);
        k_acc(i, j) = mean(double(pred == y(c.test(j)))) * 100;
        fprintf('Acur�cia: %f\n', k_acc(i, j));

    end
    
    accuracy(i) = mean(k_acc(i, :));
    fprintf('\nAcur�cia m�dia final do modelo (p/ lambda = %f): %f\n', ...
        lambda(i), accuracy(i));
    
end

% Corre��o do early-stop da fmincg, ajustando as dimens�es das matrizes de
% custo
for i=1:ml,
    for j=1:k,
        while size(list_costs{i, j}, 1) ~= max_iter,
            list_costs{i, j}(size(list_costs{i, j}, 1) + 1) = list_costs{i, j}(end);
        end
    end
end
    
if k == 3, % Pr�-definido para este experimento
    f = figure;
    for i=1:ml,
        v = [list_costs{i, 1} list_costs{i, 2} list_costs{i, 3}];
        cost = mean(v, 2);
        x = 1:size(cost, 1);
        p = plot(x, cost);
        p(1).LineWidth = 1.2;
        hold on;
    end
    title('Progresso da otimiza��o do custo em fmincg');
    xlabel('Itera��o');
    ylabel('Custo');
    hleg = legend(  'lambda = 0', 'lambda = 1e?06', ...
                    'lambda = 1', 'lambda = 2', ...
                    'lambda = 3', 'lambda = 4', ...
                    'lambda = 10', 'lambda = 1e+03', ...
                    'lambda = 1e+06');
    set(hleg,'Location','best');
end