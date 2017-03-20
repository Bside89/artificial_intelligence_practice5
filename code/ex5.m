% Universidade Federal de Mato Grosso
% Instituto de Engenharia
% Intelig�ncia Artificial - 2016/2
%
% Script Octave/MATLAB que deve ser utilizado para testar a implementa��o 
% do exerc�cio.
% Este arquivo n�o deve ser  alterado.
%
% Voc� deve alterar apenas os arquivos: 
% 
%     predict.m
%     cost_function.m
%

clear ; close all; clc

%% Parametros da rede
input_layer_size  = 400;  % imagens de 20x20 
hidden_layer_size = 25;   % 25 unidades na camada intermediária
num_labels = 10;          % 10 classe, de 1 a 10
                          % (note que o digito "0" foi mapeado para a classe 10)

%% =========== Carregando e Visualisando os Dados  =============


fprintf('Carregando e Visualisando os Dados ...\n')

load('ex5data.mat');
m = size(X, 1);

% Seleciona 100 imagens aleatoriamente
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Programa parado. Digite enter para continuar.\n');
pause;


%% ================ Carregando Par�metros ================


fprintf('\nCarregando par�metros salvos ...\n')

load('ex5weights.mat');

nn_params = [Theta1(:) ; Theta2(:)];


%% ============= Parte 1: Testando a implementa��o da classifica��o =============
%  
%  

pred = predict(Theta1, Theta2, X);

fprintf('\nAcur�cia do treinamento: %f\n', mean(double(pred == y)) * 100);

fprintf('Programa parado. Digite enter para continuar.\n');
pause;


% Veja  10 exemplos aleatorios

%  Randomly permute examples
rp = randperm(m);

for i = 1:10
    % Display
    fprintf('\nMostrando Imagem\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nEsta imagem foi classificada como: %d (d�gito %d)\n', pred, mod(pred, 10));

    % Pause
	fprintf('Programa parado. Digite enter para continuar.\n');
    pause;
end



%% ======= Parte 2a: Testando implementa��o da fun��o custo (sem regularizacao) =======
%  
%
fprintf('\nFun��o de custo (sem regulariza��o)...\n')

% desconsidera a regulariza��o.
lambda = 0;

J = cost_function(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Custo calculado considerado os valores salvos em ex5weights: %f '...
         '\n(O valor correto deve ser pr�ximo de 0.287629)\n'], J);

fprintf('Programa parado. Digite enter para continuar.\n');
pause;

%% =============== Parte 2b: Testando implementa��o da fun��o custo (com regularizacao) ===============
%  

fprintf('\nFun��o de custo (com regulariza��o)...\n')

lambda = 1;

J = cost_function(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Custo calculado considerado os valores salvos em ex5weights: %f '...
         '\n(O valor correto deve ser pr�ximo de 0.383770)\n'], J);

fprintf('Programa parado. Digite enter para continuar.\n');
pause;




%% =============== Parte 2c: Testando implementa��o do Backpropagation (sem regulariza��o)===============
%  
%
fprintf('\nTestando Backpropagation (sem Regulariza��o) ... \n')

lambda = 0;
checkNNGradients;

fprintf('Programa parado. Digite enter para continuar.\n');
pause;


%% =============== Parte 2d: Testando implementa��o do Backpropagation (com regulariza��o)= ===============
%  
%

fprintf('\nTestando Backpropagation (com Regulariza��o) ... \n')

lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = cost_function(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCusto: %f ' ...
         '\n(O valor correto deve ser pr�ximo de 0.576051)\n\n'], debug_J);
