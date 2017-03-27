function [J, grad] = cost_function( nn_params, ...
                                    input_layer_size, ...
                                    hidden_layer_size, ...
                                    num_labels, ...
                                    X, y, lambda)
%COST_FUNCTION implementa a função de custo da rede neural
%   [J grad] = COST_FUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula a função de custo e o gradiente da rede.
%

% Não altere!!
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Variáveis úteis
m = size(X, 1);

% Você deve preencher as seguintes variáveis corretamente
%J = 0;
Theta1_grad = zeros(size(Theta1)); % gradiente de Theta1
Theta2_grad = zeros(size(Theta2)); % gradiente de Theta2

% Mudança de representação de y para um vetor Y
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
    Y(i, :)= I(y(i), :);
end

% =============== Sua implementação deve ser vir aqui ==================

% ==========================
% === Cálculo do custo J ===
% ==========================

% H é a função hypotehsis calculada em cada exemplo de X
% dim(H) = dim(Y)
H = zeros(m, num_labels);
for i=1:m
    H(i, :) = hyp(Theta1, Theta2, X(i, :));
end

M = -(Y.*log(H)) - (1 - Y).*log(1 - H);

J = sum(sum(M))/m;

% ==========================
% === Regularização de J ===
% ==========================

J = J + (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2)) + ...
                        sum(sum(Theta2(:, 2:end).^2)));
                    
% =====================================================
% === Cálculo de grad(J): Algoritmo Backpropagation ===
% =====================================================

for i=1:m,
    
    x = X(i, :);            % entrada da camada 1 (x)
    a1 = [1 ; x'];          % saída da camada 1 + bias 1
    z2 = Theta1*a1;         % entrada da camada intermediária 2
    a2 = [1 ; sigmoid(z2)]; % saída da camada intermediária + bias 1
    z3 = Theta2*a2;         % entrada da camada de saída 3
    a3 = sigmoid(z3);       % saída da camada de saída = h
    
    gamma3 = a3 - Y(i, :);
    gamma2 = ((Theta2')*gamma3).*sigmoidGradient(z2);
    
end

% Não altere esta linha!!
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
