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
%Theta1_grad = zeros(size(Theta1)); % gradiente de Theta1
%Theta2_grad = zeros(size(Theta2)); % gradiente de Theta2

% Mudança de representação de y para um vetor Y
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
    Y(i, :)= I(y(i), :);
end

% =============== Sua implementação deve vir aqui ==================

% Cálculo do custo J

% H é a função hypotehsis calculada em cada exemplo de X
% dim(H) = dim(Y)
H = hyp(Theta1, Theta2, X)';

J = sum(sum(-(Y.*log(H)) - (1 - Y).*log(1 - H)))/m;

% Regularização de J

if lambda ~= 0,
    J = J + (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2)) + ...
                            sum(sum(Theta2(:, 2:end).^2)));
end

% Cálculo de grad(J): Algoritmo Backpropagation

[A3, A2, A1, ~, Z2] = hyp(Theta1, Theta2, X);

G3 = A3 - Y'; % Gamma 3

T = (Theta2')*G3; % Matriz temporária (descarta a primeira coluna de G2)
T = T(2:end, :);

G2 = T.*sigmoidGradient(Z2); % Gamma 2

Theta1_grad = (1/m)*(G2*A1'); % Delta 1 (média)
Theta2_grad = (1/m)*(G3*A2'); % Delta 2 (média)

% Regularização do grad(J)

if lambda ~= 0,
    R1 = (lambda/m)*Theta1;
    R2 = (lambda/m)*Theta2;
    R1(:, 1) = zeros(size(R1, 1), 1); % Descarta a regularização do viés
    R2(:, 1) = zeros(size(R2, 1), 1); % Descarta a regularização do viés

    Theta1_grad = Theta1_grad + R1;
    Theta2_grad = Theta2_grad + R2;
end

% ==================================================================

% Não altere esta linha!!
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
