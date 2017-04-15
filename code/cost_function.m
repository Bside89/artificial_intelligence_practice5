function [J, grad] = cost_function( nn_params, ...
                                    input_layer_size, ...
                                    hidden_layer_size, ...
                                    num_labels, ...
                                    X, y, lambda)
%COST_FUNCTION implementa a fun��o de custo da rede neural
%   [J grad] = COST_FUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula a fun��o de custo e o gradiente da rede.
%

% N�o altere!!
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Vari�veis �teis
m = size(X, 1);

% Voc� deve preencher as seguintes vari�veis corretamente
% J = 0;
% Theta1_grad = zeros(size(Theta1)); % gradiente de Theta1
% Theta2_grad = zeros(size(Theta2)); % gradiente de Theta2

% Mudan�a de representa��o de y para um vetor Y
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
    Y(i, :)= I(y(i), :);
end

% =============== Sua implementa��o deve vir aqui ==================

% ==========================
% === C�lculo do custo J ===
% ==========================

% H � a fun��o hypotehsis calculada em cada exemplo de X
% dim(H) = dim(Y)
H = hyp(Theta1, Theta2, X)';

J = sum(sum(-(Y.*log(H)) - (1 - Y).*log(1 - H)))/m;

% ==========================
% === Regulariza��o de J ===
% ==========================

if lambda ~= 0,
    J = J + (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2)) + ...
                            sum(sum(Theta2(:, 2:end).^2)));
end

% =====================================================
% === C�lculo de grad(J): Algoritmo Backpropagation ===
% =====================================================

[A3, A2, A1, ~, Z2] = hyp(Theta1, Theta2, X);

G3 = A3 - Y'; % Gamma 3

T = (Theta2')*G3; % Matriz tempor�ria (descarta a primeira coluna de G2)
T = T(2:end, :);

G2 = T.*sigmoidGradient(Z2); % Gamma 2

Theta1_grad = (1/m)*(G2*A1'); % Delta 1 (m�dia)
Theta2_grad = (1/m)*(G3*A2'); % Delta 2 (m�dia)

% ================================
% === Regulariza��o do grad(J) ===
% ================================

if lambda ~= 0,
    R1 = (lambda/m)*Theta1;
    R2 = (lambda/m)*Theta2;
    R1(:, 1) = zeros(size(R1, 1), 1); % Descarta a regulariza��o do vi�s
    R2(:, 1) = zeros(size(R2, 1), 1); % Descarta a regulariza��o do vi�s

    Theta1_grad = Theta1_grad + R1;
    Theta2_grad = Theta2_grad + R2;
end

% ==================================================================

% N�o altere esta linha!!
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
