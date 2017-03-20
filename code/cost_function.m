function [J, grad] = cost_function(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%COST_FUNCTION implementa a fun��o de custo da rede neural
%   [J grad] = COST_FUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula a fun��o de custo  e o gradiente da rede. 
%

% N�o altere!!
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Vari�veis �teis
m = size(X, 1);

% Voc� deve preencher as seguintes vari�veis corretamente
J = 0;
Theta1_grad = zeros(size(Theta1)); % gradiente de Theta1
Theta2_grad = zeros(size(Theta2)); % gradiente de Theta2



% Mudan�a de representa��o de y para um vetor Y
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% =============== Sua implementa��o deve ser vir aqui ==================







% N�o altere esta linha!!
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
