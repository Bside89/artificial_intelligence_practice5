function W = randInitializeWeights(L_in, L_out)

%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) atribui valores iniciais para 
%   par�metros em uma camada com L_in unidades de entrada e L_out unidades
%   de sa�da.
%

W = zeros(L_out, 1 + L_in);




epsilon_init = 0.12;
W = rand(L_out, 1+L_in)*2*epsilon_init - epsilon_init;


end
