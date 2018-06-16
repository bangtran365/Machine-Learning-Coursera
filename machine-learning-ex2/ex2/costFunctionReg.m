function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% old cost code
% for i = 1:m
%   J = J-y(i)*log(sigmoid(theta'*X(i,:)'))-(1-y(i))*log(1-sigmoid(theta'*X(i,:)'));
% end
% J =J/ m;

prediction =X*theta; % prediction of hypthesis on all m
J=1/(m) * sum(-y'*log(sigmoid(prediction))-(1-y)'*log(1-sigmoid(prediction)));



% add regularization [avoiding theta(1)] implement by loop
% for j = 2:size(theta)
  % J =J+ lambda * theta(j)^2 / (2*m);
% end


%Implement by vector
J= J+lambda *sum(theta(2:end).^2)/(2*m);

% 
% old gradient code
% for j = 1:size(grad)
%   for i = 1:m
%     grad(j) = grad(j)+(sigmoid(theta'*X(i,:)')-y(i))*X(i,j);
%   end
%   grad(j) =  grad(j)/m;
% end

% Gradient calculation by vector
grad=1/(m)*(X'*(sigmoid(prediction)-y));


% add regularization by looping
%for j = 2:size(grad)
  %grad(j) = grad(j)+(lambda/m)*theta(j);
%end

grad(2:end) = grad (2:end)+ (lambda/m)*theta(2:end);



% =============================================================

end
