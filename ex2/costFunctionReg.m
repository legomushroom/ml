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

% You need to return the following variables correctly
  %  J = 0; %#ok<NASGU>
  %  grad = zeros(size(theta)); %#ok<NASGU>
   %
  %  sig = sigmoid(X * theta);   % hypothesis logistic regression
  %  reg_term = sum(theta(2:end) .^ 2) * lambda / (2 * m);
   %
  %  J = mean((-y .* log(sig)) - ((1 - y) .* log(1 - sig))) + reg_term;
   %


sig = sigmoid(X * theta);

% cost function:
pos = -y .* log(sig);
neg = (1 - y) .* log(1 - sig);
reg = sum(theta(2:end) .^ 2) * (lambda/(2*m));
J = (sum( pos - neg ) ./ m) + reg;

theta_reg = theta;
theta_reg(1) = 0;

% gradient:
normalGrad = (1/m) .* sum( (sig - y) .* X )';
regGrad = (lambda/m) * theta_reg;
grad = normalGrad + regGrad;

% =============================================================

end
