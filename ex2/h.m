function [prediction] = h(x, theta)
  prediction = sigmoid( x * theta );
end
