function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Here are the sizes for the character recognition example, 
% a1: 5000x401 z2: 5000x25 a2: 5000x26 a3: 5000x10 d3: 5000x10 d2: 5000x25 
% Theta1, Delta1 and Theta1grad: 25x401 
% Theta2, Delta2 and Theta2grad: 10x26

% Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). 
% This is most easily done using an eye() matrix of size num_labels, 
% with vectorized indexing by 'y', as in "eye(num_labels)(y,:)". 

Y = eye(num_labels)(y,:); % 10x10 5000 x 10

% perform the forward propagation:
% a1 equals the X input matrix with a column of 1's added (bias units)
% z2 equals the product of a1 and Θ1
% a2 is the result of passing z2 through g() a2 then has a column of 1st added (bias units)
% z3 equals the product of a2 and Θ2
% a3 is the result of passing z3 through g()Cost Function, non-regularized

a1 = [ones(m, 1) X]; % 500x401
z2 = a1 * Theta1'; % % 500x401 x 25x401
a2old = sigmoid(z2); % 500 x 26
z3old = [ones(m, 1) a2old] * Theta2'; % 500 x 26 * 10x26
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';

a3 = sigmoid(z3); % 5000x10


% Compute the unregularized cost according to ex4.pdf (top of Page 5), 
% (I had a hard time understanding this equation mainly that 
% I had a misconception that y(i)k is a vector, instead it is just simply one number) using a3, 
% your ymatrix, and m (the number of training examples). 
% Cost should be a scalar value. If you get a vector of cost values, 
% you can sum that vector to get the cost.
% Remember to use element-wise multiplication with the log() function.
% (op1 is 10x5000, op2 is 5000x10)

% J = sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3), 2)) / m;
%J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);


% Compute the regularized component of the cost according to ex4.pdf Page 6, 
% using Θ1 and Θ2 (ignoring the columns of bias units), along with λ, and m. 
% The easiest method to do this is to compute the regularization terms separately, 
% then add them to the unregularized cost from Step 3.
% You can run ex4.m to check the regularized cost, 
%J = (1/m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));

% Add regularized error. Drop the bias terms in the 1st columns.
reg2 = (lambda / (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2));
reg2 = reg2 + (lambda / (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2));

J = J + reg2;

% -------------------------------------------------------------

% Now we work from the output layer back to the hidden layer, 
% calculating how bad the errors are. See ex4.pdf Page 9 for reference.
% δ3 equals the difference between a3 and the y_matrix.
% δ2 equals the product of δ3 and Θ2 (ignoring the Θ2 bias units), 
% then multiplied element-wise by the g′() of z2 (computed back in Step 2).
% Note that at this point, the instructions in ex4.pdf are specific to looping implementations, so the notation there is different.
% Δ2 equals the product of δ3 and a2. This step calculates the product and sum of the errors.
% Δ1 equals the product of δ2 and a1. This step calculates the product and sum of the errors.

% d3 is the difference between a3 and the y_matrix. 
% The dimensions are the same as both, (m x r).  5000 x 10

d3 = a3 .- Y; % 5000 x 10
%d2 =  d3 * Theta2(:, 2:end) .* a2(:, 2:end); % 5000x10 * 10x25
%Delta2 = d3' * a2; % 5000x10 x 5000x26
%Delta2 = Delta2(:, 2:end);
d2 = (d3*Theta2) .* a2;
d2 = d2(:, 2:end);

Delta1 = d2' * a1;
Delta2 = d3'*a2;

%Delta2 = Delta2 (:, 2:end);
% Now we calculate the non-regularized theta gradients, using the sums of the errors 
% we just computed. (see ex4.pdf bottom of Page 11)
% Θ1 gradient equals Δ1 scaled by 1/m
% Θ2 gradient equals Δ2 scaled by 1/m
% The ex4.m script will also perform gradient checking for you, using a smaller test case than the full character classification example. So if you're debugging your nnCostFunction() using the "keyboard" command during this, you'll suddenly be seeing some much smaller sizes of X and the Θvalues. Do not be alarmed.If the feedback provided to you by ex4.m for gradient checking seems OK, you can now submit Part 4 to the grader. 

Theta1_grad = (Delta1./m); % 5x4
Theta2_grad = (Delta2./m); % 3x4
%size(Theta1_grad)
%size(Theta2_grad)
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
