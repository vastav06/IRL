from collections import defaultdict
#from statistics import linear_regression

import numpy as np

from forwardPass import forward_pass


def backprop(nn, X, Y, loss):
    """
    This function takes as input a neural network, nn, and input-output
    pairs, <X,Y>, and computes the gradient of the loss for the neural
    network's current predictions. The code assumes that layers,
    {1,2,...,l-1}, have ReLU activations and the final layer has a linear
    activation function.

    Args:
        nn: The weights and biases of the neural network. nn[i][0] corresponds to the weights for the ith layer and
            nn[i][1] corresponds to the biases for the ith layer
        X: This matrix is d x n matrix and contains the input features for n examples, each with d features.
        Y: This term is an 1 x n vector of labels for n examples.
        loss: 'MSE' or 'Softmax'

    Returns: A cell of dimension num_layers where num_layers is the number of layers in neural network, nn. grad[i][0]
             corresponds to the gradients for ith layer weights, and grad[i][1] corresponds to the gradient for ith
             layer bias.

    """
    # for PingPong
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 6
    assert len(X.shape) == 2

    num_layers = len(nn)                                            # Get the number of layers of our neural network
    batch_size = X.shape[1]

    Y_hat, outputs, linear_outputs = forward_pass(nn, X, final_softmax=(loss == 'Softmax'), full_return=True)  # Perform the forward pass on the neural network
    delta = [0.0] * num_layers                                      # Initialize the cell to store the error at level i
    # propagate error
    for i in reversed(range(num_layers)):                           # Iterate over all layers
        if i == num_layers - 1:
            if loss == 'MSE':
                """
                The error for the output layer is simply the NEGATIVE difference
                between the targets, Y, and the predictions, Y_hat. Note that the
                use of this difference is based upon an assumption that we are
                applying an MSE loss to train our neural network.
                """
                delta[i] = (-(Y-Y_hat))# INSERT CODE HERE
            elif loss == 'Softmax':
                a = Y# Get the action that was sampled, a, which is the third input to this function
                # Here we do not have a ground truth Y. a is the index of the action we took.

                '''
                Compute the derivative term
                                /
                d output_j      \\   output_j * (1 - output_j)   if  j == k
                ---------- =    |
                d input_k       /   -1 * output_j * output_k
                                \\

                
                '''
                derivative=np.zeros(Y_hat.shape)
                               
                for j in range(Y_hat.shape[0]):
                    if j==int(a):
                        derivative[j]=Y_hat[j]*(1-Y_hat[j])
                    else:
                        derivative[j]=-1*(Y_hat[j]*Y_hat[a])
                
             
                
                                   
                delta[i] = derivative # INSERT CODE HERE multiply q(s,t,at)
        else:
            linear_output = linear_outputs[i].copy()
            """
            ``derivative'' is an  n^{(i)} x 1 vector where element
             j \\in {1,..., n^{(i)}} is the derivative of the output of node j
             in layer i w.r.t. the input to that node. In other words, this
             term is the derivative of the activation function w.r.t. its
             inputs. We assume that the activation function for all non-final
             layers is ReLU, which is defined as

             output = f(input) =   / input   if  input >= 0
                                   \\  0       otherwise

             Therefore, the derivative is given by

                           /
             d output      \\   1   if  input >= 0
             -------- =    |
             d input       /   0   otherwise
                           \\
            """
                                   
            # MAY NEED TO WRITE HELPER CODE HERE BEFORE SETTING VALUE FOR delta
            derivative = linear_output  # Get the output of the activation function
            derivative[linear_output >= 0] = 1.0  # Compute the derivative for elements >= 0
            derivative[linear_output < 00] = 0.0 # Compute the derivative for elements < 0
            #print(nn[i+1][0].shape)
            delta[i] = np.matmul(nn[i+1][0].T, delta[i+1])*derivative  # INSERT CODE HERE    			  # Compute the error term for layer i

    # Compute the gradients of all of the neural network's weights using the error term, delta
    grad = defaultdict(list)  # Initialize a cell array, where cell i contains the gradients for the weight matrix in layer i of the neural network

    for i in range(num_layers):
        if i == 0:
            # Gradients for the first layer are calculated using the examples, X.
            grad[i].append(np.dot(delta[i], X.T))
            grad[i].append(np.dot(delta[i], np.ones((batch_size, 1))))
        else:
            # Gradients for subsequent layers are calculated using the output of the previous layers.
            grad[i].append(np.dot(delta[i], outputs[i - 1].T))
            grad[i].append(np.dot(delta[i], np.ones((batch_size, 1))))
        if np.isnan(grad[i][0]).any() or np.isnan(grad[i][1]).any():
            print("WARNING: Gradients/biases are nan")
            exit(-1)
    return grad

