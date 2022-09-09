import numpy as np


def forward_pass(nn, X, final_softmax=False, full_return=False):
    """
    This function takes as input a neural network, nn, and inputs, X, and
    performs a forward pass on the neural network. The code assumes that
    layers, {1,2,...,l-1}, have ReLU activations and the final layer has a
    linear activation (or softmax if final_softmax=True) function.

    Args:
        nn: The weights and biases of the neural network. nn[i][0] corresponds to the weights for the ith layer and
            nn[i][1] corresponds to the biases for the ith layer
        X: This matrix is d x n matrix and contains the input features for n examples, each with d features.
        final_softmax: whether the final layer should have softmax activation or linear
        full_return: if True, returns all the intermediate results with final outputs; if False, returns final outputs.

    Returns: if True, returns all the intermediate results with final outputs; if False, return 1 x n vector of predicted labels for the n examples.
    """
    # for PingPong
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 6
    assert len(X.shape) == 2

    num_layers = len(nn)             # Get the number of layers of our neural network

    linear_outputs = []  # Outputs after linear transformation but before activation for each hidden layer. To be used in backprop.
    outputs = []         # Outputs after activation for each hidden layer.
    for i in range(num_layers):
        # Compute the result of linear transformation of layer i
        if i == 0:
            # z[0] = dot(W[0].T, X) + b[0]   (0-> first layer)
            linear_outputs.append(np.matmul(nn[i][0], X) + nn[i][1])
        else:
            # Subsequent z computations use output from previous layer
            # z[i] = dot(W[i].T, output[i - 1]) + b[i]
            linear_outputs.append(np.matmul(nn[i][0], outputs[i-1]) + nn[i][1])

        # Compute the result after activation of layer i
        if i < num_layers-1:
            # If layer i is not the output layer, then apply the ReLU activation function for the nodes at this layer.
            outputs.append(np.maximum(0,linear_outputs[i])) # INSERT CODE HERE relu
        else:
            if final_softmax:
                # Apply a softmax activation
                outputs.append(np.exp(linear_outputs[i]) / np.sum(np.exp(linear_outputs[i]))) # INSERT CODE HERE
            else:
                # Apply a linear activation (i.e., no activation)
                outputs.append(linear_outputs[i])  # INSERT CODE HERE

    Y_hat = outputs[-1]
    if full_return:
        return Y_hat, outputs, linear_outputs
    else:
        return Y_hat
