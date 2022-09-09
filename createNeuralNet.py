import numpy as np


def create_neural_net(numNodesPerLayer, numInputDims, numOutputDims):
    """
    This function takes as input the number of nodes per hidden layer as well
    as the size of the input and outputs of the neural network and returns a
    randomly initialized neural network. Weights for the network are
    generated randomly using the method of He et al. ICCV'15.

    Args:
        numNodesPerLayer: This list contains natural numbers for the quantity of nodes contained in each hidden layer.
        numInputDims: This number represents the cardinality of the input vector for the neural network.
        numOutputDims: This number represents the cardinality of the output vector of the neural network.

    Returns: the neural network created
    """
    nn = []
    num_layers = len(numNodesPerLayer)
    for i in range(num_layers + 1):
        if i == 0:
            # Use numInputDims for the input size
            nn.append([np.random.random((numNodesPerLayer[i], numInputDims)) * np.sqrt(2.0 / numInputDims), np.zeros((numNodesPerLayer[i], 1))])
        elif i == num_layers:
            # Use numOutputDims for the output size
            nn.append([np.random.random((numOutputDims, numNodesPerLayer[i-1])) * np.sqrt(2.0 / numNodesPerLayer[i-1]), np.zeros((numOutputDims, 1))])
        else:
            nn.append([np.random.random((numNodesPerLayer[i], numNodesPerLayer[i-1])) * np.sqrt(2.0 / numNodesPerLayer[i-1]), np.zeros((numNodesPerLayer[i], 1))])
    return nn
