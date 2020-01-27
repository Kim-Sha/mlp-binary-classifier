import numpy as np
import math

def initialize_parameters(layer_dimensions):
        """
        Parameters
        ----------
        layer_dimensions : list
            Dimensions of each layer in the neural network
        
        Returns
        -------
        parameters: python dictionary
            Parameters "W1", "b1", ..., "WL", "bL":
                        Wi : weight matrix of shape (layer_dimensions[i],
                                                     layer_dimensions[i - 1])
                        bi : bias vector of shape (layer_dimensions[i], 1)
        """
        parameters = {}
        L = len(layer_dimensions)

        for i in range(1, L):
            parameters['W' + str(i)] = np.random.randn(layer_dimensions[i],
                layer_dimensions[i - 1]) * np.sqrt(2 / layer_dimensions[i - 1])
            parameters['b' + str(i)] = np.zeros((layer_dimensions[i], 1))
        
            assert(parameters['W' + str(i)].shape == (layer_dimensions[i], 
                                                      layer_dimensions[i - 1]))
            assert(parameters['b' + str(i)].shape == (layer_dimensions[i], 1))
        
        return parameters

def initialize_adam_parameters(parameters) :
        """
        Initializes v and s as two python dictionaries with:
            - keys: "dW1", "db1", ..., "dWL", "dbL" 
            - values: numpy arrays of zeros of the same shape as the 
                      corresponding gradients/parameters.
        
        Returns
        ------- 
        v : dict
            Exponentially weighted average of the gradients.
                v["dW" + str(l)] = ...
                v["db" + str(l)] = ...
        s : dict
            Exponentially weighted average of the squared gradients.
                s["dW" + str(l)] = ...
                s["db" + str(l)] = ...

        """
        
        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}
        
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
            s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
        return v, s

def initialize_minibatches(X, Y, minibatch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Parameters
        ----------
        minibatch_size : int
            Size of the mini-batches, integer
        
        Returns
        -------
        minibatches : list
            List of synchronous (minibatch_X, minibatch_Y)
        """
        
        np.random.seed(seed)
        m = X.shape[1] # number of training examples
        minibatches = []
            
        # Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        # Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m / minibatch_size)
        for k in range(0, num_complete_minibatches):
            minibatch_X = shuffled_X[:, k * minibatch_size : (k + 1) * minibatch_size]
            minibatch_Y = shuffled_Y[:, k * minibatch_size : (k + 1) * minibatch_size]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)
        
        # Handling the end case (last mini-batch < minibatch_size)
        if m % minibatch_size != 0:
            final_batch_size = m - minibatch_size * math.floor(m / minibatch_size)
            minibatch_X = shuffled_X[:, num_complete_minibatches * minibatch_size : num_complete_minibatches * minibatch_size + final_batch_size]
            minibatch_Y = shuffled_Y[:, num_complete_minibatches * minibatch_size : num_complete_minibatches * minibatch_size + final_batch_size]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)
        
        return minibatches