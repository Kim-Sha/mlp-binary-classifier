import math
import numpy as np
import matplotlib.pyplot as plt
from activation_functions import sigmoid, sigmoid_backward, relu, relu_backward
from sklearn import preprocessing

class MultiLayerNN:
    """Brief class description
    
    Some more extensive description
    
    Attributes
    ----------
    X : numpy array
        Purpose of attr1.
    Y : numpy array
        Purpose of attr2.
    parameters : dict
    """
    
    def __init__(self, X, Y):
        """
        Please see help(MultiLayerNN) for more information.
        """
        self.X = X
        self.Y = Y
        self.parameters = {}
        self.final_cost = 0
    
    """
    INITIALIZATION
    """

    def initialize_parameters(self, layer_dimensions):
        """Example of docstring on the __init__ method.
        
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
        np.random.seed(3)
        parameters = {}
        L = len(layer_dimensions)

        for i in range(1, L):
            parameters['W' + str(i)] = np.random.randn(layer_dimensions[i], layer_dimensions[i-1]) * np.sqrt(2 / layer_dimensions[i-1])
            parameters['b' + str(i)] = np.zeros((layer_dimensions[i], 1))
        
            assert(parameters['W' + str(i)].shape == (layer_dimensions[i], 
                                                      layer_dimensions[i - 1]))
            assert(parameters['b' + str(i)].shape == (layer_dimensions[i], 1))
        
        return parameters
    
    def initialize_adam_parameters(self, parameters) :
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Parameters
        ----------
        parameters : dict
            parameters["W" + str(l)] = Wl; parameters["b" + str(l)] = bl
        
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

    def initialize_minibatches(self, X, Y, minibatch_size = 64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Parameters
        ----------
        X : numpy array
            Input data, of shape (input size, number of examples)
        Y : numpy array
            True "label" vector of shape (1, number of examples)
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
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/minibatch_size) # number of mini batches of size minibatch_size in your partitionning
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

    """
    FORWARD PROP
    """

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Parameters
        ----------
        A_prev : numpy array
            activations from previous layer (or input data): 
            (size of previous layer, number of examples)
        W : numpy array
            weights matrix of shape (current layer size, previous layer size)
        b : numpy array
            Bias vector of shape (current layer size, 1)
        activation : str
            Activation to be used in this layer, "sigmoid" or "relu"

        Returns
        -------
        A : numpy array
            Output of the activation function 
        cache : tuple
            "linear_cache" and "activation_cache" stored for computing 
            the backward pass efficiently
        """
        
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)

        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            A, activation_cache = relu(Z)
        
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Parameters
        ----------
        X : numpy array
            Output of initialize_parameters() of shape 
            (input size, number of examples)
        
        Returns
        -------
        AL : numpy array
            last post-activation value
        caches : list
            Every cache of linear_activation_forward() 
            (there are L-1 of them, indexed from 0 to L-1)
        """
        caches = []
        A = X
        L = len(parameters) // 2 
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev,
                                                      parameters['W' + str(l)],
                                                      parameters['b' + str(l)],
                                                      activation = "relu")
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A,
                                                   parameters['W' + str(L)],
                                                   parameters['b' + str(L)],
                                                   activation = "sigmoid")
        caches.append(cache)
        
        assert(AL.shape == (1, X.shape[1]))
                
        return AL, caches

    """
    COST FUNCTION
    """

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Parameters
        ----------
        AL : numpy array
            Probability vector corresponding to your label predictions 
            with shape (1, number of examples)
        Y : numpy array
            True "label" vector (for example: containing 0 if non-cat, 1 if cat) 
            with shape (1, number of examples)

        Returns
        -------
        cost : float
            cross-entropy cost
        """
        # Compute loss from aL and y.
        cost = -np.sum((np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y)))
        assert(cost.shape == ())
        
        return cost
    
    """
    BACKPROP 
    """

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Parameters
        ----------
        dA : numpy array
            post-activation gradient for current layer l 
        cache : tuple
            (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation : str
            Activation to be used in this layer, "sigmoid" or "relu"
        
        Returns
        -------
        dA_prev : numpy array
            Gradient of the cost wrt the activation (of the previous layer l-1), 
            same shape as A_prev
        dW : numpy array
            Gradient of the cost wrt W (current layer l), same shape as W
        db : numpy array
            Gradient of the cost wrt b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db
    
    def back_prop(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Parameters
        ----------
        AL : numpy array
            Output of the forward propagation (forward_prop())
        Y : numpy array
            True "label" vector (containing 0 if non-cat, 1 if cat)
        caches : list
            list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns
        -------
        grads : dict
            A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL,
                                                                                                        current_cache,
                                                                                                        "sigmoid")        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)],
                                                                        current_cache,
                                                                        "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    """
    UPDATE PARAMETERS
    """

    def gd_update(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Parameters
        ----------
        parameters : dict
        grads : dict
            Gradients output by back_prop()
        
        Returns
        -------
        parameters : dict
            Updated parameters:
                parameters["W" + str(l)] = ... 
                parameters["b" + str(l)] = ...
        """
        
        L = len(parameters) // 2 

        # Update rule for each parameter
        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        return parameters
    
    def adam_update(self, parameters, grads, v, s, t, learning_rate = 0.01,
                                    beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        """
        Update parameters using Adam
        
        Parameters
        ----------
        parameters : dict
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        grads : dict
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        v : dict
            Adam variable, moving average of the first gradient
        s : dict
            Adam variable, moving average of the squared gradient
        learning_rate : float
        beta1 : float
            Exponential decay hyperparameter for the first moment estimates 
        beta2 : float
            Exponential decay hyperparameter for the second moment estimates 
        epsilon : float
            hyperparameter preventing division by zero in Adam updates

        Returns
        -------
        parameters : dict
        v : dict
        s : dict
        """
        
        L = len(parameters) // 2  # number of layers in the neural networks
        v_corrected = {}          # Initializing first moment estimate
        s_corrected = {}          # Initializing second moment estimate
        
        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - np.power(beta1, t))
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - np.power(beta1, t))

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.power(grads["dW" + str(l+1)], 2)
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.power(grads["db" + str(l+1)], 2)

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - np.power(beta2, t))
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - np.power(beta2, t))

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

        return parameters, v, s

    """
    MODEL
    """
    
    def fit_binary(self, layer_dimensions, optimizer = "adam", learning_rate = 0.0007,
                   minibatched = True, minibatch_size = 64, beta = 0.9, beta1 = 0.9, 
                   beta2 = 0.999, epsilon = 1e-8, num_epochs = 10000, print_cost = True):
        """
        L-layer neural network model which can be run in different optimizer modes.
        
        Parameters
        ----------
        layer_dimensions : list
        optimizer : string {"gd", "adam"}
            Optimization algorithm to use
        learning_rate : float
        minibatched : boolean
            Whether to implement mini-batches
        minibatch_size : float
        beta : float
            Momentum hyperparameter
        beta1 : float
            Exponential decay hyperparameter for the past gradients estimates 
        beta2 : float
            Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon : float
            hyperparameter preventing division by zero in Adam updates
        num_epochs : int
            number of epoch iterations
        print_cost : boolean
            True to print the cost every 1000 epochs

        Returns
        -------
        parameters : dict
        """

        L = len(layer_dimensions) # number of layers in the neural networks
        costs = []                # to keep track of the cost
        t = 0                     # initializing counter required for Adam update
        seed = 10
        m = self.X.shape[1]       # number of training examples
        
        # Initialize parameters
        parameters = self.initialize_parameters(layer_dimensions)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no initialization required for gradient descent
        elif optimizer == "adam":
            v, s = self.initialize_adam_parameters(parameters)
        else:
            raise ValueError("The only supported optimizer modes are 'gd' and 'adam'.")
        
        # Optimization loop
        for i in range(num_epochs):
            
            cost_total = 0

            if minibatched:

                # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
                seed = seed + 1
                minibatches = self.initialize_minibatches(self.X, self.Y, minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                    AL, caches = self.forward_prop(minibatch_X, parameters)

                    # Compute cost and add to the cost total
                    cost_total += self.compute_cost(AL, minibatch_Y)

                    # Backward propagation
                    grads = self.back_prop(AL, minibatch_Y, caches)

                    # Update parameters
                    if optimizer == "gd":
                        parameters = self.gd_update(parameters, grads, learning_rate)
                    elif optimizer == "adam":
                        t = t + 1 # Adam counter
                        parameters, v, s = self.adam_update(parameters, grads, v, s,
                                                            t, learning_rate, beta1, beta2,  epsilon)
            else:
                AL, caches = self.forward_prop(self.X, parameters)
                cost_total += self.compute_cost(AL, self.Y)
                grads = self.back_prop(AL, self.Y, caches)
                parameters = self.gd_update(parameters, grads, learning_rate)

            cost_avg = cost_total / m
            
            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0: 
                print("Cost after epoch %i: %f" %(i, cost_avg))
            if i % 100 == 0:
                costs.append(cost_avg)
        
        # Update instance attributes
        self.final_cost = cost_avg
        self.parameters = parameters

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    """
    PREDICT
    """

    def predict_binary(self, X, y):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        p = np.zeros((1,m))
        
        # Forward propagation
        probabilities = self.forward_prop(X, self.parameters)[0]

        # convert probabilities to 0/1 predictions
        for i in range(0, probabilities.shape[1]):
            if probabilities[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p


# X_train = np.loadtxt("predict-moons/data-moons/x_train.csv")
# y_train = np.loadtxt("predict-moons/data-moons/y_train.csv")
# y_train = y_train.reshape(1, y_train.shape[0])
# moons_nn = MultiLayerNN(X = X_train, Y = y_train)
# moons_nn.fit_binary(layer_dimensions = [2, 5, 2, 1],
#                     optimizer = "adam")