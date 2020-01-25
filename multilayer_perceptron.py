import math
import numpy as np
import matplotlib.pyplot as plt
from propagation_helpers import linear_activation_forward, linear_activation_backward
from sklearn import preprocessing

class MultiLayerNN:
    """
    Multi-layer perceptron (MLP) / deep feedforward neural network (DFFNN) to 
    tackle binary classification problems.
    
    Attributes
    ----------
    X : numpy array
        Input feature set used in training, of shape 
        (input size, number of examples)
    Y : numpy array
        Labeled outputs of shape (1, number of examples) to train NN against,
        given the set of input features X
        
    parameters : dict
        Tracker for the weight and bias terms learned by the NN
    final_cost : float
        Tracker for the cost computed from the loss function
    """
    
    def __init__(self, X, Y):
        """
        Please see help(MultiLayerNN) for more information.
        """
        self.X = X
        self.Y = Y
        self.sample_size = X.shape[1]
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
    
    def initialize_adam_parameters(self) :
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
        
        L = len(self.parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}
        
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros(self.parameters["W" + str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(self.parameters["b" + str(l+1)].shape)
            s["dW" + str(l+1)] = np.zeros(self.parameters["W" + str(l+1)].shape)
            s["db" + str(l+1)] = np.zeros(self.parameters["b" + str(l+1)].shape)
        
        return v, s

    def initialize_minibatches(self, minibatch_size = 64, seed = 0):
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
        m = self.X.shape[1] # number of training examples
        minibatches = []
            
        # Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = self.X[:, permutation]
        shuffled_Y = self.Y[:, permutation].reshape((1, m))

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

    """
    FORWARD PROP
    """

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
            A, cache = linear_activation_forward(A_prev,
                                                      parameters['W' + str(l)],
                                                      parameters['b' + str(l)],
                                                      activation = "relu")
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = linear_activation_forward(A,
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
        total_cost : float
            cross-entropy cost
        """
        # Compute loss from aL and y.
        log_cost = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
        total_cost = np.sum(log_cost)
        assert(total_cost.shape == ())
        
        return total_cost
    
    """
    BACKPROP 
    """
    
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
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                        current_cache,
                                                                                                        "sigmoid")        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)],
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
        v_new = {}          # Initializing first moment estimate
        s_new = {}          # Initializing second moment estimate
        
        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_new".
            v_new["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - np.power(beta1, t))
            v_new["db" + str(l+1)] = v["db" + str(l+1)] / (1 - np.power(beta1, t))

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.power(grads["dW" + str(l+1)], 2)
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.power(grads["db" + str(l+1)], 2)

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_new".
            s_new["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - np.power(beta2, t))
            s_new["db" + str(l+1)] = s["db" + str(l+1)] / (1 - np.power(beta2, t))

            # Update parameters. Inputs: "parameters, learning_rate, v_new, s_new, epsilon". Output: "parameters".
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_new["dW" + str(l+1)] / (np.sqrt(s_new["dW" + str(l+1)]) + epsilon)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_new["db" + str(l+1)] / (np.sqrt(s_new["db" + str(l+1)]) + epsilon)

        return parameters, v, s

    """
    MODEL
    """
    
    def fit_binary(self, layer_dimensions, optimizer = "adam", learning_rate = 0.025,
                   learning_decay_rate = 1e-7, minibatched = True, 
                   minibatch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999,
                   epsilon = 1e-8, num_epochs = 10000, print_cost = True):
        """
        L-layer neural network model which can be run in different optimizer modes.
        
        Parameters
        ----------
        layer_dimensions : list
        optimizer : string {"gd", "adam"}
            Optimization algorithm to use
        learning_rate : float
        learning_decay_rate : float
            Rate at which to decay the learning_rate
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
        # Check that layer dimensions are passed in correctly
        if layer_dimensions[0] != self.X.shape[0]:
            raise ValueError('''Incorrect dimensions for input layer. Condition:\
                layer_dimensions[0] = X_train.shape[0] must be satisfied.''')
        if layer_dimensions[-1] != 1:
            raise ValueError('''Incorrect dimensions for output layer of binary\
                classifier. Condition: layer_dimesnsions[-1] = 1 must be \
                    satisfied for binary classifier''')

        L = len(layer_dimensions) # number of layers in the neural networks
        costs = []                # to keep track of the cost
        learning_rates = []       # to keep track of decaying learning rate
        t = 0                     # initializing counter required for Adam update
        seed = 10
        m = self.X.shape[1]       # number of training examples
        
        # Initialize parameters
        self.parameters = self.initialize_parameters(layer_dimensions)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no optimizer used for vanilla gradient descent
        elif optimizer == "adam":
            v, s = self.initialize_adam_parameters()
        else:
            raise ValueError("The only supported optimizer modes are 'gd' and 'adam'.")
        
        # Optimization loop
        for i in range(num_epochs):
            
            cost_total = 0

            if minibatched:

                # Define the random minibatches. We increment the seed to 
                # reshuffle differently the dataset after each epoch
                seed = seed + 1
                minibatches = self.initialize_minibatches(minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                    AL, caches = self.forward_prop(minibatch_X, self.parameters)

                    # Compute cost and add to the cost total
                    cost_total += self.compute_cost(AL, minibatch_Y)

                    # Backward propagation
                    grads = self.back_prop(AL, minibatch_Y, caches)

                    # Update parameters
                    if optimizer == "gd":
                        self.parameters = self.gd_update(self.parameters, grads,
                                                         learning_rate)
                    elif optimizer == "adam":
                        t = t + 1 # Adam counter
                        self.parameters, v, s = self.adam_update(self.parameters,
                            grads, v, s, t, learning_rate, beta1, beta2, epsilon)
            else:
                AL, caches = self.forward_prop(self.X, self.parameters)
                cost_total += self.compute_cost(AL, self.Y)
                grads = self.back_prop(AL, self.Y, caches)
                self.parameters = self.gd_update(self.parameters, grads, learning_rate)

            # Print the cost every 1000 epoch
            cost_avg = cost_total / m
            if print_cost and i % 1000 == 0: 
                print("Cost after epoch %i: %f" %(i, cost_avg))
                print("Learning rate after epoch %i: %f" %(i, learning_rate))
            if i % 100 == 0:
                costs.append(cost_avg)
                learning_rates.append(learning_rate)

            # Decay learning_rate
            learning_rate = learning_rate / (1 + learning_decay_rate * i)
        
        # Update instance attributes
        self.final_cost = cost_avg

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