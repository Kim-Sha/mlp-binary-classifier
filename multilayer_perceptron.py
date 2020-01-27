import math
import numpy as np
import matplotlib.pyplot as plt
from initialization_helpers import initialize_parameters, initialize_adam_parameters, initialize_minibatches
from fwdprop_helpers import forward_prop
from loss_function_helpers import compute_cost, regularized_cost
from backprop_helpers import back_prop
from parameter_helpers import gd_update, adam_update

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
        self.parameters = {}
        self.final_cost = 0

    """
    MODEL
    """
    
    def fit_binary(self, layer_dimensions, lambd = 0.1, optimizer = "adam",
                   learning_rate = 0.025, learning_decay_rate = 1e-7, minibatched = True, 
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
        adam_counter = 0                     # initializing counter required for Adam update
        seed = 0
        m = self.X.shape[1]       # number of training examples
        
        # Initialize parameters
        self.parameters = initialize_parameters(layer_dimensions)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no optimizer used for vanilla gradient descent
        elif optimizer == "adam":
            v, s = initialize_adam_parameters(self.parameters)
        else:
            raise ValueError("The only supported optimizer modes are 'gd' and 'adam'.")
        
        # Optimization loop
        for i in range(num_epochs):
            
            cost_total = 0

            if minibatched:

                # Define the random minibatches. We increment the seed to 
                # reshuffle differently the dataset after each epoch
                seed = seed + 1
                minibatches = initialize_minibatches(self.X, self.Y, minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                    AL, caches = forward_prop(minibatch_X, self.parameters)

                    # Compute cost and add to the cost total
                    cost_total += regularized_cost(AL, minibatch_Y, self.parameters, lambd)

                    # Backward propagation
                    gradients = back_prop(AL, minibatch_Y, caches, lambd)

                    # Update parameters
                    if optimizer == "gd":
                        self.parameters = gd_update(self.parameters, gradients,
                                                         learning_rate)
                    elif optimizer == "adam":
                        adam_counter += 1 # Adam counter
                        self.parameters, v, s = adam_update(self.parameters,
                            gradients, v, s, adam_counter, learning_rate,
                            beta1, beta2, epsilon)
            else:
                AL, caches = forward_prop(self.X, self.parameters)
                cost_total += regularized_cost(AL, self.Y, self.parameters, lambd)
                gradients = back_prop(AL, self.Y, caches, lambd)
                self.parameters = gd_update(self.parameters, gradients, learning_rate)

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
        probabilities = forward_prop(X, self.parameters)[0]

        # convert probabilities to 0/1 predictions
        for i in range(0, probabilities.shape[1]):
            if probabilities[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p