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
    
    def fit_binary(self, layer_dimensions, learning_rate = 0.001,
                   learning_decay_rate = 1e-7, lambd = 0.1, 
                   minibatched = True, minibatch_size = 64, optimizer = "adam", 
                   beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, 
                   num_epochs = 3000, print_cost = True):
        """
        L-layer neural network model including support for L2 regularization,
        minibatch gradient descent, and ADAM optimization.
        
        Parameters
        ----------
        layer_dimensions : list
            Dimensions of the NN, including input, hidden, and output layers.
            N.B. input layer must have dimension of the number of input features
            in X. Output layer must have dimension of 1.
        learning_rate : float
            Defaults to 0.001.
        learning_decay_rate : float
            Rate at which to decay the learning_rate. Defaults to 1e-7.
        lambd : float
            L2 regularization hyperparameter. Defaults to 0.1.
        minibatched : boolean
            Whether to implement mini-batches. Defaults to True.
        minibatch_size : int
            Size of minibatch in powers of 2 (64, 128, 256...). Defaults to 64.
        optimizer : string {"gd", "adam"}
            Optimization algorithm to use. Defaults to ADAM optimizer.
        beta : float
            Momentum hyperparameter. Defaults to 0.9
        beta1 : float
            Exponential decay hyperparameter. Defaults to 0.9.
        beta2 : float
            Exponential decay hyperparameter. Defaults to 0.999. 
        epsilon : float
            Small value to prevent division by 0 in ADAM. Defaults to 1e-8.
        num_epochs : int
            Number of epoch iterations. Defaults to 3000.
        print_cost : boolean
            Whether to print the cost every 1000 epochs. Defaults to True.

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

        # Make assertions
        assert (learning_rate >= 0)
        assert (learning_decay_rate >= 0)
        assert (lambd >= 0)

        # Set trackers and counters
        costs = []
        learning_rates = []
        adam_counter = 0 
        seed = 0
        sample_size = self.X.shape[1]
        
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

            # Minibatch Gradient Descent
            if minibatched:

                # Define random minibatches and increment seed to reshuffle 
                # dataset differently after each epoch
                seed = seed + 1
                minibatches = initialize_minibatches(self.X, self.Y, minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                    AL, caches = forward_prop(minibatch_X, self.parameters)

                    if lambd > 0:
                        # Compute L2 regularized loss function
                        cost_total += regularized_cost(AL, minibatch_Y, self.parameters, lambd)
                    else:
                        # Compute vanilla loss function
                        cost_total += compute_cost(AL, minibatch_Y)

                    # Backward propagation
                    gradients = back_prop(AL, minibatch_Y, caches, lambd)

                    # Update parameters
                    if optimizer == "gd":
                        self.parameters = gd_update(self.parameters,
                                                    gradients,
                                                    learning_rate)
                    elif optimizer == "adam":
                        adam_counter += 1 # Adam counter
                        self.parameters, v, s = adam_update(self.parameters,
                                                            gradients, v, s, 
                                                            adam_counter, 
                                                            learning_rate,
                                                            beta1, beta2,
                                                            epsilon)
            
            # Batch Gradient Descent
            else:

                AL, caches = forward_prop(self.X, self.parameters)

                if lambd > 0:
                    cost_total += regularized_cost(AL, self.Y, self.parameters, lambd)
                else:
                    cost_total += compute_cost(AL, self.Y)

                gradients = back_prop(AL, self.Y, caches, lambd)

                if optimizer == "gd":
                    self.parameters = gd_update(self.parameters,
                                                gradients,
                                                learning_rate)
                elif optimizer == "adam":
                    adam_counter += 1
                    self.parameters, v, s = adam_update(self.parameters,
                                                        gradients, v, s, 
                                                        adam_counter, 
                                                        learning_rate,
                                                        beta1, beta2,
                                                        epsilon)

            # Print the cost every 1000 epoch
            cost_avg = cost_total / sample_size

            if print_cost and i % 1000 == 0: 
                print("Cost after epoch %i: %f" %(i, cost_avg))
                print("Learning rate after epoch %i: %f" %(i, learning_rate))
            if i % 100 == 0:
                costs.append(cost_avg)
                learning_rates.append(learning_rate)

            # Decay learning_rate
            learning_rate = learning_rate / (1 + learning_decay_rate * i)
        
        # Update instance attributes after final iteration
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
        Predicts output labels of dataset given parameters trained in the 
        L-layer NN. Evaluates against true labels to determine accuracy.
        
        Parameters
        ----------
        X : numpy array
            Input dataset of examples you to predict labels for.
        y : numpy array
            True labels of examples to compare predictions against.
        
        Returns
        -------
        p : numpy array
            predictions for given dataset X.
        """
        
        sample_size = X.shape[1]
        p = np.zeros((1, sample_size))
        
        # Forward propagation
        probabilities = forward_prop(X, self.parameters)[0]

        # convert probabilities to 0/1 predictions
        for i in range(0, probabilities.shape[1]):
            if probabilities[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        print("Accuracy: "  + str(np.sum((p == y) / sample_size)))
            
        return p