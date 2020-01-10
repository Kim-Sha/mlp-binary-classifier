import numpy as np
import matplotlib.pyplot as plt
from activation_functions import sigmoid, sigmoid_backward, relu, relu_backward
from sklearn import preprocessing

np.random.seed(1)

class BinaryNN:
    """Brief class description
    
    Some more extensive description
    
    Attributes
    ----------
    attr1 : string
        Purpose of attr1.
    attr2 : float
        Purpose of attr2.
    
    """
    
    def __init__(self, X, Y):
        """Example of docstring on the __init__ method.
        
        Parameters
        ----------
        param1 : str
            Description of `param1`.
        param2 : float
            Description of `param2`.
        param3 : int, optional
            Description of `param3`, defaults to 0.
        
        """
        self.X = X
        self.Y = Y
        self.parameters = {}
    
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
        np.random.seed(1)
        parameters = {}
        L = len(layer_dimensions)

        for i in range(1, L):
            parameters['W' + str(i)] = np.random.randn(layer_dimensions[i], layer_dimensions[i-1]) / np.sqrt(layer_dimensions[i-1])
            parameters['b' + str(i)] = np.zeros((layer_dimensions[i], 1))
        
            assert(parameters['W' + str(i)].shape == (layer_dimensions[i], 
                                                      layer_dimensions[i - 1]))
            assert(parameters['b' + str(i)].shape == (layer_dimensions[i], 1))
        
        return parameters
    
    """
    FORWARD PROP
    """

    def __linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Parameters
        ----------
        A : numpy array
            Activations from previous layer (or input data):
            (previous layer size, number of examples)
        W : numpy array 
            Weights matrix of shape (current layer size, previous layer size)
        b : numpy array
            Bias vector of shape (current layer size, 1)

        Returns
        -------
        Z : numpy array
            Input of activation function 
        cache : tuple
            "A", "W" and "b" stored for computing the backward pass efficiently
        """
        Z = np.dot(W, A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache

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
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
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
        
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = -(1/m)*np.sum((np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y)))
        
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        return cost
    
    """
    BACKPROP 
    """

    def __linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Parameters
        ----------
        dZ : numpy array
            Gradient of the cost wrt the linear output (of current layer l)
        cache : tuple
            (A_prev, W, b) coming from the forward propagation in the current layer

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
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db

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
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db
    
    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Parameters
        ----------
        AL : numpy array
            Output of the forward propagation (L_model_forward())
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
        m = AL.shape[1]
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

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Parameters
        ----------
        parameters : dict
        grads : dict
            Gradients output by L_model_backward()
        
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
    
    def L_layer_model(self, layer_dimensions, learning_rate = 0.0075,
                      num_iterations = 2500, print_cost = False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Parameters
        ----------
        X : numpy array
            data of shape (num_px * num_px * 3, number of examples)
        Y : numpy array
            true "label" vector of shape (1, number of examples)
        layer_dimensions : list
            input size and each layer size, of length (number of layers + 1).
        learning_rate : float
            learning rate of the gradient descent update rule
        num_iterations : int
            number of iterations of the optimization loop
        print_cost : boolean
            if True, it prints the cost every 100 steps
        
        Returns
        -------
        parameters : numpy array
            parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = [] # keep track of cost
        
        # Parameters initialization. (≈ 1 line of code)
        parameters = self.initialize_parameters(layer_dimensions)
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward(self.X, parameters)
            
            # Compute cost.
            cost = self.compute_cost(AL, self.Y)
        
            # Backward propagation.
            grads = self.L_model_backward(AL, self.Y, caches)
    
            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if i % 100 == 0:
                costs.append(cost)

        self.parameters = parameters
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
            
    def predict(self, X, y):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward(X, self.parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p

'''
--------------------------------------------------------------------------------
'''

# cat_X_train = np.loadtxt("cat_train_x.csv")
# cat_y_train = np.loadtxt("cat_train_y.csv")
# cat_X_test = np.loadtxt("cat_test_x.csv")
# cat_y_test = np.loadtxt("cat_test_y.csv")

# cat_X_train = cat_X_train
# cat_y_train = cat_y_train.reshape(1, cat_y_train.shape[0])
# cat_X_test = cat_X_test
# cat_y_test = cat_y_test.reshape(1, cat_y_test.shape[0])

# cat_layers_dims = [12288, 20, 7, 5, 1]

# cat_nn = BinaryNN(X = cat_X_train, Y = cat_y_train)
# cat_parameters = cat_nn.L_layer_model(layer_dimensions = cat_layers_dims, print_cost = True)

# Z1 = np.dot(cat_parameters["W1"], cat_X_train) + cat_parameters["b1"]
# A1 = relu(Z1)[0]
# Z2 = np.dot(cat_parameters["W2"], A1) + cat_parameters["b2"]
# A2 = relu(Z2)[0]
# Z3 = np.dot(cat_parameters["W3"], A2) + cat_parameters["b3"]
# A3 = relu(Z3)[0]
# Z4 = np.dot(cat_parameters["W4"], A3) + cat_parameters["b4"]
# A4 = sigmoid(Z4)[0]

# pred_train = np.where(A4 > 0.5, 1, 0)
# print(accuracy_score(cat_y_train[0], pred_train[0]))


'''
--------------------------------------------------------------------------------
'''

# X_train = np.loadtxt("x_train.csv")
# y_train = np.loadtxt("y_train.csv")
# X_test = np.loadtxt("x_test.csv")
# y_test = np.loadtxt("y_test.csv")

# X_train = X_train.T
# y_train = y_train.reshape(1, y_train.shape[0])
# X_test = X_test.T
# y_test = y_test.reshape(1, y_test.shape[0])

# scaler = preprocessing.StandardScaler()
# scaler.fit(X_train)
# norm_X_train = scaler.transform(X_train)

# churn_nn = BinaryNN(norm_X_train, y_train)
# churn_nn.L_layer_model(layer_dimensions = [39, 20, 10, 5, 1], print_cost = True)
# pred_train = churn_nn.predict(norm_X_train, y_train)