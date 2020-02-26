import numpy as np

def forward_prop(X, parameters):
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

def linear_activation_forward(A_prev, W, b, activation):
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

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Parameters
    ----------
    Z : numpy array
    
    Returns
    -------
    A : numpy array
    cache : numpy array
    """
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Parameters
    ----------
    Z : numpy array
        Output of the linear layer, of any shape

    Returns
    -------
    A : numpy array
        Post-activation parameter, of the same shape as Z
    cache : dict
        a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache