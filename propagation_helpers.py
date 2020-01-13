import numpy as np

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

def linear_activation_backward(dA, cache, activation):
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

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Parameters
    ----------
    dA : numpy array
        post-activation gradient, of any shape
    cache : 
        'Z' where we store for computing backward propagation efficiently

    Returns
    -------
    dZ : 
        Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Parameters
    ----------
    dA : 
        post-activation gradient, of any shape
    cache : 
        'Z' where we store for computing backward propagation efficiently

    Returns
    -------
    dZ : 
        Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

