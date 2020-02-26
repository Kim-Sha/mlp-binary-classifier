import numpy as np

def back_prop(AL, Y, caches, lambd):
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
    gradients : dict
        A dictionary with the gradients
            gradients["dA" + str(l)] = ... 
            gradients["dW" + str(l)] = ...
            gradients["db" + str(l)] = ... 
    """
    gradients = {}
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "gradients["dAL-1"], gradients["dWL"], gradients["dbL"]
    current_cache = caches[L-1]
    gradients["dA" + str(L-1)], gradients["dW" + str(L)], gradients["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                    current_cache,
                                                                                                    "sigmoid",
                                                                                                    lambd)        
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "gradients["dA" + str(l + 1)], current_cache". Outputs: "gradients["dA" + str(l)] , gradients["dW" + str(l + 1)] , gradients["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(gradients["dA" + str(l + 1)],
                                                                    current_cache,
                                                                    "relu",
                                                                    lambd)
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return gradients

def linear_activation_backward(dA, cache, activation, lambd):
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

    dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * W
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
    dZ = np.array(dA, copy = True) # just converting dz to a correct object.
    
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

