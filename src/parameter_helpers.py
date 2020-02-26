import numpy as np

def gd_update(parameters, gradients, learning_rate):
    """
    Update parameters using gradient descent
    
    Parameters
    ----------
    parameters : dict
    gradients : dict
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
        parameters["W" + str(l+1)] -= learning_rate * gradients["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * gradients["db" + str(l+1)]
    return parameters
    
def adam_update(parameters, gradients, v, s, adam_counter, learning_rate = 0.01,
                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Parameters
    ----------
    parameters : dict
        parameters['W' + str(l)] = Wl
        parameters['b' + str(l)] = bl
    gradients : dict
        gradients['dW' + str(l)] = dWl
        gradients['db' + str(l)] = dbl
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
        # Moving average of the gradients. Inputs: "v, gradients, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * gradients["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * gradients["db" + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, adam_counter". Output: "v_new".
        v_new["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - np.power(beta1, adam_counter))
        v_new["db" + str(l+1)] = v["db" + str(l+1)] / (1 - np.power(beta1, adam_counter))

        # Moving average of the squared gradients. Inputs: "s, gradients, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.power(gradients["dW" + str(l+1)], 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.power(gradients["db" + str(l+1)], 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, adam_counter". Output: "s_new".
        s_new["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - np.power(beta2, adam_counter))
        s_new["db" + str(l+1)] = s["db" + str(l+1)] / (1 - np.power(beta2, adam_counter))

        # Update parameters. Inputs: "parameters, learning_rate, v_new, s_new, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_new["dW" + str(l+1)] / (np.sqrt(s_new["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_new["db" + str(l+1)] / (np.sqrt(s_new["db" + str(l+1)]) + epsilon)

    return parameters, v, s