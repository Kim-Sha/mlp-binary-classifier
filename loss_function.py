import numpy as np 

def compute_cost(AL, Y):
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
        total_cost = np.nansum(log_cost)
        assert(total_cost.shape == ())
        
        return total_cost

def regularized_cost(AL, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization.
    
    Parameters
    ----------
    AL : numpy array
        post-activation, output of forward propagation, 
        of shape (output size, number of examples)
    Y : numpy array
        "true" labels vector, of shape (output size, number of examples)
    parameters : dict
        python dictionary containing parameters of the model
    
    Returns
    -------
    total_regularized_cost : 
        value of the regularized loss function (formula (2))
    """
    cost_cross_entropy = compute_cost(AL, Y) # This gives you the cross-entropy part of the cost
    cost_L2 = (lambd / 2) * np.nansum(list(map(lambda x: np.nansum(np.square(x)),
                                                             parameters.values())))    
    total_regularized_cost = cost_cross_entropy + cost_L2
    
    return total_regularized_cost