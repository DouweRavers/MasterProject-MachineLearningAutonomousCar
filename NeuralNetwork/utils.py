import numpy as np
from scipy import optimize

import NeuralNetwork as nn

neural_network = nn.NeuralNetwork(21, 10, 1)

#other help functions
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm

def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))

    for i in range(p):
        X_poly[:, i] = X[:, 0] ** (i + 1)

    return X_poly

#find theta parameter
def trainLinearReg(linearRegCostFunction, X, y, lambda_ = 0.0, maxiter=200):
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    return res.x

def trainLinearReg2(cf, X, y, lambda_ = 0.0, maxiter=100):
    #random initializaiton of parameters
    initial_nn_params = neural_network.randInitializeWeights()
    
    options = {'maxiter': maxiter}
    #lambda_ = 1

    #costFunction = lambda t: cf(X, y, t, lambda_)
    def costFunction(initial_nn_params): return neural_network.costfunction(
    initial_nn_params, X, y, lambda_)

    #optimize parameters
    res = optimize.minimize(costFunction,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)
    return res.x
