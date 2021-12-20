import numpy as np
import math
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures

import NeuralNetworkLogistic as nn

neural_network = nn.NeuralNetworkLogistic(21, 10, 1)

#other help functions
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm

def polyFeatures2(X, p):
    poly = PolynomialFeatures(p)
    X_poly = poly.fit_transform(X)
    
    return X_poly

def polyFeatures3(X, p): #X_train 28922, 21
    X_poly = np.zeros((X.shape[0], p)) #21,1

    for i in range(X.shape[1] * p):
        X_poly[:, i] = X[:, math.floor(i / p)] ** (i % p + 1)

    return X_poly 

def polyFeatures(X, p): 
    X_poly = np.zeros((X.shape[0], X.shape[1] * p)) 

    for i in range(X.shape[1] * p):
        X_poly[:, i] = X[:, math.floor(i/p)] ** (i%p + 1)

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
