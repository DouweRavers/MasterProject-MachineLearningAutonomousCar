import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import iterable

import NeuralNetwork as nn



from numpy.lib import utils
from scipy import optimize
from scipy.linalg.decomp_svd import null_space

import utils

class EvaluateNN():
    
    def __init__(self, neural_network):
        self.neural_network = neural_network
    
    def splitDataSet(self, X, Y, print_process = False):
        if print_process: print("Splitting data in train, test and validation sets...")
        X_train, X_test, X_val = np.split(X, [int(0.6*len(X)),int(0.8*len(X))])
        Y_train, Y_test, Y_val = np.split(Y, [int(0.6*len(X)),int(0.8*len(X))])
        if print_process: print("Splitting complete!")
        return X_train, Y_train, X_test, Y_test, X_val, Y_val

    def learningCurve(self, X_train, Y_train, X_val, Y_val, lambda_ = 0, print_process=False):
        m = Y_train.size
        sample_sizes_array = np.arange(1, m+1, 1 if m < 1000 else int(m / 100))
        error_train = np.zeros(len(sample_sizes_array))
        error_val   = np.zeros(len(sample_sizes_array))
        if print_process: print("Generate data for sample count learning curve...")
        for i in range(len(sample_sizes_array)):
            x_train = X_train[:sample_sizes_array[i],:]
            y_train = Y_train[:sample_sizes_array[i]]
            nn_params = self.neural_network.learnByGradientDecent(x_train, y_train, lambda_, print_process=False)
            error_train[i-1], _ = self.neural_network.costfunction(nn_params, x_train, y_train)
            error_val[i-1], _ = self.neural_network.costfunction(nn_params, X_val, Y_val)
        if print_process: print("Finished data generation!")
        return error_train, error_val, sample_sizes_array

    def learningCurvePlot(self, X_train, Y_train, X_val, Y_val, lambda_ = 0, print_process=False):
        m = Y_train.size
        error_train, error_val, sample_sizes_array = self.learningCurve(X_train, Y_train, X_val, Y_val, lambda_, print_process)
        plt.plot(sample_sizes_array, error_train, sample_sizes_array, error_val, lw=2)
        plt.title('Learning curve for neural network')
        plt.legend(['Train', 'Cross Validation'])
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.axis([0, m, 0, 1])
        plt.show()


    # Depending on bias-variance -> adding more or less features
    # F.e. High bias (underfitting) -> adding more features
    def evaluatePol(self):
        pol_vec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        lambda_ = 100

        error_train = np.zeros(len(pol_vec))
        error_val = np.zeros(len(pol_vec))

        cur_error = 0
        pol_ideal = 1
        for i in range(len(pol_vec)):
            X_poly = self.polyFeatures(self.X, i)
            X_poly, mu, sigma = self.featureNormalize(X_poly)
            X_poly = np.concatenate([np.ones((self.Y_train.size, 1)), X_poly], axis=1)

            theta_t = utils.trainLinearReg2(self.costfunction, X_poly, self.Y_train, lambda_=lambda_, maxiter=55)
            error_val[i], _ = self.costfunction(theta_t, self.X_val, self.Y_val, lambda_ = 0)
            error_train[i], _ = self.costfunction(theta_t, self.X_train, self.Y_train, lambda_ = 0)

            if (error_val[i] >= cur_error):
                pol_ideal = pol_vec[i]
                cur_error = error_val[i]
    
        return pol_vec, error_val, _, pol_ideal
    
    def evaluatePolplot(self):
        pol_vec, error_val, error_train, pol_ideal = self.evaluatePol()

        pyplot.plot(pol_vec, error_train, '-o', pol_vec, error_val, '-o', lw=2)
        pyplot.legend(['Train', 'Cross Validation'])
        pyplot.xlabel('lambda')
        pyplot.ylabel('Error')
            

    # Depending on bias-variance -> higher or lower regularisation term
    # F.e. High bias -> lower regularisation term
    def validationCurve(self):
        # Selected values of lambda (you should not change this)
        lambda_vec = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10]

        # You need to return these variables correctly.
        error_train = np.zeros(len(lambda_vec))
        error_val = np.zeros(len(lambda_vec))


        cur_error = 0
        lambda_ideal = 1
        for i in range(len(lambda_vec)):
            lambda_try = lambda_vec[i]
            theta_t = utils.trainLinearReg2(self.costfunction, self.X_train, self.Y_train, lambda_ = lambda_try)
            error_train[i], _ = self.costfunction(theta_t, self.X_train, self.Y_train, lambda_ = 0)
            error_val[i], _ = self.costfunction(theta_t, self.X_val, self.Y_val, lambda_ = 0)
            #choose lambda with min(error_val)
            if (error_val[i] >= cur_error):
                lambda_ideal = lambda_vec[i]
                cur_error = error_val[i]

        return lambda_vec, error_train, error_val, lambda_ideal

    def validationCurvePlot(self):
        lambda_vec, error_train, error_val, _ = self.validationCurve(self.X, self.Y, self.X_val, self.Y_val)

        pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
        pyplot.legend(['Train', 'Cross Validation'])
        pyplot.xlabel('lambda')
        pyplot.ylabel('Error')

    #calculate test-error (evaluate model on a test set that was not used in any part of training)
    def testError(self, lambda_ideal):
        theta_t = utils.trainLinearReg2(self.costfunction, self.X_train, self.Y_train, lambda_ = lambda_ideal)
        error_test = self.costfunction(theta_t, self.X_test, self.Y_test, lambda_ = 0)

        return error_test


    # other tips for neural network architectures and over-/underfitting
    # small neural network: more prone to underfitting (comp cheaper)
    # large neural network: more prone to overfitting (comp more expensive)

    #after this: manual error analysis



    