import numpy as np
import matplotlib.pyplot as plt
import time as t
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog

# for old code
from numpy.lib import utils
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
    
    # Evaluate the error in respect of the number of data points
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
            error_train[i], _ = self.neural_network.costfunction(nn_params, x_train, y_train)
            error_val[i], _ = self.neural_network.costfunction(nn_params, X_val, Y_val)
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
        plt.axis([0, m, 0, error_train.max() if error_train.max() > error_val.max() else error_val.max()])
        plt.show()
        
    # Checking the error in respect of the number of hidden layers
    def errorHiddenLayersCurve(self, X_train, Y_train, X_val, Y_val, sizes=np.arange(5, 15), Linear = True, print_process=False):
        error_val = np.zeros(sizes.shape)
        error_train = np.zeros(sizes.shape)
        if print_process: print("Generate data for hidden layer size curve...")
        for i in range(len(sizes)):
            layer_size = sizes[i]
            if Linear:
                neural_network = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=layer_size, num_labels=1)
            else:
                neural_network = nnlog.NeuralNetworkLogistic(input_layer_size=21, hidden_layer_size_alpha=layer_size, num_labels=1)
            
            nn_params = neural_network.learnByGradientDecent(X_train, Y_train, 0, print_process=False)
            error_train[i], _ = neural_network.costfunction(nn_params, X_train, Y_train)
            error_val[i], _ = neural_network.costfunction(nn_params, X_val, Y_val)
        if print_process: print("Finished data generation!")
        return error_train, error_val
        
    def errorHiddenLayersPlot(self, X_train, Y_train, X_val, Y_val, print_process=False):
        sizes=np.arange(1, 50)
        error_train, error_val = self.errorHiddenLayersCurve(X_train, Y_train, X_val, Y_val, sizes, print_process)
        plt.plot(sizes, error_train, sizes, error_val, lw=2)
        plt.title('Hidden layer error for neural network')
        plt.legend(['Train', 'Cross Validation'])
        plt.xlabel('Number of elements in hidden layer')
        plt.ylabel('Error')
        plt.axis([sizes[0], sizes[-1], 0, error_train.max() if error_train.max() > error_val.max() else error_val.max()])
        plt.show()
    
    # Checking the error in respect of the number of hidden layers
    def performanceHiddenLayersCurve(self, X_train, Y_train, sizes=np.arange(5, 15), Linear = True, print_process=False):
        performance = np.zeros(sizes.shape)
        if print_process: print("Measure speed of network training...")
        for i in range(len(sizes)):
            layer_size = sizes[i]
            if Linear:
                neural_network = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=layer_size, num_labels=1)
            else:
                neural_network = nnlog.NeuralNetworkLogistic(input_layer_size=21, hidden_layer_size_alpha=layer_size, num_labels=1)
            before = t.time()
            neural_network.learnByGradientDecent(X_train, Y_train, 0, print_process=False)
            performance[i] = t.time() - before
        if print_process: print("Finished measuring!")
        return performance
        
    def performanceHiddenLayersPlot(self, X_train, Y_train, print_process=False):
        sizes=np.arange(1, 50)
        performance = self.performanceHiddenLayersCurve(X_train, Y_train, sizes, print_process)
        plt.plot(sizes, performance)
        plt.title('Hidden layer performance for neural network')
        plt.xlabel('Number of elements in hidden layer')
        plt.ylabel('Performance')
        plt.axis([sizes[0], sizes[-1], 0, performance.max()])
        plt.show()
    
    def datasetAnalysis(self, X, Y, print_process=False):
        non_half_percentage = len(Y[Y != 0.5]) / len(Y) * 100
        half_percentage = len(Y[Y == 0.5]) / len(Y) * 100
        if print_process: print("=========== Actual message ===========\n")
        print("percentage half values: ", round(half_percentage, 2),"% and thus ", round(non_half_percentage, 2), "% non-half values, half means idle steering, no left no right.")
        if print_process: print("\n=========== End of message ===========")
    
    def datasetAndPredictionVisualtization(self, X, Y, print_process=False):
        if print_process: print("Learn parameters for full dataset...")
        nn_params = self.neural_network.learnByGradientDecent(X, Y, 0)
        if print_process: print("Learning succesfull!")
        X_reduced = X[:1000,:]
        Y_reduced = Y[:1000]
        if print_process: print("Predict output of subset of ", len(Y), " random values")
        P = self.neural_network.predict(nn_params, X_reduced)
        if print_process: print("With error of ", round(self.neural_network.costfunction(nn_params, X, Y)[0], 2))
        if print_process: print("Reduce dimensions of X to 2 for plotting...")
        m = len(Y)
        Sigma = 1/m * np.matmul(X.transpose(), X)
        U, S, V = np.linalg.svd(Sigma)
        Ureduce = U[:,:2]
        Z = np.matmul(X_reduced, Ureduce)
        if print_process: print("Compression succesfull!")
        self._plotReduced(Z, Y_reduced, P, 'Dataset vs predictions on compressed features')
        
        
    def _plotReduced(self, Z, Y, P, title):
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.scatter(Z[:,0], Z[:,1], Y, 'b')
        ax.scatter(Z[:,0], Z[:,1], P, 'r')
        ax.set_title('Plot')
        ax.set_xlabel("Z_0")
        ax.set_ylabel("Z_1")
        ax.set_zlabel("Y")
        ax.set_title(title)
        ax.legend(['Dataset', 'Prediction'])
        plt.show() 

    
    def errorAmountFeatures(self, X_train, Y_train, X_val, Y_val, sizes=np.arange(0, 30), Linear = True, print_process=False):
        pass


   
    def errorPolynomialCurve(self, X_train, Y_train, X_val, Y_val, Linear = True, print_process=False):
        pol_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        error_val = np.zeros(len(pol_vec))
        error_train = np.zeros(len(pol_vec))
        if print_process: print("Generate data for polynomial curve...")
        for p in range(len(pol_vec)):
            cur_p = pol_vec[p]
            X_poly = utils.polyFeatures(X_train, cur_p)
            #X_poly = utils.featureNormalize(X_poly)
            X_poly_val = utils.polyFeatures(X_val, cur_p)
            #X_poly_val = utils.featureNormalize(X_poly_val)
            #X_poly = np.concatenate([np.ones((Y_train.size, 1)), X_poly], axis=1)

            if Linear:
                neural_network = nnlin.NeuralNetworkLinear(input_layer_size=X_poly.shape[1], hidden_layer_size_alpha=10, num_labels=1)
            else:
                neural_network = nnlog.NeuralNetworkLogistic(input_layer_size=X_poly.shape[1], hidden_layer_size_alpha=10, num_labels=1)
            
            nn_params = neural_network.learnByGradientDecent(X_poly, Y_train, 0, print_process=False)
            error_train[p], _ = neural_network.costfunction(nn_params, X_poly, Y_train)
            error_val[p], _ = neural_network.costfunction(nn_params, X_poly_val, Y_val)
            print(p, "success")
        if print_process: print("Finished data generation!")
        return pol_vec, error_train, error_val

    def polynomialCurvePlot(self, X_train, Y_train, X_val, Y_val, print_process=False):
        pol_vec, error_val, error_train = self.errorPolynomialCurve(X_train, Y_train, X_val, Y_val, print_process)

        plt.plot(pol_vec, error_train, '-o', pol_vec, error_val, '-o', lw=2)
        plt.legend(['Train', 'Cross Validation'])
        plt.xlabel('lambda')
        plt.ylabel('Error')
        plt.show()

    

    def validationCurve(self, X_train, Y_train, X_val, Y_val, Linear = True, print_process=False):
        lambda_vec = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10]

        error_train = np.zeros(len(lambda_vec))
        error_val = np.zeros(len(lambda_vec))

        for i in range(len(lambda_vec)):
            lambda_try = lambda_vec[i]
            if Linear:
                neural_network = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
            else:
                neural_network = nnlog.NeuralNetworkLogistic(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)

            nn_params = neural_network.learnByGradientDecent(X_train, Y_train, lambda_try, print_process=False)
            error_train[i], _ = neural_network.costfunction(nn_params, X_train, Y_train)
            error_val[i], _ = neural_network.costfunction(nn_params, X_val, Y_val)

        return lambda_vec, error_train, error_val

    def validationCurvePlot(self, X_train, Y_train, X_val, Y_val, print_process=False):
        lambda_vec, error_train, error_val = self.validationCurve(X_train, Y_train, X_val, Y_val, print_process)

        plt.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
        plt.legend(['Train', 'Cross Validation'])
        plt.xlabel('lambda')
        plt.ylabel('Error')
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# ============================
# OLD CODE STARTS HERE
# ============================


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
            

    def validationCurvePlot2(self):
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



    