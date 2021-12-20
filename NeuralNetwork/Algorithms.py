# liberaries
import numpy as np
import time

# objects
import FileIO as io
import UnityConnector as uc
import EvaluateNN as ev

class Algorithms:
    def __init__(self, neural_network):
        self.neural_network = neural_network 

    def loadData(self, print_process = False, limiter = -1, limit_on_load = True):
        if print_process: print("Reading data from file...")
        X , Y = io.loadDataFile(limiter if limit_on_load else -1)
        if not limit_on_load:
            X = X[:limiter,:]
            Y = Y[:limiter]
        if print_process: print("Data read succesfully!")
        return X, Y

    def nnLearn(self, X, Y, lambda_ = 0, store_to_file=False, print_process = False):
        nn_params = self.neural_network.learnByGradientDecent(X, Y, lambda_, print_process)
        if store_to_file:
            if print_process: print("Store parameters to file...")
            nn_params.tofile("data/theta.csv", sep=",")
            if print_process: print("Parameters stored to data/theta.csv!")
        return nn_params
    
    def nnRunAI(self, nn_params = [], print_process = False):
        if nn_params == []: 
            # get theta values from file
            if print_process: print("Load parameters from file...")
            nn_params = np.genfromtxt("data/theta.csv", delimiter=",", dtype=float)
            if print_process: print("Parameters loaded succesfully!")
        # connect to unity game
        if print_process: print("Connecting to unity game...")
        unity_connection = uc.UnityConnector()
        if print_process: print("Connected succesfully!")
        
        # get game data and predict steering 
        if print_process: print("Start player AI...")
        while True:
            sensor_values = unity_connection.request_sensor_values()
            steering_value = self.neural_network.predict(nn_params, sensor_values)
            steering_value = 2 * steering_value[0,0] - 1
            unity_connection.send_steering_value(steering_value)
            time.sleep(0.5)

    def ShowLearningCurve(self, X, Y, lambda_ = 0, print_process = False):
        evaluator = ev.EvaluateNN(self.neural_network)
        X_train, Y_train, X_test, Y_test, X_val, Y_val = evaluator.splitDataSet(X, Y, print_process)
        evaluator.learningCurvePlot(X_train, Y_train, X_val, Y_val, lambda_, print_process)