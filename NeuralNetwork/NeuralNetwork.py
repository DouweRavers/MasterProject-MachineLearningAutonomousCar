import numpy as np
from scipy import optimize


# Inputlayer has 21 distance values between 0 - 1
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoidGradient(z):
    g = sigmoid(z) * (1-sigmoid(z))
    return g


class NeuralNetwork:

    input_layer_size = 0
    hidden_layer_size = 0
    num_labels = 0

    def __init__(self, input_layer_size, hidden_layer_size, num_labels):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.num_labels = num_labels

    def predict(self, nn_params, X):
        theta1 = np.reshape(nn_params[:self.hidden_layer_size * (self.input_layer_size + 1)],
                            (self.hidden_layer_size, (self.input_layer_size + 1)))

        theta2 = np.reshape(nn_params[(self.hidden_layer_size * (self.input_layer_size + 1)):],
                            (self.num_labels, (self.hidden_layer_size + 1)))

        m = X.shape[0]
        # print(X.shape)
        a_1 = np.concatenate([np.ones((m, 1)), X], axis=1)
        # print(a_1.shape)
        z_2 = np.matmul(a_1, theta1.transpose())
        a_2 = np.concatenate([np.ones((m, 1)), sigmoid(z_2)], axis=1)
        # print(a_2.shape)
        z_3 = np.matmul(a_2, theta2.transpose())
        a_3 = sigmoid(z_3)
        # print(a_3.shape)
        return a_3

    def costfunction(self, nn_params,
                     X, y, lambda_=0.0):
        m = y.size
        y = y.reshape(m, 1)

        theta1 = np.reshape(nn_params[:self.hidden_layer_size * (self.input_layer_size + 1)],
                            (self.hidden_layer_size, (self.input_layer_size + 1)))

        theta2 = np.reshape(nn_params[(self.hidden_layer_size * (self.input_layer_size + 1)):],
                            (self.num_labels, (self.hidden_layer_size + 1)))

        # print("theta1", self.theta1.shape, ", theta2", self.theta2.shape)
        # print("X", X.shape, ", Y", y.shape)

        a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
        z2 = np.matmul(a1, theta1.transpose())
        a2 = np.concatenate([np.ones((m, 1)), sigmoid(z2)],  axis=1)
        z3 = np.matmul(a2, theta2.transpose())
        a3 = sigmoid(z3)

        # print("a1 ", a1.shape, "a2 ", a2.shape, "a3 ", a3.shape)
        J = np.sum(y*np.log(a3)+(1-y)*np.log(1-a3))/(-m) + (lambda_/(2*m)) * \
            (np.sum(pow(theta1[:, 1:], 2)) + np.sum(pow(theta2[:, 1:], 2)))

        delta3 = a3 - y
        delta2 = np.matmul(delta3, theta2[:, 1:]) * sigmoidGradient(z2)
        # print("delta3 ", delta3.shape, "delta2 ", delta2.shape)
        Delta1 = np.matmul(delta2.transpose(), a1)
        Delta2 = np.matmul(delta3.transpose(), a2)
        # print("Delta1 ", Delta1.shape, "Delta2 ", Delta2.shape)
        # print("=============================")
        theta1_grad = 1/m * Delta1
        theta2_grad = 1/m * Delta2

        theta1_grad[:, 1:] += (lambda_/m) * theta1[:, 1:]
        theta2_grad[:, 1:] += (lambda_/m) * theta2[:, 1:]

        grad = np.concatenate([theta1_grad.ravel(), theta2_grad.ravel()])

        return J, grad

    def randInitializeWeights(self, epsilon_init=0.12):
        theta1 = np.random.rand(
            self.hidden_layer_size, 1 + self.input_layer_size) * 2 * epsilon_init - epsilon_init
        theta2 = np.random.rand(
            self.num_labels, 1 + self.hidden_layer_size) * 2 * epsilon_init - epsilon_init
        nn_params = np.concatenate([theta1.ravel(), theta2.ravel()], axis=0)
        return nn_params

    def learnByGradientDecent(self, X, Y, lambda_, iterations = 100, print_process = False):
        # random initializaiton of parameters
        initial_nn_params = self.randInitializeWeights()
        costFunction = lambda nn_params: self.costfunction(nn_params, X, Y, lambda_)
        
        #optimize parameters
        if print_process: print("Start gradient decent with initial randomized parameters...")
        res = optimize.minimize(costFunction,
                                initial_nn_params,
                                jac=True,
                                method='TNC',
                                options={'maxiter': iterations})
        if print_process: print("Gradient decent finished succesfully!")
        return res.x