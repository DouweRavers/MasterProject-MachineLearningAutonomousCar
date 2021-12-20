import Algorithms as al
import numpy as np
import math
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog
import matplotlib.pyplot as plt

# TEST MORE OR LESS HIDDEN NODES FOR 1 LAYER

# neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
# algorithm = al.Algorithms(neural_network_linear)
# X, Y = algorithm.loadData(print_process=True, limit_on_load=False)
# X = X[Y!=0.5,:]
# Y = Y[Y!=0.5]
# # algorithm.ShowDataSetStats(X, Y, True)
# nn_params = neural_network_linear.learnByGradientDecent(X, Y, 0)

# X = X[:100,:]
# Y = Y[:100]
# P = neural_network_linear.predict(nn_params, X)
# print(np.average(abs(P-Y)))
# m = len(Y)
# Sigma = 1/m * np.matmul(X.transpose(), X)
# U, S, V = np.linalg.svd(Sigma)
# Ureduce = U[:,:2]
# Z = np.matmul(X, Ureduce)
# # plt.plot(Z[:,0], Z[:,1], "bo") #, Z, P, "r.")

# fig = plt.figure()
 
# # syntax for 3-D projection
# ax = plt.axes(projection ='3d')
 
# # plotting
# ax.scatter(Z[:,0], Z[:,1], Y, 'b')
# ax.scatter(Z[:,0], Z[:,1], P, 'r')
# ax.set_title('Plot')
# ax.set_xlabel("Z_0")
# ax.set_ylabel("Z_1")
# ax.set_zlabel("Y")
# plt.show()


# # plt.title('Reduced plot')
# # plt.xlabel('Reduced X')
# # plt.ylabel('Y')
# # plt.show()


# print_process = True
# neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=5, num_labels=1)
# algorithm = al.Algorithms(neural_network_linear)
# X, Y = algorithm.loadData(print_process=print_process)
# algorithm.ShowDataSetStats(X, Y, print_process)


def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], X.shape[1] * p))

    for i in range(X.shape[1] * p):
        X_poly[:, i] = X[:, math.floor(i / p)] ** (i % p + 1)

    return X_poly

X = np.array([
    1,0.637527,1,0.4358858,0.8894251,0.3299929,0.4549366,0.2670704,0.319089,0.2271014,0.2553387,0.2010406,0.2193554,0.1833904,0.1974745,0.1724566,0.184175,0.1674189,0.1762452,0.1669539,0.1724981
]).reshape(1, 21)
p = 3
pX = polyFeatures(X, 50)
print(pX)