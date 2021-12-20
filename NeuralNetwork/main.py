import Algorithms as al
import numpy as np
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog
import matplotlib.pyplot as plt

# TEST MORE OR LESS HIDDEN NODES FOR 1 LAYER

neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
algorithm = al.Algorithms(neural_network_linear)
X, Y = algorithm.loadData(print_process=True, limit_on_load=False)

# algorithm.ShowDataSetStats(X, Y, True)
nn_params = neural_network_linear.learnByGradientDecent(X, Y, 0)
X = X[:100,:]
Y = Y[:100]
P = neural_network_linear.predict(nn_params, X)
O = np.sum(pow(P - Y, 2)) /200
print(O)
m = len(Y)
Sigma = 1/m * np.matmul(X.transpose(), X)
U, S, V = np.linalg.svd(Sigma)
Ureduce = U[:,:2]
Z = np.matmul(X, Ureduce)
# plt.plot(Z[:,0], Z[:,1], "bo") #, Z, P, "r.")

fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# plotting
ax.scatter(Z[:,0], Z[:,1], Y, 'green')
ax.scatter(Z[:,0], Z[:,1], P, 'yellow')
ax.set_title('Plot')
plt.show()


# plt.title('Reduced plot')
# plt.xlabel('Reduced X')
# plt.ylabel('Y')
# plt.show()
