import Algorithms as al
import numpy as np
import math
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog
import matplotlib.pyplot as plt
neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
algorithm = al.Algorithms(neural_network_linear)
X, Y = algorithm.loadData(print_process=False)
algorithm.ShowDataSet(X, Y, False)