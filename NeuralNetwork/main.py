import Algorithms as al
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog

# TEST MORE OR LESS HIDDEN NODES FOR 1 LAYER


neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
algorithm = al.Algorithms(neural_network_linear)
X, Y = algorithm.loadData(print_process=True, limit_on_load=False, limiter=2000)
algorithm.ShowLearningCurve(X, Y, print_process=True)

