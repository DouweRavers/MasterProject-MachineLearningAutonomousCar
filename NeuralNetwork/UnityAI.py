import Algorithms as al
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog

print_process = True

algorithm = al.Algorithms(nnlin.NeuralNetworkLinear(
    input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1))
X, Y = algorithm.loadData(print_process=print_process,
                          limit_on_load=False, limiter=900)
nn_params = algorithm.nnLearn(
    X, Y, store_to_file=True, print_process=print_process)
algorithm.nnRunAI(nn_params=nn_params, print_process=print_process)
