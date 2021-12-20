import Algorithms as al
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog

# Rapport sequence

# ================
# We started from the neural network of the exercise session
# A logistic regression algorithm worked just good enough for the car to drive around.
# Show movie
# We analysed the network and discovered it has a high bias at 60%.
# ================
neural_network_logistic = nnlog.NeuralNetworkLogistic(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
algorithm = al.Algorithms(neural_network_logistic)
X, Y = algorithm.loadData(print_process=True, limit_on_load=False, limiter=2000)
algorithm.ShowLearningCurve(X, Y, print_process=True)
# ================
# Based on the data we know a fundamental change to the algorithm had to be made.
# As we analysed the task that has to be performed better we realised the 
# most importand change is the fact that the AI should be able to steer like a human
# meaning instead of going from super left to super right it would perform better when
# outputting linear values also inbetween left, center and right, center
# Thus we redesigned the neural network to be a linear one
# The result was even more impactfull as we though
# We went from a high bias of 60% to a ?low bias or what is this? of 0.6% 
# ================
neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
algorithm = al.Algorithms(neural_network_linear)
X, Y = algorithm.loadData(print_process=True, limit_on_load=False, limiter=2000)
algorithm.ShowLearningCurve(X, Y, print_process=True)
