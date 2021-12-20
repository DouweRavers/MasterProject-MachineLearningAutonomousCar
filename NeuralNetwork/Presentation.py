import Algorithms as al
import NeuralNetworkLinear as nnlin
import NeuralNetworkLogistic as nnlog

print_process = True

# Rapport sequence
# ================
# We started from the neural network of the exercise session
# A logistic regression algorithm worked just good enough for the car to drive around.
# Show movie
# We analysed the network and discovered it has a high bias at 60%.
# ================
neural_network_logistic = nnlog.NeuralNetworkLogistic(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
algorithm = al.Algorithms(neural_network_logistic)
X, Y = algorithm.loadData(print_process=print_process, limit_on_load=False, limiter=2000)
# algorithm.ShowLearningCurve(X, Y, print_process)
# ================
# Based on the data we know a fundamental change to the algorithm had to be made.
# As we analysed the task that has to be performed better we realised the 
# most importand change is the fact that the AI should be able to steer like a human
# meaning instead of going from super left to super right it would perform better when
# outputting linear values also inbetween left, center and right, center
# Thus we redesigned the neural network to be a linear one
# The result was even more impactfull as we though
# We went from a high bias of 60% to a ?low bias or what is this? of 0.6%. Still High bias I think
# Also we see that the curve converges at 400 meaning that 500/0.6 = 830 ~= 900 is enough data to get good results
# ================
neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=10, num_labels=1)
algorithm = al.Algorithms(neural_network_linear)
# algorithm.ShowLearningCurve(X, Y, print_process=print_process)
# ================
# Once we know that the linear network was better suited for the task we checked the next variable of the network.
# The number of ?elements? in the hidden layer
# First we checked the impact on the error in respect to the hidden layer size.
# This appeared to be constant
# Meaning we can just chose the layer with the best performance for this.
# This we again did by plotting the performance in respect to the hidden layer size
# 5 appeared a good size  
# ================
X, Y = algorithm.loadData(print_process=print_process, limit_on_load=False, limiter=900)
# algorithm.ShowHiddenLayerStats(X, Y, print_process)
neural_network_linear = nnlin.NeuralNetworkLinear(input_layer_size=21, hidden_layer_size_alpha=5, num_labels=1)
algorithm = al.Algorithms(neural_network_linear)


	

# ================
# Now we know the error is very low but the question still arises. Is this because the of skewed data or not.
# One argument is already dat in linear regression there are no false positives or false negatives just errors.
# This means the error actually always represents a good estimate even if only few cases of interest occur.
# When using the logistic neural network we saw that the car almost never just drove forward, but instead kept
# turning left and right very quickly. This also explains the high error yet possible still finishing the course.
# 
# To check if driving forward is rare we will check the data on non 0.5 values. which means slightly going right or left.
# 
# 
# 
# ================
X, Y = algorithm.loadData(print_process=print_process)
algorithm.ShowDataSetStats(X, Y, print_process)

# ================
# Adding polynomial features
# ================
X, Y = algorithm.loadData(print_process=print_process, limit_on_load=False, limiter=2000)
algorithm.ShowPolynomialCurve(X, Y, print_process)

# ================
# lambda
# ================
X, Y = algorithm.loadData(print_process=print_process, limit_on_load=False, limiter=2000)
algorithm.ShowValidationCurve(X, Y, print_process)
