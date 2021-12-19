from enum import Enum
import Algorithms as al

algorithm = al.Algorithms()
X, Y = algorithm.loadData(print_process=True, limit_on_load=False, limiter=500)
algorithm.ShowLearningCurve(X, Y, print_process=True)