import numpy as np
import time

import neural_network
import unity_com_server
import main_learn

unity_connection = unity_com_server.UnityConnector()
neural_network = neural_network.NeuralNetwork(21, 10, 1)

while True:
    sensor_values = unity_connection.request_sensor_values()
    steering_value = neural_network.calculate(
        main_learn.nn_params, sensor_values)
    steering_value = 2 * steering_value - 1
    unity_connection.send_steering_value(steering_value)
    time.sleep(0.5)
