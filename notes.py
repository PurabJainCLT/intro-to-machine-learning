# a simple neuron takes an x vector (x1, x2) and takes the dot product against a weight vector (w1, w2)
# from that you get (x1 * w1) + (x2 * w2)
# after that, add a bias (b) to get (x1 * w1) + (x2 * w2) + b
# finally, pass it through an "activation function" to get y = f((x1 * w1) + (x2 * w2) + b)
# the "activation function" turns the bound you get into a predictable curve
# for instance: the sigmoid function below is used commonly
# this entire process is called feeding forward

import numpy as np

def sigmoid(x):
    # activiation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # weight inputs, add the bias, then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1, this is our weight vector
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994 = 1 / (1 + e^(-((2 * 0) + (3 * 1) + 4)))

