# a simple neuron takes an x vector (x1, x2) and takes the dot product against a weight vector (w1, w2)
# from that you get (x1 * w1) + (x2 * w2)
# after that, add a bias (b) to get (x1 * w1) + (x2 * w2) + b
# finally, pass it through an "activation function" to get y = f((x1 * w1) + (x2 * w2) + b)
# the "activation function" turns the bound you get into a predictable curve
# for instance: the sigmoid function above is used commonly
# this entire process is called feeding forward

# the sigmoid function is important because it takes this number ranging from (-inf, inf) to [0, 1]
# in other words, this entire neuron is turning an arbritrary value into a predictable curve like prev. mentioned

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

# a neural network is a bunch of neurons connected together, you have input, hidden, and output layers
# you can have multiple hidden layers, they just need to be between the input and output layers.
# so you might have the input vector (x1, x2), that feeds into both neurons h1 and h2 in the hidden layer 
# and both of those layers finally connected to output layer neuron o1, this makes it a network of neurons
# lets say all of the neurons are feeding forward neurons
# you solve for neurons h1 and h2 (which are equal) using the input vector and then use those values for o1
# below is the same scenario i just described in code

class NeuralNetwork:
  def __init__(self):
    weights = np.array([0, 1])
    bias = 0

    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x) # feeds forward from (x1, x2) vector
    out_h2 = self.h2.feedforward(x)
    # the outputs from h1 and h2 are turned into a new input vector for o1
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.7216325609518421

# now, we can train this neural network to link certain things to numbers, making the computer understand
# for instance, taking two arbritrary values and predicting if this is one thing or another
# a good example is using weight and height to identify if a person is male or female
# you represent male or female as a 1 or 0, and you shift the values by a mean weight and height for ease of calculation
# before training, check the error with the means squared equation using values from our samples and what the neural network returns
# (1/n) * sum from i = 1 to n of (y_true - y_pred)^2
# n is number of samples 
# y is the value being predicted
# y_true is the actual value is (1 or 0 being male or female in this case)
# y_pred is what your neural network returns from o1, aka the predicted value
# therefore, this error is the reciprocal of the number of samples multiplied by the sum of the square of the difference in the true value minus the predicted value
