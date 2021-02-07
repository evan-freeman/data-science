import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.output = np.zeros(y.shape)

        self.input_layers = x.shape[1]
        self.hidden_layers = 2
        self.output_layers = y.shape[1]

        self.input = x
        self.weights1 = np.random.rand(self.input_layers, self.hidden_layers)
        self.weights2 = np.random.rand(self.hidden_layers, self.output_layers)

        # We're setting biases = 0 for now

    def feedforward(self):
        self.layer1 = sigmoid
