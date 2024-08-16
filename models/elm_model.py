import numpy as np
import torch

class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_bias = np.random.randn(hidden_size)
        self.output_weights = None
        self.output_size = output_size

    def relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y):
        hidden_layer_output = self.relu(np.dot(X, self.hidden_weights) + self.hidden_bias)
        self.output_weights = np.dot(np.linalg.pinv(hidden_layer_output), y)

    def predict(self, X):
        hidden_layer_output = self.relu(np.dot(X, self.hidden_weights) + self.hidden_bias)
        return np.dot(hidden_layer_output, self.output_weights)
