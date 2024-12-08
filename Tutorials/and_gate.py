import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import Network

# imports to create a network from scratch
from layer import Layer
from activationlayer import ActivationLayer

# Example on AND gate
input_dim = 2
output_dim = 1

# AND Gate.
train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_true = np.array([[0], [0], [0], [1]])

# Initialize the model and add layers
my_net = [
    Layer(2, 4),
    ActivationLayer('relu'),
    Layer(4,1),
    ActivationLayer('sigmoid')
]

my_model = Network(my_net)

# Train the model
my_model.train(train_data, Y_true, error='mse', epochs=5000, lr=0.001)

print(my_model.predict(train_data).round())

my_model.save()
file = 'network.json'
model = Network.load(file)

print(model)