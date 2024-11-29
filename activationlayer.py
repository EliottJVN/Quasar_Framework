import numpy as np

class ActivationLayer:
    def __init__(self, activation):
        self.act = activation
        activations = {
            'sigmoid': sigmoid,
            'relu': relu,
            'tanh': tanh,
            'softmax': softmax
        }
        activations_prime = {
            'sigmoid': sigmoid_prime,
            'relu': relu_prime,
            'tanh': tanh_prime,
            'softmax': softmax_prime
        }
        self.activation = activations.get(activation, None)     # get the activation function
        self.activation_prime = activations_prime.get(activation, None)     # get the derivative of the activation function
    
    def __str__(self) -> str:
        return f"Activation Layer: {self.act}"
    
    def forward(self, X):
        self.X = X
        self.Y = self.activation(self.X)
        return self.Y
    
    def backward(self, grad_output, lr):
        # Multiplie le gradient par la dérivée de l'activation
        return grad_output * self.activation_prime(self.X)

    def to_dict(self):
        return {
            "type": "ActivationLayer",
            "activation": self.act
        }
    
    def from_dict(data):
        return ActivationLayer(data['activation'])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    x = sigmoid(x)
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def softmax_prime(x):
    s = softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)  # Jacobian of the softmax