import numpy as np
from activationfunctions import *

class ActivationLayer:
    def __init__(self, activation_name, **kwargs):
        self.params = kwargs  # Garde une trace des paramètres
        self.act = activation_name
        
        # Dictionnaire des activations avec paramètres personnalisés
        activations = {
            'sigmoid': Sigmoid(),
            'relu': ReLU(),
            'leakyrelu': LeakyReLU(**kwargs),  # Passe les paramètres comme alpha
            'elu': ELU(**kwargs),
            'swish': Swish(),
            'tanh': Tanh(),
            'softmax': Softmax(),
            'gelu': GELU(),
            'selu': SELU(),
        }
        
        activation_instance = activations.get(activation_name)
        if not activation_instance:
            raise ValueError(f"Activation '{activation_name}' non supportée.")
        
        self.activation = activation_instance.activation
        self.activation_prime = activation_instance.derivative

    def __str__(self):
        return f"Activation Layer: {self.act}, Params: {self.params}"
    
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
            "activation": self.act,
            "params": self.params
        }
    
    def from_dict(data):
        return ActivationLayer(data['activation'], **data['params'])