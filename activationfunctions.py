import numpy as np

# Base class for activations
class ActivationFunction:
    @staticmethod
    def activation(x):
        raise NotImplementedError

    @staticmethod
    def derivative(x):
        raise NotImplementedError

# Sigmoid
class Sigmoid(ActivationFunction):
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        s = Sigmoid.activation(x)
        return s * (1 - s)

# ReLU
class ReLU(ActivationFunction):
    @staticmethod
    def activation(x):
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)

# Leaky ReLU
class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def activation(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

# ELU
class ELU(ActivationFunction):
    def __init__(self, alpha=1):
        self.alpha = alpha
    
    def activation(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def derivative(self, x):
        return np.where(x > 0, 1, ELU.activation(x, self.alpha) + self.alpha)

# Swish
class Swish(ActivationFunction):
    @staticmethod
    def activation(x):
        return x / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x * (1 + x * (1 - sigmoid_x))

class Tanh(ActivationFunction):
    @staticmethod
    def activation(x):
        return np.tanh(x)
    
    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x)**2
    
class Softmax(ActivationFunction):
    @staticmethod
    def activation(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def derivative(x):
        return x - np.sum(x * x, axis=1, keepdims=True)
    
class GELU(ActivationFunction):
    @staticmethod
    def activation(x):
        raise NotImplementedError
    @staticmethod
    def derivative(x):
        raise NotImplementedError

class SELU(ActivationFunction):
    @staticmethod
    def activation(x):
        raise NotImplementedError
    
    @staticmethod
    def derivative(x):
        raise NotImplementedError