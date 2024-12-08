import numpy as np
import json

from layer import Layer
from activationlayer import ActivationLayer
from error import Error

class Network:
    def __init__(self, layers:list=None):
        self.unfit_layers = []
        self.layers = []
        
        if not layers is None:
            self.layers = layers
            
    def __str__(self) -> str:
        """Print the Network structure"""
        display = '='*50 + '\n'
        for i, layer in enumerate(self.layers):
            display += f'{i+1}. {layer}\n'
        display += '='*50

        return display

    def add(self, layer):
        """Add a layer to the network."""
        try:
            self.layers.append(layer)
        except:
            raise ValueError("Invalid layer type. Must be Layer (for dense layers) or ActivationLayer (for activation layers).")

    def train(self, train_data, Y_true, error='mse', epochs=100, lr=0.01, save=True):
        """Train the network with the specified data, error function, epochs, and learning rate."""
        for epoch in range(epochs):
            # Forward pass
            Y = self.forward(train_data)

            # Backward pass with error calculation
            loss = self.backward(Y, Y_true, error, lr)
            
            # Output the error value for this epoch
            pad = len(str(epochs))
            print(f"Epoch: {epoch + 1:>{pad}}/{epochs:<{pad}} | Loss {error}: {loss:.6e} | Learning Rate: {lr}")

        self.save() if save else None
    
    def forward(self, data):
        """Forward propagate through all layers of the network."""
        for layer in self.layers:
            data = layer.forward(data)
        return data
    
    def backward(self, Y, Y_true, error, lr):
        """Backward propagate through all layers of the network."""
        # Calculate error and gradient with respect to the final output Y
        error = Error(error, Y, Y_true)
        value_error, grad_Y = error()
        
        # Apply gradient clipping to prevent large gradient updates
        grad_Y = np.clip(grad_Y, -1, 1)

        # Propagate gradient back through each layer
        for layer in reversed(self.layers):
            grad_Y = layer.backward(grad_Y, lr)  # Pass updated grad_Y to the previous layer

        return value_error  # Return error value to monitor during training

    def predict(self, data):
        """Make predictions on the input data."""
        return self.forward(data)
    
    def save(self, filename:str=None):
        """Save the network to a file
        Args:
        filename (str): The filename to save the network to. If None, the network will be
        saved to a file named 'network.json' in the current directory.
        """
        if filename is None:
            filename = 'network.json'
        data = {
            'layers': [layer.to_dict() for layer in self.layers]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(filename:str):
        """Load the network from a file
        Args:
        filename (str): The filename to load the network from.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        model = Network()
        model.layers = [model._create_layer(layer_data) for layer_data in data['layers']]
        return model

    def _create_layer(self, layer_data):
        if layer_data['type'] == 'Layer':
            return Layer.from_dict(layer_data)
        elif layer_data['type'] == 'ActivationLayer':
            return ActivationLayer.from_dict(layer_data)
        else:
            raise ValueError(f"Type de couche inconnu : {layer_data['type']}")
