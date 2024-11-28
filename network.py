import numpy as np
from layer import Layer
from activationlayer import ActivationLayer
from error import Error

class Network:
    def __init__(self):
        self.unfit_layers = []
        self.layers = []
    
    def add(self, layer):
        """Add an uninitialized layer with the specified number of neurons."""
        self.layers.append(layer)

    def fit(self, input_dim):
        # Initialize each layer with input and output dimensions
        fitted_layers = []  # Placeholder for Layer objects
        for output_dim in self.layers:
            if type(output_dim) == int:
                layer = Layer(input_dim, output_dim)  # Create a Layer object
                input_dim = output_dim  # Update input dimension for the next layer
            elif type(output_dim) == str:
                activation = output_dim
                layer = ActivationLayer(activation)  # Create an ActivationLayer object
            
            fitted_layers.append(layer)
        self.layers = fitted_layers  # Replace the original list with actual Layer instances

    def train(self, train_data, Y_true, error='mse', epochs=100, lr=0.01):
        """Train the network with the specified data, error function, epochs, and learning rate."""
        for epoch in range(epochs):
            # Forward pass
            Y = self.forward(train_data)

            # Backward pass with error calculation
            loss = self.backward(Y, Y_true, error, lr)
            
            # Output the error value for this epoch
            pad = len(str(epochs))
            print(f"Epoch: {epoch + 1:>{pad}}/{epochs:<{pad}} | Loss {error}: {loss:.6e} | Learning Rate: {lr}")


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


if __name__ == '__main__':
    # Example on AND gate
    input_dim = 2
    output_dim = 1
    
    # AND Gate.
    train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_true = np.array([[0], [0], [0], [1]])

    # Initialize the model and add layers
    my_model = Network()
    my_model.add(4)   # First hidden layer with 4 neurons
    my_model.add('relu')
    my_model.add(1)   # Output layer with 1 neuron
    my_model.add('sigmoid') 
    
    # Fit the model with the specified input dimension
    my_model.fit(input_dim)

    # Train the model
    my_model.train(train_data, Y_true, error='mse', epochs=5000, lr=0.01)

    print(my_model.forward(train_data).round())

    