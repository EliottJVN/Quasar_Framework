import numpy as np
from layer import Layer
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
            layer = Layer(input_dim, output_dim)  # Create a Layer object
            fitted_layers.append(layer)
            input_dim = output_dim  # Update input dimension for the next layer
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
    # Example data
    input_dim = 2
    output_dim = 1
    train_data = np.array([[1, 1]])  # Input data (shape: [1, input_dim])
    Y_true = np.array([[0.5]])       # True output data (shape: [1, output_dim])

    # Initialize the model and add layers
    my_model = Network()
    my_model.add(64)   # First hidden layer with 64 neurons
    my_model.add(output_dim)  # Output layer with 1 neuron

    # Fit the model with the specified input dimension
    my_model.fit(input_dim)

    # Train the model
    my_model.train(train_data, Y_true, error='mse', epochs=100, lr=0.001)

    print(my_model.forward(train_data))

    