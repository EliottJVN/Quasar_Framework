import numpy as np

class Layer:
    def __init__(self, input_layer, output_layer):
        """Create a layer for a neural network."""
        # Define the dimensions of the input and output of the layer.
        self.input_layer = input_layer
        self.output_layer = output_layer

        # Default weights and bias.
        self.W = np.random.normal(loc=0, scale=1, size=(input_layer, output_layer))
        self.B = np.random.normal(loc=0, scale=1, size=(1, output_layer))

        # Gradient weights and bias.
        self.grad_W = np.zeros((input_layer, output_layer))
        self.grad_B = np.zeros((1, output_layer))

    def forward(self, X):
        """Forward propagation."""
        self.X = X
        return X @ self.W + self.B  # Pass to next layer.
    
    def backward(self, grad_Y, lr=0.1):
        """Backward propagation."""
        # Find gardients.
        grad_X = grad_Y @ self.W.T  # Compute gradient for the input.
        self.grad_W = self.X.T @ grad_Y  # Compute gradient for weights.
        self.grad_B = grad_Y.sum(axis=0, keepdims=True)  # Compute gradient for biases.

        # Update weights and biases.
        self.update(lr)
        return grad_X  # Pass to next layer.

    def update(self, lr=0.1):
        """Update weights and biases based on gradients."""
        self.W -= lr * self.grad_W  
        self.B -= lr * self.grad_B     

        # Reset gradients.
        self.grad_W = np.zeros((self.input_layer, self.output_layer))
        self.grad_B = np.zeros((1, self.output_layer))

    
if __name__ == '__main__':
    from error import Error
    
    # Example usage in a neural network training step.
    input_dim = 3
    output_dim = 3

    # Example data
    X = np.ones((1, input_dim))           # Input data
    Y_true = np.array([[0.5, 0.2, 0.1]])  # True output data

    # Create layer and forward pass
    my_neuron = Layer(input_dim, output_dim)
    Y_pred = my_neuron.forward(X)

    # Compute error and gradient with respect to Y
    error = Error('mse', Y_pred, Y_true)
    error_value, grad_Y = error()

    # Backward pass using grad_Y
    X_grad = my_neuron.backward(grad_Y)

    print("Output after forward pass:", Y_pred)
    print("Error value:", error_value)
    print("Gradient with respect to input after backward pass:", X_grad)