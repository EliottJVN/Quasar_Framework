import numpy as np

class Error:
    def __init__(self, error, Y, Y_true):
        # All possible metrics errors define after this class.
        errors = {
            'mae': mae,
            'mse': mse,
        }
        self.Y = Y
        self.Y_true = Y_true
        self.error = errors.get(error, None)
        # Check if the error is valid.
        if self.error == None:
            raise ValueError("Invalid error type. Please choose from 'mae' or 'mse'.")
    
    def __call__(self):
        error_value, grad_y = self.error(self.Y, self.Y_true)
        return error_value, grad_y


def mae(Y, Y_true):
    """Mean Absolute Error: 1/n * |Y_true - Y| and Gradient of MAE with respect to Y"""
    mae_value = np.mean(np.abs(Y_true - Y))
    mae_gradient = -np.sign(Y_true - Y) / len(Y)
    return mae_value, mae_gradient

def mse(Y, Y_true):
    """Mean Squared Error: 1/n * (Y_true - Y)² and Gradient of MSE with respect to Y"""
    mse_value = np.mean(np.square(Y_true - Y))
    mse_gradient = -2 * (Y_true - Y) / len(Y)
    return mse_value, mse_gradient
